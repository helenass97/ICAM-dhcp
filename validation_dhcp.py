import torch
from options_3d import *
from model import MD_multi
import saver_3d
from saver_3d import save_imgs_np, save_imgs
import os
from train_3d import init_dhcp_blob_dataloader, init_biobank_blob_dataloader, init_biobank_dataloader, init_dhcp_dataloader
random_seed = 8
import time
import numpy as np
import json
from argparse import Namespace
from dataloader_utils import *
from sklearn.manifold import TSNE
import seaborn as sns
affine = np.load('affine.npy')
from dataloader_utils import mask_erosion
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, multilabel_confusion_matrix, \
    mean_absolute_error, jaccard_score, average_precision_score


SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def cross_correlation(image, mask):
    "Takes numpy input image and mask"
    mask = mask.detach().cpu().numpy().flatten()
    image = image.detach().cpu().numpy().flatten()
    mask = abs(mask)
    image = abs(image)

    image = (image - np.mean(image)) / (
                np.std(image) * len(image))
    mask = (mask - np.mean(mask)) / (np.std(mask))

    # use the default mode='valid'
    # cross_corr_a = np.mean(np.correlate(np.absolute(image.flatten()), np.absolute(mask.flatten())))
    cross_corr_a = np.mean(np.correlate(image.flatten(), mask.flatten()))

    return cross_corr_a

def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.
    epsilon = 10e-8

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    dice = dice.mean(dim=0)
    dice = torch.clamp(dice, 0, 1.0-epsilon)

    return  dice


def test(opts, override_opts, shuffle_test=False, generate=True, scores=True, tsne=False, group_analysis=False,
         random_gen_group=False, random_gen_extreme_group=False, random_gen=False, interpolation=False,
         cross_corr_thresh=False):
    opts.random_seed = random_seed
    gpu = opts.gpu
    device = opts.device if torch.cuda.is_available() and opts.gpu else 'cpu'

    print('\n--- load dataset ---')
    if opts.data_type == 'dhcp3d':
        _, _, healthy_test_dataloader, _, _, anomaly_test_dataloader = init_dhcp_dataloader(opts, shuffle_test)
    elif opts.data_type == 'dhcp3d_injury':
        healthy_test_dataloader, anomaly_test_dataloader = init_dhcp_injury_dataloader(opts, shuffle_test)
    elif opts.data_type == 'dhcp3d_lesions_test':
        _, _, healthy_test_dataloader, _, _, anomaly_test_dataloader = init_dhcp_lesions_test_dataloader(opts)

    num_iter = len(healthy_test_dataloader)

    resume = opts.resume


    if opts.resume is None:
        print('No experiment to load')
    else:
        with open(resume + '/parameters.json', 'r') as JSON:
            opts_temp = json.load(JSON)
        opts = vars(opts)
        opts.update(opts_temp)
        opts.update(override_opts)
        opts = Namespace(**opts)
        opts.results_path = os.path.join(opts.result_dir, opts.name)
        if not os.path.exists(opts.results_path):
            os.makedirs(opts.results_path)

        print('\n--- load model ---')
        model = MD_multi(opts)
        model.setgpu(device)
        print('Training device: ', device)
        if not gpu:
            _, _, _ = model.resume(opts, os.path.join(resume, opts.model_name), 'cuda:'+str(opts.device), 'cpu')
            opts.device = 'cpu'
        else:
            # print(model.enc_a)
            # print(model.enc_a.fc_reg.weight[0])
            _, _, _ = model.resume(opts=opts, model_dir=os.path.join(resume, opts.model_name),
                                   device_0='cuda:'+str(opts.device), device_1='cuda:'+str(device), train=False,
                                   optim=False)
            opts.device = device
            # print(model.enc_a)
            # print(model.enc_a.fc_reg.weight[0])


    model_name = opts.model_name.split('.')[0]
    test_path = os.path.join(resume, 'test_' + model_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    save_test = {}

    val_accuracy = 0
    val_f1 = 0
    val_precision = 0
    val_recall = 0
    cross_corr_a = 0
    cross_corr_b = 0
    cross_corr_a_std = 0
    cross_corr_b_std = 0
    val_mae = 0
    dice_a_thresh = 0
    dice_b_thresh = 0
    dice_a_std_thresh = 0
    dice_b_std_thresh = 0

    # tsne
    if tsne:
        healthy_val_iter = iter(healthy_test_dataloader)
        z_healthy = np.zeros((len(healthy_test_dataloader), opts.nz))
        z_healthy_labels_age = np.zeros((len(healthy_test_dataloader)))
        z_anomaly_labels_age = np.zeros((len(anomaly_test_dataloader)))
        z_anomaly = np.zeros((len(anomaly_test_dataloader), opts.nz))
        labels_anomaly = np.ones((len(anomaly_test_dataloader)))
        labels_healthy = np.zeros((len(healthy_test_dataloader)))

        for j in range(len(healthy_test_dataloader)):
            print(j)
            healthy_val_images, label, _ = healthy_val_iter.next()
            age=label[0]
            healthy_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            healthy_val_c_org[:, 0] = 1
            images_val = healthy_val_images.to(opts.device).detach()
            c_org_val = healthy_val_c_org.to(opts.device).detach()

            z_temp = model.test_enc_a(images_val, c_org_val)
            z_healthy[j, :] = z_temp.view(z_temp.size(0), -1).detach().cpu().numpy()
            # age labels
            if age >= 40:
                z_healthy_labels_age[j]=0
            elif age >= 37 and age <40:
                z_healthy_labels_age[j]=1

        anomaly_val_iter = iter(anomaly_test_dataloader)
        for j in range(len(anomaly_test_dataloader)):
            print(j)
            anomaly_val_images, label, _ = anomaly_val_iter.next()
            age=label[0]
            anomaly_val_c_org = torch.zeros((anomaly_val_images.size(0), opts.num_domains)).to(opts.device)
            anomaly_val_c_org[:, 1] = 1
            images_val = anomaly_val_images.to(opts.device).detach()
            c_org_val = anomaly_val_c_org.to(opts.device).detach()

            z_temp = model.test_enc_a(images_val, c_org_val)
            z_anomaly[j, :] = z_temp.view(z_temp.size(0), -1).detach().cpu().numpy()
            # age labels
            if age < 37 and age >= 32:
                z_anomaly_labels_age[j]=2
            if age < 32:
                z_anomaly_labels_age[j]=3

        preplexity = [10, 30, 50, 70, 100]

        z_all = np.concatenate((z_healthy, z_anomaly), axis=0)
        labels = np.concatenate((labels_healthy, labels_anomaly), axis=0)
        labels_breakdown_age = np.concatenate((z_healthy_labels_age, z_anomaly_labels_age), axis=0)
        num_unique = len(np.unique(labels_breakdown_age))
        for p in preplexity:
            tsne = TSNE(n_components=2, perplexity=p)
            tsne_results = tsne.fit_transform(z_all)
            df = pd.DataFrame(columns=['x', 'y', 'labels'])
            df['x'] = tsne_results[:, 0]
            df['y'] = tsne_results[:, 1]
            df['labels'] = labels.astype(int)

            plt.figure(figsize=(16, 10))
            sns_plot = sns.scatterplot(
                x="x", y="y",
                hue="labels",
                palette=sns.color_palette("hls", 2),
                data=df,
                legend="full",
                # alpha=0.3
            )
            sns_plot.figure.savefig(os.path.join(test_path, 'tsne' + '_p' + str(p) + '.png'))
            sns_plot.figure.savefig(os.path.join(test_path, 'tsne' + '_p' + str(p) + '.svg'))
            plt.close()

        for p in preplexity:
            tsne = TSNE(n_components=2, perplexity=p)
            tsne_results = tsne.fit_transform(z_all)
            df = pd.DataFrame(columns=['x', 'y', 'labels'])
            df['x'] = tsne_results[:, 0]
            df['y'] = tsne_results[:, 1]
            df['labels'] = labels_breakdown_age.astype(int)

            plt.figure(figsize=(16, 10))
            sns_plot = sns.scatterplot(
                x="x", y="y",
                hue="labels",
                palette=sns.color_palette("hls", num_unique),
                data=df,
                legend="full",
                # alpha=0.3
            )
            sns_plot.figure.savefig(os.path.join(test_path, 'tsne_age_breakdown' + '_p' + str(p) + '.png'))
            sns_plot.figure.savefig(os.path.join(test_path, 'tsne_age_breakdown' + '_p' + str(p) + '.svg'))
            plt.close()

    # scores
    if scores:
        print('scores')
        healthy_val_iter = iter(healthy_test_dataloader)
        anomaly_val_iter = iter(anomaly_test_dataloader)
        val_pred_temp = np.zeros((0))
        val_labels = np.zeros((0))
        val_reg_pred_temp = np.zeros((0))
        val_reg_labels = np.zeros((0))
        cross_corr_temp_a = np.zeros((0))
        cross_corr_temp_b = np.zeros((0))
        for j in range(len(healthy_test_dataloader)):

            if j < len(anomaly_test_dataloader):
                healthy_val_images, reg_label_healthy, _ = healthy_val_iter.next()
                anomaly_val_images, reg_label_anomaly, mask = anomaly_val_iter.next()
                healthy_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
                healthy_val_c_org[:, 0] = 1
                anomaly_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
                anomaly_val_c_org[:, 1] = 1
                images_val = torch.cat((healthy_val_images, anomaly_val_images), dim=0).type(torch.FloatTensor)
                c_org_val = torch.cat((healthy_val_c_org, anomaly_val_c_org), dim=0).type(torch.FloatTensor)
                reg_val = torch.cat((reg_label_healthy[0], reg_label_anomaly[0]), dim=0).type(torch.FloatTensor)

            else:
                healthy_val_images, reg_label_healthy, _ = healthy_val_iter.next()
                healthy_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
                healthy_val_c_org[:, 0] = 1
                images_val = healthy_val_images
                c_org_val = healthy_val_c_org
                reg_val = reg_label_healthy[0].type(torch.FloatTensor)

            images_val = images_val.to(opts.device).detach()
            c_org_val = c_org_val.to(opts.device).detach()
            reg_val = reg_val.to(opts.device).detach()
            mask = mask.to(opts.device).detach()
            print(j)

            if opts.loss_cls_E:
                _, _, pred, reg_pred = model.enc_a.forward(images_val, c_org_val)
                _, y_pred = torch.max(pred, 1)
                _, labels_temp = torch.max(c_org_val, 1)

                val_pred_temp = np.append(val_pred_temp, y_pred.data.cpu().numpy())
                val_labels = np.append(val_labels, labels_temp.data.cpu().numpy())
                val_reg_pred_temp = np.append(val_reg_pred_temp, reg_pred.data.cpu().numpy())
                val_reg_labels = np.append(val_reg_labels, reg_val.data.cpu().numpy())

            if opts.cross_corr:
                cross_corr_a, cross_corr_b = model.cross_correlation(images_val, mask, c_org_val)
                cross_corr_temp_a = np.append(cross_corr_temp_a, cross_corr_a)
                cross_corr_temp_b = np.append(cross_corr_temp_b, cross_corr_b)

        if opts.regression:
            val_mae = mean_absolute_error(val_reg_labels, val_reg_pred_temp)

            np.save(test_path + '/test_age_reg_labels.npy', val_reg_labels)
            np.save(test_path + '/test_age_reg_pred.npy', val_reg_pred_temp)
            save_test['test_mae'] = val_mae

            x, y = line_best_fit(val_reg_labels, val_reg_pred_temp)
            yfit = [x + y * xi for xi in val_reg_labels]
            plt.figure()
            plt.plot(val_reg_labels, val_reg_pred_temp, '+')
            plt.plot(val_reg_labels, yfit, 'k', linewidth=1)
            plt.xlabel('true values')
            plt.ylabel('predicted values')
            plt.title('True vs predicted values plot')
            plt.savefig(test_path + '/val_regression_plot.png')
            plt.close()

            plt.figure()
            plt.plot(val_reg_labels, val_reg_pred_temp, '+')
            plt.plot(val_reg_labels, yfit, 'k', linewidth=1)
            plt.xlabel('true values')
            plt.ylabel('predicted values')
            plt.title('True vs predicted values plot')
            plt.savefig(test_path + '/val_regression_plot.svg', format='svg')
            plt.close()


        if opts.loss_cls_E:
            val_accuracy = accuracy_score(val_pred_temp, val_labels)
            val_f1 = f1_score(val_pred_temp, val_labels, average='macro')
            val_precision = precision_score(val_pred_temp, val_labels, average='macro')
            val_recall = recall_score(val_pred_temp, val_labels, average='macro')

            np.save(test_path + '/test_age_class_labels.npy', val_labels)
            np.save(test_path + '/test_age_class_pred.npy', val_pred_temp)

            save_test['test_accuracy'] = val_accuracy
            save_test['test_f1'] = val_f1
            save_test['test_precision'] = val_precision
            save_test['test_recall'] = val_recall

        if opts.cross_corr:
            cross_corr_a = np.mean(cross_corr_temp_a)
            cross_corr_b = np.mean(cross_corr_temp_b)
            cross_corr_a_std = np.std(cross_corr_temp_a)
            cross_corr_b_std = np.std(cross_corr_temp_b)

            save_test['test_cross_corr_a'] = cross_corr_a
            save_test['test_cross_corr_a_std'] = cross_corr_a_std
            save_test['test_cross_corr_b'] = cross_corr_b
            save_test['test_cross_corr_b_std'] = cross_corr_b_std

        with open(os.path.join(test_path, 'test_results.json'), 'w') as file:
            json.dump(save_test, file, indent=4, sort_keys=True)


    if cross_corr_thresh and opts.cross_corr:
        df = pd.DataFrame(
            columns=["experiment", "threshold",
                     "dice_a", "dice_a_std",
                     "dice_b", "dice_b_std",
                     "f1_score_a", "f1_score_a_std",
                     "f1_score_b", "f1_score_b_std",
                     "recall_a", "recall_a_std",
                     "recall_b", "recall_b_std",
                     "precision_a", "precision_a_std",
                     "precision_b", "precision_b_std",
                     "iou_a", "iou_a_std",
                     "iou_b", "iou_b_std",
                     ])

        thresholds = [0, 0.01, 0.02]

        for i, thresh in enumerate(thresholds):
            print('thresh: ', thresh)
            dice_temp_a = np.zeros((0))
            dice_temp_b = np.zeros((0))
            f1_temp_a = np.zeros((0))
            f1_temp_b = np.zeros((0))
            precision_temp_a = np.zeros((0))
            precision_temp_b = np.zeros((0))
            recall_temp_a = np.zeros((0))
            recall_temp_b = np.zeros((0))
            iou_temp_a = np.zeros((0))
            iou_temp_b = np.zeros((0))

            healthy_val_iter = iter(healthy_test_dataloader)
            anomaly_val_iter = iter(anomaly_test_dataloader)
            for j in range(len(healthy_test_dataloader)):
                healthy_val_images, reg_label_healthy, mask_healthy = healthy_val_iter.next()
                images_val = healthy_val_images

                images_val = images_val.to(opts.device).detach()
                mask = mask_healthy.to(opts.device).detach()
                print(j)
                c_org_trans = torch.zeros((images_val.size(0), opts.num_domains)).to(opts.device)
                c_org_trans[:, 1] = 1

                _, diff_m_pos_mean, _, _, _ = model.test_forward_random_group(images_val, c_org_trans, num=1)

                if opts.cross_corr_pos_mask == 'True':
                    diff_m_pos_mean = diff_m_pos_mean * (diff_m_pos_mean > 0).type(torch.float)
                elif opts.cross_corr_pos_mask == 'False':
                    diff_m_pos_mean = diff_m_pos_mean * (diff_m_pos_mean < 0).type(torch.float)

                diff_m_pos_mean = abs(diff_m_pos_mean) * (abs(diff_m_pos_mean) > thresh).type(torch.float)

                mask = (mask>0).type(torch.float)

                dice_a = dice_coeff(diff_m_pos_mean, mask).item()
                mask = mask.detach().cpu().numpy().flatten()

                diff_m_pos_mean = (diff_m_pos_mean>0).type(torch.float)
                diff_m_pos_mean = diff_m_pos_mean.detach().cpu().numpy().flatten()

                intersection = np.logical_and(mask, diff_m_pos_mean)
                union = np.logical_or(mask, diff_m_pos_mean)
                iou_score = np.sum(intersection) / np.sum(union)

                f1 = f1_score(mask, diff_m_pos_mean)
                precision = precision_score(mask, diff_m_pos_mean)
                recall = recall_score(mask, diff_m_pos_mean)
                # jaccard = jaccard_score(mask, diff_m_pos_mean)
                dice_temp_a = np.append(dice_temp_a, dice_a)
                f1_temp_a = np.append(f1_temp_a, f1)
                precision_temp_a = np.append(precision_temp_a, precision)
                recall_temp_a = np.append(recall_temp_a, recall)
                iou_temp_a = np.append(iou_temp_a, iou_score)

            for j in range(len(anomaly_test_dataloader)):
                anomaly_val_images, reg_label_anomaly, mask_anomaly = anomaly_val_iter.next()
                images_val = anomaly_val_images

                images_val = images_val.to(opts.device).detach()
                mask = mask_anomaly.to(opts.device).detach()
                print(j)
                c_org_trans = torch.zeros((images_val.size(0), opts.num_domains)).to(opts.device)
                c_org_trans[:, 1] = 1

                _, diff_m_pos_mean, _, _, _ = model.test_forward_random_group(images_val, c_org_trans, num=1)

                if opts.cross_corr_pos_mask is not None:
                    if opts.cross_corr_pos_mask:
                        diff_m_pos_mean = diff_m_pos_mean * (diff_m_pos_mean > 0).type(torch.float)
                    else:
                        diff_m_pos_mean = diff_m_pos_mean * (diff_m_pos_mean < 0).type(torch.float)

                diff_m_pos_mean = abs(diff_m_pos_mean) * (abs(diff_m_pos_mean) > thresh).type(torch.float)

                mask = (mask>0).type(torch.float)

                dice_b = dice_coeff(diff_m_pos_mean, mask).item()
                mask = mask.detach().cpu().numpy().flatten()

                diff_m_pos_mean = (diff_m_pos_mean>0).type(torch.float)
                diff_m_pos_mean = diff_m_pos_mean.detach().cpu().numpy().flatten()

                intersection = np.logical_and(mask, diff_m_pos_mean)
                union = np.logical_or(mask, diff_m_pos_mean)
                iou_score = np.sum(intersection) / np.sum(union)

                f1 = f1_score(mask, diff_m_pos_mean)
                precision = precision_score(mask, diff_m_pos_mean)
                recall = recall_score(mask, diff_m_pos_mean)
                # jaccard = jaccard_score(mask, diff_m_pos_mean)
                dice_temp_b = np.append(dice_temp_b, dice_b)
                f1_temp_b = np.append(f1_temp_b, f1)
                precision_temp_b = np.append(precision_temp_b, precision)
                recall_temp_b = np.append(recall_temp_b, recall)
                iou_temp_b = np.append(iou_temp_b, iou_score)


            dice_raw = np.concatenate((dice_temp_a, dice_temp_b))
            f1_raw = np.concatenate((f1_temp_a, f1_temp_b))
            recall_raw = np.concatenate((recall_temp_a, recall_temp_b))
            precision_raw = np.concatenate((precision_temp_a, precision_temp_b))
            iou_raw = np.concatenate((iou_temp_a, iou_temp_b))

            if opts.cross_corr_pos_mask == 'True':
                save_path = '/pos_mask'
            elif opts.cross_corr_pos_mask == 'False':
                save_path = '/neg_mask'
            else:
                save_path = '/all_mask'


            np.save(test_path + save_path + '/test_lesion_mask_dice_thresh_' + str(thresh) + '.npy', dice_raw)
            np.save(test_path + save_path + '/test_lesion_mask_f1_thresh_' + str(thresh) + '.npy', f1_raw)
            np.save(test_path + save_path + '/test_lesion_mask_recall_thresh_' + str(thresh) + '.npy', recall_raw)
            np.save(test_path + save_path + '/test_lesion_mask_precision_thresh_' + str(thresh) + '.npy', precision_raw)
            np.save(test_path + save_path + '/test_lesion_mask_iou_thresh_' + str(thresh) + '.npy', iou_raw)

            dice_a = np.mean(dice_temp_a)
            dice_a_std = np.std(dice_temp_a)
            dice_b = np.mean(dice_temp_b)
            dice_b_std = np.std(dice_temp_b)

            f1_a = np.mean(f1_temp_a)
            f1_a_std = np.std(f1_temp_b)
            f1_b = np.mean(f1_temp_b)
            f1_b_std = np.std(f1_temp_b)

            recall_a = np.mean(recall_temp_a)
            recall_a_std = np.std(recall_temp_a)
            recall_b = np.mean(recall_temp_b)
            recall_b_std = np.std(recall_temp_b)

            precision_a = np.mean(precision_temp_a)
            precision_a_std = np.std(precision_temp_a)
            precision_b = np.mean(precision_temp_b)
            precision_b_std = np.std(precision_temp_b)

            iou_a = np.mean(iou_temp_a)
            iou_a_std = np.std(iou_temp_a)
            iou_b = np.mean(iou_temp_b)
            iou_b_std = np.std(iou_temp_b)


            df.loc[i, :] = resume, thresh, \
                           dice_a, dice_a_std, \
                           dice_b, dice_b_std, \
                           f1_a, f1_a_std, \
                           f1_b, f1_b_std, \
                           recall_a, recall_a_std, \
                           recall_b, recall_b_std, \
                           precision_a, precision_a_std, \
                           precision_b, precision_b_std, \
                           iou_a, iou_a_std, \
                           iou_b, iou_b_std

        results_name = 'results_model_lesions'

        if opts.cross_corr_pos_mask == 'True':
            results_name = 'pos_mask/' + results_name + '_pos_pred'
        elif opts.cross_corr_pos_mask == 'False':
            results_name = 'neg_mask/' + results_name + '_neg_pred'
        else:
            results_name = 'all_mask/' + results_name


        df.to_csv(os.path.join(test_path, results_name + '.csv'))

    # generations for injury subjects
    if generate:
        anomaly_test_iter = iter(anomaly_test_dataloader)
        num = np.minimum(opts.num, len(anomaly_test_dataloader))
        for idx, (anomaly_images, labels_a, _) in enumerate(anomaly_test_iter):
            id_a = labels_a[1][0]
            c_org_trans = torch.zeros((anomaly_images.size(0), opts.num_domains)).to(opts.device)
            c_org_trans[:, 1] = 1

            c_org_recon = torch.zeros((anomaly_images.size(0), opts.num_domains)).to(opts.device)
            c_org_recon[:, 0] = 1

            images = anomaly_images.to(opts.device).detach()
            c_org_trans = c_org_trans.to(opts.device).detach()
            c_org_recon = c_org_recon.to(opts.device).detach()
            with torch.no_grad():
                output_b, _, _, _, _ = model.test_forward_random_group(images, c_org_trans, num=1)
                output_a, _, _, _, _ = model.test_forward_random_group(images, c_org_recon, num=1)

            input_a =  anomaly_images.clone().cpu().numpy()[0, 0, :, :, :]

            output_b =  output_b.cpu().numpy()[0, 0, :, :, :]
            output_a =  output_a.cpu().numpy()[0, 0, :, :, :]
            imgs_a = [input_a, output_b, output_a]
            names_a = ['anomaly_input_' + id_a, 'anomaly_output_trans_' + id_a, 'anomaly_output_recon_'+ id_a]
            save_imgs_np(imgs_a, names_a, test_path+'/generate')
            print(idx)


    # group comparison by sex/ age
    if group_analysis:

        anomaly_test_iter = iter(healthy_test_dataloader)
        num = np.minimum(opts.num, len(anomaly_test_dataloader))
        for idx, (anomaly_images, labels_a, _) in enumerate(anomaly_test_iter):
            if idx > num:
                break
            label_a = labels_a[0].numpy()[0]
            id_a = labels_a[1][0]
            diff_map_neg = np.zeros((1, anomaly_images.size(2), anomaly_images.size(3), anomaly_images.size(4)))
            diff_map_pos = np.zeros((1, anomaly_images.size(2), anomaly_images.size(3), anomaly_images.size(4)))
            outputs_b = np.zeros((1, anomaly_images.size(2), anomaly_images.size(3), anomaly_images.size(4)))
            healthy_test_iter = iter(anomaly_test_dataloader)
            for idx_h, (healthy_images, labels_b, _) in enumerate(healthy_test_iter):
                label_b = labels_b[0].numpy()[0]

                if not label_a == label_b:
                    continue

                healthy_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
                healthy_c_org[:, 1] = 1
                anomaly_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
                anomaly_c_org[:, 0] = 1
                images = torch.cat((healthy_images, anomaly_images), dim=0).type(torch.FloatTensor)
                c_org = torch.cat((healthy_c_org, anomaly_c_org), dim=0).type(torch.FloatTensor)
                images = images.to(opts.device).detach()
                c_org = c_org.to(opts.device).detach()
                with torch.no_grad():
                    output_b, diff_b_pos, diff_b_neg, _, _, _ = model.test_forward_transfer(images, c_org)

                output_b = output_b.cpu().numpy()[0, :, :, :, :]
                diff_b_pos = diff_b_pos.cpu().numpy()[0, :, :, :, :]
                diff_b_neg = diff_b_neg.cpu().numpy()[0, :, :, :, :]

                diff_map_pos = np.append(diff_map_pos, diff_b_pos, axis=0)
                diff_map_neg = np.append(diff_map_neg, diff_b_neg, axis=0)
                outputs_b = np.append(outputs_b, output_b, axis=0)

                # if outputs_b.shape[0] > 5:
                #     break

            diff_map_pos = np.mean(diff_map_pos[1:,:,:,:], axis=0)
            diff_map_neg = np.mean(diff_map_neg[1:,:,:,:], axis=0)
            outputs_b = np.mean(outputs_b[1:,:,:,:], axis=0)
            input_a =  anomaly_images.cpu().numpy()[0, 0, :, :, :]
            imgs_a = [input_a, outputs_b, diff_map_pos, diff_map_neg]
            names_a = ['input_a', 'mean_output_b', 'mean_diff_map_pos', 'mean_diff_map_neg']
            save_imgs_np(imgs_a, names_a, os.path.join(test_path, '{}_{}'.format('group_diff_heal_to_ano_label_' +str(label_a)+ '_'+ id_a, str(idx))))


        anomaly_test_iter = iter(anomaly_test_dataloader)
        num = np.minimum(opts.num, len(anomaly_test_dataloader))
        for idx, (anomaly_images, labels_a, _) in enumerate(anomaly_test_iter):
            if idx > num:
                break
            label_a = labels_a[0].numpy()[0]
            id_a = labels_a[1][0]
            diff_map_neg = np.zeros((1, anomaly_images.size(2), anomaly_images.size(3), anomaly_images.size(4)))
            diff_map_pos = np.zeros((1, anomaly_images.size(2), anomaly_images.size(3), anomaly_images.size(4)))
            outputs_b = np.zeros((1, anomaly_images.size(2), anomaly_images.size(3), anomaly_images.size(4)))
            healthy_test_iter = iter(healthy_test_dataloader)
            for idx_h, (healthy_images, labels_b, _) in enumerate(healthy_test_iter):
                label_b = labels_b[0].numpy()[0]

                if not label_a == label_b:
                    continue

                healthy_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
                healthy_c_org[:, 0] = 1
                anomaly_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
                anomaly_c_org[:, 1] = 1
                images = torch.cat((healthy_images, anomaly_images), dim=0).type(torch.FloatTensor)
                c_org = torch.cat((healthy_c_org, anomaly_c_org), dim=0).type(torch.FloatTensor)
                images = images.to(opts.device).detach()
                c_org = c_org.to(opts.device).detach()
                with torch.no_grad():
                    output_b, diff_b_pos, diff_b_neg, _, _, _ = model.test_forward_transfer(images, c_org)

                output_b = output_b.cpu().numpy()[0, :, :, :, :]
                diff_b_pos = diff_b_pos.cpu().numpy()[0, :, :, :, :]
                diff_b_neg = diff_b_neg.cpu().numpy()[0, :, :, :, :]

                diff_map_pos = np.append(diff_map_pos, diff_b_pos, axis=0)
                diff_map_neg = np.append(diff_map_neg, diff_b_neg, axis=0)
                outputs_b = np.append(outputs_b, output_b, axis=0)

            diff_map_pos = np.mean(diff_map_pos[1:,:,:,:], axis=0)
            diff_map_neg = np.mean(diff_map_neg[1:,:,:,:], axis=0)
            outputs_b = np.mean(outputs_b[1:,:,:,:], axis=0)
            input_a =  anomaly_images.cpu().numpy()[0, 0, :, :, :]
            imgs_a = [input_a, outputs_b, diff_map_pos, diff_map_neg]
            names_a = ['input_a', 'mean_output_b', 'mean_diff_map_pos', 'mean_diff_map_neg']
            save_imgs_np(imgs_a, names_a, os.path.join(test_path, '{}_{}'.format('group_diff_ano_to_heal_label_' +str(label_a)+ '_'+ id_a, str(idx))))

    if random_gen_group:
        anomaly_test_iter = iter(anomaly_test_dataloader)
        num = np.minimum(opts.num, len(anomaly_test_dataloader))
        for idx, (anomaly_images, labels_a, _) in enumerate(anomaly_test_iter):
            print(idx)
            age = labels_a[0].numpy()[0]
            id_a = labels_a[1][0]

            c_org_trans = torch.zeros((anomaly_images.size(0), opts.num_domains)).to(opts.device)
            c_org_trans[:, 1] = 1

            c_org_recon = torch.zeros((anomaly_images.size(0), opts.num_domains)).to(opts.device)
            c_org_recon[:, 0] = 1

            images = anomaly_images.to(opts.device).detach()
            c_org_trans = c_org_trans.to(opts.device).detach()
            c_org_recon = c_org_recon.to(opts.device).detach()

            # for i in range(10):
            # mean_image = np.zeros((10, images.size(2), images.size(3), images.size(4)))
            # diff_image = np.zeros((10, images.size(2), images.size(3), images.size(4)))
            # var_image = np.zeros((10, images.size(2), images.size(3), images.size(4)))

            if opts.regression:
                _, _, pred, reg_pred = model.enc_a.forward(images, c_org_trans)
                reg_pred = reg_pred.detach().cpu().numpy()[0][0]
            else:
                reg_pred = 0

            with torch.no_grad():
                output_b, diff_b_pos, diff_b_neg, diff_b_pos_std, diff_b_neg_std = model.test_forward_random_group(images, c_org_trans, num=100)
                output_a, diff_a_pos, diff_a_neg, diff_a_pos_std, diff_a_neg_std = model.test_forward_random_group(images, c_org_recon, num=100)

            diff_b_a_pos = diff_b_pos-diff_a_pos
            diff_b_a_neg = diff_b_neg-diff_a_neg

            input_a =  anomaly_images.clone().cpu().numpy()[0, 0, :, :, :]

            anomaly_images[anomaly_images>0] = 1
            mask = anomaly_images.cpu().numpy()[0, 0, :, :, :]

            # mask = mask_erosion(mask, it=3)

            output_b =  output_b.cpu().numpy()[0, 0, :, :, :]
            output_a =  output_a.cpu().numpy()[0, 0, :, :, :]
            diff_b_pos =  diff_b_pos.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_neg =  diff_b_neg.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_pos =  diff_a_pos.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_neg =  diff_a_neg.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_pos_std =  diff_a_pos_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_neg_std =  diff_a_neg_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_pos_std =  diff_b_pos_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_neg_std =  diff_b_neg_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_a_pos =  diff_b_a_pos.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_a_neg = diff_b_a_neg.cpu().numpy()[0, 0, :, :, :]*mask
            # mean_image[i] = output_b
            # diff_image[i] = diff_b_pos
            # var_image[i] = diff_b_pos_std
            imgs_a = [input_a, output_b, output_a, diff_b_pos, diff_b_neg, diff_b_pos_std, diff_b_neg_std, diff_a_pos, diff_a_neg, diff_a_pos_std, diff_a_neg_std, diff_b_a_pos, diff_b_a_neg]
            names_a = ['input_a', 'mean_output_trans', 'mean_output_recon',
                      'mean_diff_map_trans_pos',  'mean_diff_map_trans_neg',
                       'mean_diff_map_trans_pos_std',  'mean_diff_map_trans_neg_std',
                       'mean_diff_map_recon_pos', 'mean_diff_map_recon_neg' ,
                       'mean_diff_map_recon_pos_std', 'mean_diff_map_recon_neg_std',
                       'mean_diff_map_class_pos', 'mean_diff_map_class_neg']
            save_imgs_np(imgs_a, names_a, os.path.join(test_path, '{}'.format('random_group_ano_to_heal_label_' + id_a+'_age_'+str(age[0])+'_pred_'+str(reg_pred))))

            # mean_image = np.mean(mean_image, axis=0)
            # diff_image = np.std(diff_image, axis=0)
            # var_image = np.std(var_image, axis=0)
            #
            # imgs_a = [mean_image, diff_image, var_image]
            # names_a = ['mean_image', 'dif_image','var_image']
            # save_imgs_np(imgs_a, names_a, os.path.join(test_path, '{}'.format(id_a+'_summary')))


        healthy_test_iter = iter(healthy_test_dataloader)
        num = np.minimum(opts.num, len(healthy_test_dataloader))
        for idx, (healthy_images, labels_a, _) in enumerate(healthy_test_iter):
            print(idx)
            age = labels_a[0].numpy()[0]
            id_a = labels_a[1][0]

            c_org_trans = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
            c_org_trans[:, 0] = 1

            c_org_recon = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
            c_org_recon[:, 1] = 1

            images = healthy_images.to(opts.device).detach()
            c_org_trans = c_org_trans.to(opts.device).detach()
            c_org_recon = c_org_recon.to(opts.device).detach()

            if opts.regression:
                _, _, pred, reg_pred = model.enc_a.forward(images, c_org_trans)
                reg_pred = reg_pred.detach().cpu().numpy()[0][0]
            else:
                reg_pred = 0

            with torch.no_grad():
                output_b, diff_b_pos, diff_b_neg, diff_b_pos_std, diff_b_neg_std = model.test_forward_random_group(images, c_org_trans, num=100)
                output_a, diff_a_pos, diff_a_neg, diff_a_pos_std, diff_a_neg_std = model.test_forward_random_group(images, c_org_recon, num=100)

            diff_b_a_pos = diff_b_pos-diff_a_pos
            diff_b_a_neg = diff_b_neg-diff_a_neg

            input_a =  healthy_images.clone().cpu().numpy()[0, 0, :, :, :]

            healthy_images[healthy_images>0] = 1
            mask = healthy_images.cpu().numpy()[0, 0, :, :, :]

            # mask = mask_erosion(mask, it=3)

            output_b =  output_b.cpu().numpy()[0, 0, :, :, :]
            output_a =  output_a.cpu().numpy()[0, 0, :, :, :]
            diff_b_pos =  diff_b_pos.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_neg =  diff_b_neg.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_pos =  diff_a_pos.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_neg =  diff_a_neg.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_pos_std =  diff_a_pos_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_neg_std =  diff_a_neg_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_pos_std =  diff_b_pos_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_neg_std =  diff_b_neg_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_a_pos =  diff_b_a_pos.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_a_neg = diff_b_a_neg.cpu().numpy()[0, 0, :, :, :]*mask
            imgs_a = [input_a, output_b, output_a, diff_b_pos, diff_b_neg, diff_b_pos_std, diff_b_neg_std, diff_a_pos, diff_a_neg, diff_a_pos_std, diff_a_neg_std, diff_b_a_pos, diff_b_a_neg]
            names_a = ['input_a', 'mean_output_trans', 'mean_output_recon',
                      'mean_diff_map_trans_pos',  'mean_diff_map_trans_neg',
                       'mean_diff_map_trans_pos_std',  'mean_diff_map_trans_neg_std',
                       'mean_diff_map_recon_pos', 'mean_diff_map_recon_neg' ,
                       'mean_diff_map_recon_pos_std', 'mean_diff_map_recon_neg_std',
                       'mean_diff_map_class_pos', 'mean_diff_map_class_neg']
            save_imgs_np(imgs_a, names_a, os.path.join(test_path, '{}'.format('random_group_heal_to_ano_label_' + id_a+'_age_'+str(age[0])+'_pred_'+str(reg_pred))))

    if random_gen_extreme_group:
        anomaly_test_iter = iter(anomaly_test_dataloader)
        num = len(anomaly_test_dataloader)
        for idx, (anomaly_images, labels_a, _) in enumerate(anomaly_test_iter):
            if idx > num:
                break
            label_a = labels_a[0].numpy()[0]
            age = labels_a[2].numpy()[0]

            if age < 78:
                continue
            id_a = labels_a[1][0]
            c_org_trans = torch.zeros((anomaly_images.size(0), opts.num_domains)).to(opts.device)
            c_org_trans[:, 1] = 1

            c_org_recon = torch.zeros((anomaly_images.size(0), opts.num_domains)).to(opts.device)
            c_org_recon[:, 0] = 1

            images = anomaly_images.to(opts.device).detach()
            c_org_trans = c_org_trans.to(opts.device).detach()
            c_org_recon = c_org_recon.to(opts.device).detach()
            with torch.no_grad():
                output_b, diff_b_pos, diff_b_neg, diff_b_pos_std, diff_b_neg_std = model.test_forward_random_group(images, c_org_trans, num=100)
                output_a, diff_a_pos, diff_a_neg, diff_a_pos_std, diff_a_neg_std = model.test_forward_random_group(images, c_org_recon, num=100)

            diff_b_a_pos = diff_b_pos-diff_a_pos
            diff_b_a_neg = diff_b_neg-diff_a_neg

            input_a =  anomaly_images.clone().cpu().numpy()[0, 0, :, :, :]

            anomaly_images[anomaly_images>0] = 1
            mask = anomaly_images.cpu().numpy()[0, 0, :, :, :]

            # mask = mask_erosion(mask)

            output_b =  output_b.cpu().numpy()[0, 0, :, :, :]
            output_a =  output_a.cpu().numpy()[0, 0, :, :, :]
            diff_b_pos =  diff_b_pos.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_neg =  diff_b_neg.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_pos =  diff_a_pos.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_neg =  diff_a_neg.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_pos_std =  diff_a_pos_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_neg_std =  diff_a_neg_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_pos_std =  diff_b_pos_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_neg_std =  diff_b_neg_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_a_pos =  diff_b_a_pos.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_a_neg = diff_b_a_neg.cpu().numpy()[0, 0, :, :, :]*mask
            imgs_a = [input_a, output_b, output_a, diff_b_pos, diff_b_neg, diff_b_pos_std, diff_b_neg_std, diff_a_pos, diff_a_neg, diff_a_pos_std, diff_a_neg_std, diff_b_a_pos, diff_b_a_neg]
            names_a = ['input_a', 'mean_output_trans', 'mean_output_recon',
                      'mean_diff_map_trans_pos',  'mean_diff_map_trans_neg',
                       'mean_diff_map_trans_pos_std',  'mean_diff_map_trans_neg_std',
                       'mean_diff_map_recon_pos', 'mean_diff_map_recon_neg' ,
                       'mean_diff_map_recon_pos_std', 'mean_diff_map_recon_neg_std',
                       'mean_diff_map_class_pos', 'mean_diff_map_class_neg']
            save_imgs_np(imgs_a, names_a, os.path.join(test_path, '{}_{}'.format('random_group_diff_very_old_to_young_label_' + id_a+'_age_'+str(age), str(idx))))

        healthy_test_iter = iter(healthy_test_dataloader)
        num = len(healthy_test_dataloader)
        for idx, (healthy_images, labels_a, _) in enumerate(healthy_test_iter):
            if idx > num:
                break
            label_a = labels_a[0].numpy()[0]
            age = labels_a[2].numpy()[0]

            if age > 51:
                continue

            id_a = labels_a[1][0]
            c_org_trans = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
            c_org_trans[:, 0] = 1

            c_org_recon = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
            c_org_recon[:, 1] = 1

            images = healthy_images.to(opts.device).detach()
            c_org_trans = c_org_trans.to(opts.device).detach()
            c_org_recon = c_org_recon.to(opts.device).detach()
            with torch.no_grad():
                output_b, diff_b_pos, diff_b_neg, diff_b_pos_std, diff_b_neg_std = model.test_forward_random_group(images, c_org_trans, num=100)
                output_a, diff_a_pos, diff_a_neg, diff_a_pos_std, diff_a_neg_std = model.test_forward_random_group(images, c_org_recon, num=100)

            diff_b_a_pos = diff_b_pos-diff_a_pos
            diff_b_a_neg = diff_b_neg-diff_a_neg

            input_a =  healthy_images.clone().cpu().numpy()[0, 0, :, :, :]

            healthy_images[healthy_images>0] = 1
            mask = healthy_images.cpu().numpy()[0, 0, :, :, :]

            # mask = mask_erosion(mask)

            output_b =  output_b.cpu().numpy()[0, 0, :, :, :]
            output_a =  output_a.cpu().numpy()[0, 0, :, :, :]
            diff_b_pos =  diff_b_pos.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_neg =  diff_b_neg.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_pos =  diff_a_pos.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_neg =  diff_a_neg.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_pos_std =  diff_a_pos_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_a_neg_std =  diff_a_neg_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_pos_std =  diff_b_pos_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_neg_std =  diff_b_neg_std.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_a_pos =  diff_b_a_pos.cpu().numpy()[0, 0, :, :, :]*mask
            diff_b_a_neg = diff_b_a_neg.cpu().numpy()[0, 0, :, :, :]*mask
            imgs_a = [input_a, output_b, output_a, diff_b_pos, diff_b_neg, diff_b_pos_std, diff_b_neg_std, diff_a_pos, diff_a_neg, diff_a_pos_std, diff_a_neg_std, diff_b_a_pos, diff_b_a_neg]
            names_a = ['input_a', 'mean_output_trans', 'mean_output_recon',
                      'mean_diff_map_trans_pos',  'mean_diff_map_trans_neg',
                       'mean_diff_map_trans_pos_std',  'mean_diff_map_trans_neg_std',
                       'mean_diff_map_recon_pos', 'mean_diff_map_recon_neg' ,
                       'mean_diff_map_recon_pos_std', 'mean_diff_map_recon_neg_std',
                       'mean_diff_map_class_pos', 'mean_diff_map_class_neg']
            save_imgs_np(imgs_a, names_a, os.path.join(test_path, '{}_{}'.format('random_group_diff_very_young_to_old_label_' + id_a+'_age_'+str(age), str(idx))))


    if random_gen:
        domains = ['healthy', 'anomaly']
        loaders = [healthy_test_dataloader, anomaly_test_dataloader]
        for d in range(opts.num_domains):
            for idx, data in enumerate(loaders[d]):
                # break
                img, id, _ = data
                id = id[1][0]
                print('{}/{}'.format(idx, len(loaders[d])))
                if idx > 5:
                    break;
                img = img.to(opts.device)
                img_temp = img
                img_temp[img_temp > 1] = 1
                img_temp[img_temp < 0] = 0
                imgs = [img_temp]
                names = ['input']
                for idx2 in range(5):
                    for i in range(opts.num_domains):
                        if domains[i] == 'healthy':
                            c_org = torch.zeros((img.size(0), opts.num_domains)).to(opts.device)
                            c_org[:, 0] = 1
                        elif domains[i] == 'anomaly':
                            c_org = torch.zeros((img.size(0), opts.num_domains)).to(opts.device)
                            c_org[:, 1] = 1
                        with torch.no_grad():
                            imgs_ = model.test_forward_random(img, c_org)
                        imgs.append(imgs_)
                        names.append('output_{}_{}_{}'.format(domains[d], domains[i], id+'_'+str(idx2)))
                save_imgs(imgs, names, os.path.join(test_path, '{}_{}'.format(domains[d], id+'_'+str(idx))))

    if interpolation:
        # interpolation between classes
        healthy_test_iter = iter(healthy_test_dataloader)
        anomaly_test_iter = iter(anomaly_test_dataloader)
        num = np.minimum(opts.num, len(anomaly_test_dataloader))
        for idx in range(num):
            healthy_images, id_a, _ = healthy_test_iter.next()
            anomaly_images, id_b, _ = anomaly_test_iter.next()

            id_a = id_a[1][0]
            id_b = id_b[1][0]

            healthy_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
            healthy_c_org[:, 0] = 1
            anomaly_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
            anomaly_c_org[:, 1] = 1
            images = torch.cat((healthy_images, anomaly_images), dim=0).type(torch.FloatTensor)
            c_org = torch.cat((healthy_c_org, anomaly_c_org), dim=0).type(torch.FloatTensor)
            images = images.to(opts.device).detach()
            c_org = c_org.to(opts.device).detach()
            half_size = images.size(0) // 2
            imgs_a, imgs_b = torch.split(images, half_size, dim=0)
            imgs_b[imgs_b > 1] = 1
            imgs_b[imgs_b < 0] = 0
            imgs_a = [imgs_a]
            imgs_b = [imgs_b]
            diff_map_a_pos = []
            diff_map_b_pos = []
            diff_map_a_neg = []
            diff_map_b_neg = []

            with torch.no_grad():
                outputs_a, diff_map_a_pos_, diff_map_a_neg_, outputs_b, diff_map_b_pos_, diff_map_b_neg_ = model.test_interpolation(images, c_org)
            imgs_a = imgs_a + outputs_a
            imgs_b = imgs_b + outputs_b
            diff_map_a_pos = diff_map_a_pos + diff_map_a_pos_
            diff_map_b_pos = diff_map_b_pos + diff_map_b_pos_
            diff_map_a_neg = diff_map_a_neg + diff_map_a_neg_
            diff_map_b_neg = diff_map_b_neg + diff_map_b_neg_

            names_a = ['input_a']
            names_b = ['input_b']
            names_map_a_pos = []
            names_map_b_pos = []
            names_map_a_neg = []
            names_map_b_neg = []
            for n in range(10):
                names_a.append('img_0to1_' + str(n))
                names_b.append('img_1to0_' + str(n))
            for n in range(10):
                names_map_a_pos.append('diff_map_0to1_pos_' + str(n))
                names_map_b_pos.append('diff_map_1to0_pos_' + str(n))
                names_map_a_neg.append('diff_map_0to1_neg_' + str(n))
                names_map_b_neg.append('diff_map_1to0_neg_' + str(n))

            save_imgs(imgs_a, names_a, os.path.join(test_path, '{}_{}'.format('interpolation_between_'+id_a+'_'+id_b, str(idx))))
            save_imgs(imgs_b, names_b, os.path.join(test_path, '{}_{}'.format('interpolation_between_'+id_a+'_'+id_b, str(idx))))
            save_imgs(diff_map_a_pos, names_map_a_pos,
                      os.path.join(test_path, '{}_{}'.format('interpolation_between_'+id_a+'_'+id_b, str(idx))))
            save_imgs(diff_map_b_pos, names_map_b_pos,
                      os.path.join(test_path, '{}_{}'.format('interpolation_between_'+id_a+'_'+id_b, str(idx))))
            save_imgs(diff_map_a_neg, names_map_a_neg,
                      os.path.join(test_path, '{}_{}'.format('interpolation_between_'+id_a+'_'+id_b, str(idx))))
            save_imgs(diff_map_b_neg, names_map_b_neg,
                      os.path.join(test_path, '{}_{}'.format('interpolation_between_'+id_a+'_'+id_b, str(idx))))


        # interpolation within classes
        healthy_test_iter = iter(anomaly_test_dataloader)
        anomaly_test_iter = iter(anomaly_test_dataloader)
        _, _, _ = anomaly_test_iter.next()

        num = np.minimum(opts.num, len(anomaly_test_dataloader))
        for idx in range(num):
            healthy_images, id_a, _ = healthy_test_iter.next()
            anomaly_images, id_b, _ = anomaly_test_iter.next()
            label_a = id_a[0].numpy()[0]
            label_b = id_b[0].numpy()[0]

            if label_a == label_b:
                continue

            id_a = id_a[1][0]
            id_b = id_b[1][0]

            healthy_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
            healthy_c_org[:, 0] = 1
            anomaly_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
            anomaly_c_org[:, 1] = 1
            images = torch.cat((healthy_images, anomaly_images), dim=0).type(torch.FloatTensor)
            c_org = torch.cat((healthy_c_org, anomaly_c_org), dim=0).type(torch.FloatTensor)
            images = images.to(opts.device).detach()
            c_org = c_org.to(opts.device).detach()
            half_size = images.size(0) // 2
            imgs_a, imgs_b = torch.split(images, half_size, dim=0)
            imgs_b[imgs_b > 1] = 1
            imgs_b[imgs_b < 0] = 0
            imgs_a[imgs_a > 1] = 1
            imgs_a[imgs_a < 0] = 0
            imgs_a = [imgs_a]
            imgs_b = [imgs_b]
            diff_map_a_pos = []
            diff_map_b_pos = []
            diff_map_a_neg = []
            diff_map_b_neg = []

            with torch.no_grad():
                outputs_a, diff_map_a_pos_, diff_map_a_neg_, outputs_b, diff_map_b_pos_, diff_map_b_neg_ = model.test_interpolation(images, c_org)
            imgs_a = imgs_a + outputs_a
            imgs_b = imgs_b + outputs_b
            diff_map_a_pos = diff_map_a_pos + diff_map_a_pos_
            diff_map_b_pos = diff_map_b_pos + diff_map_b_pos_
            diff_map_a_neg = diff_map_a_neg + diff_map_a_neg_
            diff_map_b_neg = diff_map_b_neg + diff_map_b_neg_

            names_a = ['input_a']
            names_b = ['input_b']
            names_map_a_pos = []
            names_map_b_pos = []
            names_map_a_neg = []
            names_map_b_neg = []
            for n in range(10):
                names_a.append('img_0to1_' + str(n))
                names_b.append('img_1to0_' + str(n))
            for n in range(10):
                names_map_a_pos.append('diff_map_0to1_pos_' + str(n))
                names_map_b_pos.append('diff_map_1to0_pos_' + str(n))
                names_map_a_neg.append('diff_map_0to1_neg_' + str(n))
                names_map_b_neg.append('diff_map_1to0_neg_' + str(n))

            save_imgs(imgs_a, names_a, os.path.join(test_path,
                                                    '{}_{}'.format('interpolation_within_ano_' + id_a + '_' + id_b,
                                                                   str(idx))))
            save_imgs(imgs_b, names_b, os.path.join(test_path,
                                                    '{}_{}'.format('interpolation_within_ano_' + id_a + '_' + id_b,
                                                                   str(idx))))
            save_imgs(diff_map_a_pos, names_map_a_pos, os.path.join(test_path, '{}_{}'.format(
                'interpolation_within_ano_' + id_a + '_' + id_b, str(idx))))
            save_imgs(diff_map_b_pos, names_map_b_pos, os.path.join(test_path, '{}_{}'.format(
                'interpolation_within_ano_' + id_a + '_' + id_b, str(idx))))
            save_imgs(diff_map_a_neg, names_map_a_neg, os.path.join(test_path, '{}_{}'.format(
                'interpolation_within_ano_' + id_a + '_' + id_b, str(idx))))
            save_imgs(diff_map_b_neg, names_map_b_neg, os.path.join(test_path, '{}_{}'.format(
                'interpolation_within_ano_' + id_a + '_' + id_b, str(idx))))

        healthy_test_iter = iter(healthy_test_dataloader)
        anomaly_test_iter = iter(healthy_test_dataloader)
        _, _, _ = anomaly_test_iter.next()

        num = np.minimum(opts.num, len(healthy_test_dataloader))
        for idx in range(num):
            healthy_images, id_a, _ = healthy_test_iter.next()
            anomaly_images, id_b, _ = anomaly_test_iter.next()
            label_a = id_a[0].numpy()[0]
            label_b = id_b[0].numpy()[0]

            if label_a == label_b:
                continue

            id_a = id_a[1][0]
            id_b = id_b[1][0]

            healthy_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
            healthy_c_org[:, 0] = 1
            anomaly_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
            anomaly_c_org[:, 1] = 1
            images = torch.cat((healthy_images, anomaly_images), dim=0).type(torch.FloatTensor)
            c_org = torch.cat((healthy_c_org, anomaly_c_org), dim=0).type(torch.FloatTensor)
            images = images.to(opts.device).detach()
            c_org = c_org.to(opts.device).detach()
            half_size = images.size(0) // 2
            imgs_a, imgs_b = torch.split(images, half_size, dim=0)
            imgs_b[imgs_b > 1] = 1
            imgs_b[imgs_b < 0] = 0
            imgs_a[imgs_a > 1] = 1
            imgs_a[imgs_a < 0] = 0
            imgs_a = [imgs_a]
            imgs_b = [imgs_b]
            diff_map_a_pos = []
            diff_map_b_pos = []
            diff_map_a_neg = []
            diff_map_b_neg = []

            with torch.no_grad():
                outputs_a, diff_map_a_pos_, diff_map_a_neg_, outputs_b, diff_map_b_pos_, diff_map_b_neg_ = model.test_interpolation(images, c_org)
            imgs_a = imgs_a + outputs_a
            imgs_b = imgs_b + outputs_b
            diff_map_a_pos = diff_map_a_pos + diff_map_a_pos_
            diff_map_b_pos = diff_map_b_pos + diff_map_b_pos_
            diff_map_a_neg = diff_map_a_neg + diff_map_a_neg_
            diff_map_b_neg = diff_map_b_neg + diff_map_b_neg_

            names_a = ['input_a']
            names_b = ['input_b']
            names_map_a_pos = []
            names_map_b_pos = []
            names_map_a_neg = []
            names_map_b_neg = []
            for n in range(10):
                names_a.append('img_0to1_' + str(n))
                names_b.append('img_1to0_' + str(n))
            for n in range(10):
                names_map_a_pos.append('diff_map_0to1_pos_' + str(n))
                names_map_b_pos.append('diff_map_1to0_pos_' + str(n))
                names_map_a_neg.append('diff_map_0to1_neg_' + str(n))
                names_map_b_neg.append('diff_map_1to0_neg_' + str(n))

            save_imgs(imgs_a, names_a, os.path.join(test_path,
                                                    '{}_{}'.format('interpolation_within_heal_' + id_a + '_' + id_b,
                                                                   str(idx))))
            save_imgs(imgs_b, names_b, os.path.join(test_path,
                                                    '{}_{}'.format('interpolation_within_heal_' + id_a + '_' + id_b,
                                                                   str(idx))))
            save_imgs(diff_map_a_pos, names_map_a_pos, os.path.join(test_path, '{}_{}'.format(
                'interpolation_within_heal_' + id_a + '_' + id_b, str(idx))))
            save_imgs(diff_map_b_pos, names_map_b_pos, os.path.join(test_path, '{}_{}'.format(
                'interpolation_within_heal_' + id_a + '_' + id_b, str(idx))))
            save_imgs(diff_map_a_neg, names_map_a_neg, os.path.join(test_path, '{}_{}'.format(
                'interpolation_within_heal_' + id_a + '_' + id_b, str(idx))))
            save_imgs(diff_map_b_neg, names_map_b_neg, os.path.join(test_path, '{}_{}'.format(
                'interpolation_within_heal_' + id_a + '_' + id_b, str(idx))))

    print(test_path)
    return val_accuracy, val_f1, val_precision, val_recall, cross_corr_a, cross_corr_b, cross_corr_a_std, cross_corr_b_std, val_mae,\
           dice_a_thresh, dice_b_thresh, dice_a_std_thresh, dice_b_std_thresh


def main():
    # batch testing
    parser = TestOptions()
    opts = parser.parse()
    override_opts = OverrideOptions()
    cross_corr_thresh = True

    df = pd.DataFrame(
        columns=["experiment", "val_accuracy", "val_f1", "val_precision",
                 "val_recall", "cross_corr_a", "cross_corr_a_std", "cross_corr_b",
                 "cross_corr_b_std", "mae",
                 "dice_a", "dice_a_std", "dice_b", "dice_b_std"])

    if not opts.batch_experiment is None:
        experiments = [d for d in sorted(os.listdir(opts.batch_experiment)) if
                       os.path.isdir(os.path.join(opts.batch_experiment, d))]
    else:
        experiments = [opts.single_experiment]
    i = 0
    for ex in experiments:
        if not opts.batch_experiment is None:
            resume = os.path.join(opts.batch_experiment, ex)
        else:
            resume = opts.single_experiment

        model_name = override_opts['model_name'].split('.')[0]
        test_path = os.path.join(resume, 'test_' + model_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        if not os.path.exists(test_path + '/pos_mask'):
            os.makedirs(test_path + '/pos_mask')
            os.makedirs(test_path + '/neg_mask')
            os.makedirs(test_path + '/all_mask')

        # if i == 6:
        val_accuracy = np.zeros(1)
        val_f1 = np.zeros(1)
        val_precision = np.zeros(1)
        val_recall = np.zeros(1)
        cross_corr_a = np.zeros(1)
        cross_corr_b = np.zeros(1)
        cross_corr_a_std = np.zeros(1)
        cross_corr_b_std = np.zeros(1)
        mae = np.zeros(1)
        dice_a = np.zeros(1)
        dice_b = np.zeros(1)
        dice_a_std = np.zeros(1)
        dice_b_std = np.zeros(1)


        opts = parser.parse()
        if not opts.batch_experiment is None:
            opts.resume = os.path.join(opts.batch_experiment, ex)
        else:
            opts.resume = opts.single_experiment
        val_accuracy[0], val_f1[0], val_precision[0], val_recall[0], cross_corr_a[0], cross_corr_b[0], \
         cross_corr_a_std[0], cross_corr_b_std[0], mae[0], dice_a[0], dice_b[0], dice_a_std[0],\
            dice_b_std[0] = test(opts, override_opts, shuffle_test=False,
                                 scores=False, generate=False, tsne=False,
                                 group_analysis=False, random_gen_group=False,
                                 random_gen_extreme_group=False,
                                 random_gen=False, interpolation=False,
                                 cross_corr_thresh=cross_corr_thresh)

        df.loc[i, :] = ex, np.mean(val_accuracy), np.mean(val_f1), \
                           np.mean(val_precision), np.mean(val_recall), \
                           np.mean(cross_corr_a), np.mean(cross_corr_a_std), np.mean(cross_corr_b), \
                           np.mean(cross_corr_b_std), np.mean(mae), \
                            np.mean(dice_a), np.mean(dice_a_std), np.mean(dice_b), np.mean(dice_b_std)
        i = i + 1

    results_name = '/results_model'


    df.to_csv(test_path + results_name + '.csv')


if __name__ == '__main__':
    main()
