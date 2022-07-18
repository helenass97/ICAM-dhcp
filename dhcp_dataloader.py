import torch
from torch.utils.data import Dataset
import os
import numpy as np
import torchvision
import scipy.misc
import imageio
import pickle
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from skimage.transform import rescale, resize, downscale_local_mean
import SimpleITK as sitk
from os.path import isfile, join
from os import listdir

class Dataset2D(torch.utils.data.Dataset):
    """2D MRI dataset loader"""

    def __init__(self, image_path='TRAIN_ga_regression/Images', image_path_2=None,
                 label_path='TRAIN_ga_regression/TRAIN_ga_regressionremove_myelin.pkl',
                 task='regression',
                 num_classes=1,
                 label_type='age',
                 output_id=False,
                 test_subjects=None,
                 data_type = 'train',
                 class_label = None,
                 transform=None):
        """
        Args:
            image_path (string): path to image folder.
            label_path (string): path to labels file.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.data_type = data_type
        self.task = task
        self.num_classes = num_classes
        self.transform = transform
        self.img_path = os.path.join(image_path)
        if image_path_2:
            self.img_path_2 = os.path.join(image_path_2)
        else:
            self.img_path_2 = image_path_2

        self.label_type = label_type
        self.output_id = output_id
        file = open(os.path.join(label_path), 'rb')

        if label_type == 'age':
            self.labels = pickle.load(file)

            ids = sorted(list(self.labels.loc[:,'id']))

            dupes = [x for n, x in enumerate(ids) if x in ids[:n]]
            dupes = set(dupes)
            dupes = list(dupes)

            for d in dupes:
                temp = self.labels.index[self.labels['id'] == d].tolist()
                # if regression keep first session, if classification keep last session
                if task == 'regression':
                    temp2 = temp[1:]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

                elif task == 'classification':
                    temp2 = temp[:-1]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

            if task == 'classification':
                # print(len(self.labels))
                self.labels = self.labels[self.labels['scan_ga']>((35-40.4482070460464)/1.9935309236699883)]
                # print(len(self.labels))

            # extract ids of certain class - keep only labels of one class
            if not(class_label is None):
                self.labels = self.labels[self.labels['is_prem'] == class_label]

            image_paths = sorted(os.listdir(self.img_path))
            remove_ind = []
            i=0
            # check which images are present in labels
            for img in image_paths:
                f = img.split('_')
                f1 = f[5]
                f1 = f1.split('-')
                subject = f1[3]
                if not(any(self.labels['id'].str.match(subject))):
                    remove_ind.append(i)
                i=i+1
            self.img_pref = f[0] + '_' + f[1] + '__' + f[3] + '_' + f[4] + '_-subj-'

            # print(len(image_paths))
            # remove images without labels
            image_paths = [i for j, i in enumerate(image_paths) if j not in remove_ind]
            # print(len(image_paths))


        elif label_type == 'cognitive':
            self.labels = pickle.load(file)
            # print(len(self.labels))

            # drop nans
            self.labels = self.labels[pd.notnull(self.labels['composite_score'])]
            self.labels = self.labels[pd.notnull(self.labels['IMD_score'])]
            self.labels = self.labels[self.labels['IMD_score']!=-998]
            # print(len(self.labels))

            ids = sorted(list(self.labels.loc[:, 'id']))

            dupes = [x for n, x in enumerate(ids) if x in ids[:n]]
            dupes = set(dupes)
            dupes = list(dupes)

            for d in dupes:
                temp = self.labels.index[self.labels['id'] == d].tolist()
                # if regression keep first session, if classification keep last session
                if task == 'regression':
                    temp2 = temp[1:]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

                elif task == 'classification':
                    temp2 = temp[:-1]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

            if task == 'classification':
                # print(len(self.labels))
                self.labels = self.labels[self.labels['scan_ga']>((35-40.4482070460464)/1.9935309236699883)]
                # print(len(self.labels))


            # extract ids of certain class - keep only labels of one class
            if not(class_label is None):
                self.labels = self.labels[self.labels['composite_score'] == class_label]

            image_paths = sorted(os.listdir(self.img_path))
            remove_ind = []
            i=0
            # check which images are present in labels
            for img in image_paths:
                f = img.split('_')
                f1 = f[5]
                f1 = f1.split('-')
                subject = f1[3]
                if not(any(self.labels['id'].str.match(subject))):
                    remove_ind.append(i)
                i=i+1

            self.img_pref = f[0] + '_' + f[1] + '__' + f[3] + '_' + f[4] + '_-subj-'

            # print(len(image_paths))
            # remove images without labels
            image_paths = [i for j, i in enumerate(image_paths) if j not in remove_ind]
            # print(len(image_paths))

        self.img_pref = f[0] + '_' + f[1] + '__' + f[3] + '_' + f[4] + '_-subj-'
        # get unique id list (without duplicates)
        if data_type == 'train':
            id_list = []
            for img in image_paths:
                f = img.split('_')
                f1 = f[5]
                sess = f[6]
                f1 = f1.split('-')
                id_list.append('sub-' + f1[3] + '_' + sess)
            id_list = list(dict.fromkeys(id_list))

            # remove test subjects
            if test_subjects:
                id_list = [e for e in id_list if e not in test_subjects]
        elif data_type == 'test':
            id_list = test_subjects

            if not(class_label is None):
                id_list = sorted(id_list)
                remove_ind = []
                i=0
                # check which images are present in labels
                for img in id_list:
                    f = img.split('_')
                    f1 = f[0]
                    f1 = f1.split('-')
                    subject = f1[1]
                    if not(any(self.labels['id'].str.match(subject))):
                        remove_ind.append(i)
                    i=i+1
                id_list = [i for j, i in enumerate(id_list) if j not in remove_ind]


        labels_new = pd.DataFrame()
        # keep only labels with images
        for img in id_list:
            f = img.split('_')
            f1 = f[0]
            f1 = f1.split('-')
            subject = f1[1]
            if any(self.labels['id'].str.match(subject)):
                labels_new = labels_new.append(self.labels.loc[self.labels['id'] == subject])
        self.labels = labels_new

        # print([len(self.labels)])
        # print(len(id_list))
        # # randomly choose test subject
        # len_labels = len(image_paths)
        # indices = list(range(len_labels))
        # np.random.shuffle(indices)
        #
        # test_labels = []
        # for t in range(10):
        #     test_labels.append(image_paths[indices[t]])
        # print(test_labels)

        self.image_paths = sorted(image_paths)
        self.id_list = sorted(id_list)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx, plot=False):
        img_name = sorted(self.id_list)[idx]
        img_name = self.img_pref + img_name
        if self.img_path_2:
            rand_num = (np.random.uniform(0,1,1) > 0.5).astype(int)
            if rand_num == 1:
                image = np.float32(np.load(os.path.join(self.img_path, img_name)))
            elif rand_num == 0:
                image = np.float32(np.load(os.path.join(self.img_path_2, img_name)))
        else:
            image = np.float32(np.load(os.path.join(self.img_path, img_name)))

        if plot:
            fig = plt.figure()
            a = fig.add_subplot(2, 3, 1)
            imgplot = plt.imshow(image[:,:,0])
            a.axis('off')
            a.set_title('Before aug')
            a = fig.add_subplot(2, 3, 2)
            imgplot = plt.imshow(image[:,:,1])
            a.axis('off')
            a = fig.add_subplot(2, 3, 3)
            imgplot = plt.imshow(image[:,:,2])
            a.axis('off')


        if self.transform:
            image = self.transform(image.astype(np.float16))
            image = image.astype(np.float32)

        if plot:
            a = fig.add_subplot(2, 3, 4)
            imgplot = plt.imshow(image[:,:,0])
            a.axis('off')
            a.set_title('After aug')
            a = fig.add_subplot(2, 3, 5)
            imgplot = plt.imshow(image[:,:,1])
            a.axis('off')
            a = fig.add_subplot(2, 3, 6)
            imgplot = plt.imshow(image[:,:,2])
            a.axis('off')
            plt.show()

        image = torch.from_numpy(image.copy()).float()
        image = image.permute(2,0,1)

        if self.label_type == 'age':
            f = img_name.split('_')[5]
            f = f.split('-')
            subject = f[3]
            if self.task == 'regression':
                label = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label_id = 'sub-' + label['id'] + '_ses-' + str(label['session'])
                label = np.array(label['scan_ga'])
            elif self.task == 'classification':
                values = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label_id = 'sub-' + values['id'] + '_ses-' + str(values['session'])
                values = values['is_prem'].astype(int)
                label = np.zeros(self.num_classes)
                label[values] = 1

            label = torch.from_numpy(label).float()

            if self.output_id:
                label = [label, label_id]

        elif self.label_type == 'cognitive':
            f = img_name.split('_')[5]
            f = f.split('-')
            subject = f[3]
            if self.task == 'regression':
                label = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label_cog = np.array(label['composite_score'])
                label_imd = np.array(label['IMD_score'])
                label_birthga = np.array(label['birth_ga'])
                label_scanga = np.array(label['scan_ga'])
                label_corrected_age = np.array(label['corrected_age'])
                label_id = 'sub-' + label['id'] + '_ses-' + str(label['session'])
                if self.output_id:
                    label = [torch.from_numpy(label_cog).float(),
                             torch.from_numpy(label_birthga).float().unsqueeze(0),
                             torch.from_numpy(label_scanga).float().unsqueeze(0),
                             torch.from_numpy(label_corrected_age).float().unsqueeze(0),
                             torch.from_numpy(label_imd).float().unsqueeze(0),
                             label_id]
                else:
                    label = [torch.from_numpy(label_cog).float(),
                             torch.from_numpy(label_birthga).float().unsqueeze(0),
                             torch.from_numpy(label_scanga).float().unsqueeze(0),
                             torch.from_numpy(label_corrected_age).float().unsqueeze(0),
                             torch.from_numpy(label_imd).float().unsqueeze(0)
                             ]
            elif self.task == 'classification':
                label = self.labels.loc[self.labels['id'] == subject].iloc[0]
                cog_temp = (label['composite_score'] > 100).astype(int)
                label_imd = np.array(label['IMD_score'])
                label_birthga = np.array(label['birth_ga'])
                label_scanga = np.array(label['scan_ga'])
                label_corrected_age = np.array(label['corrected_age'])
                label_cog = np.zeros(self.num_classes)
                label_cog[cog_temp] = 1
                label_id = 'sub-' + label['id'] + '_ses-' + str(label['session'])
                if self.output_id:
                    label = [torch.from_numpy(label_cog).float(),
                             torch.from_numpy(label_birthga).float().unsqueeze(0),
                             torch.from_numpy(label_scanga).float().unsqueeze(0),
                             torch.from_numpy(label_corrected_age).float().unsqueeze(0),
                             torch.from_numpy(label_imd).float().unsqueeze(0),
                             label_id]
                else:
                    label = [torch.from_numpy(label_cog).float(),
                             torch.from_numpy(label_birthga).float().unsqueeze(0),
                             torch.from_numpy(label_scanga).float().unsqueeze(0),
                             torch.from_numpy(label_corrected_age).float().unsqueeze(0),
                             torch.from_numpy(label_imd).float().unsqueeze(0)]


        mask = torch.zeros((0))

        return [image, label, mask]

    def make_weights_for_balanced_classes(self):
        nclasses = self.num_classes
        count = [0] * nclasses

        for name in self.id_list:
            img_name = self.img_pref + name

            if self.label_type == 'age':
                f = img_name.split('_')[5]
                f = f.split('-')
                subject = f[3]
                values = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label = values['is_prem'].astype(int)

            elif self.label_type == 'cognitive':
                f = img_name.split('_')[5]
                f = f.split('-')
                subject = f[3]
                label = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label = (label['composite_score'] > 100).astype(int)

            count[label] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(self.id_list)
        for idx, name in enumerate(self.id_list):
            img_name = self.img_pref + name

            if self.label_type == 'age':
                f = img_name.split('_')[5]
                f = f.split('-')
                subject = f[3]
                values = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label = values['is_prem'].astype(int)

            elif self.label_type == 'cognitive':
                f = img_name.split('_')[5]
                f = f.split('-')
                subject = f[3]
                label = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label = (label['composite_score'] > 100).astype(int)

            weight[idx] = weight_per_class[label]
        return weight


class Dataset2D_dhcp(torch.utils.data.Dataset): #added and modified by Helena (from Dataset2D)
    """2D MRI dataset loader"""

    def __init__(self, image_path='/data/helena/dhcp-2d', image_path_2=None,
                 label_path='/data/helena/labels_ants_full.pkl',
                 task='classification', #classification or regression
                 num_classes=2, #was 1 before
                 label_type='age', #was age for regression, but should be 1 or 0 for term/preterm classification
                 output_id=False,
                 test_subjects=None,
                 data_type = 'train',
                 class_label = None,
                 transform=None):
        """
        Args:
            image_path (string): path to image folder.
            label_path (string): path to labels file.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.data_type = data_type
        self.task = task
        self.num_classes = num_classes
        self.transform = transform
        self.img_path = os.path.join(image_path)
        if image_path_2:
            self.img_path_2 = os.path.join(image_path_2)
        else:
            self.img_path_2 = image_path_2

        self.label_type = label_type
        self.output_id = output_id
        file = open(os.path.join(label_path), 'rb')

        if label_type == 'age':
            self.labels = pickle.load(file)

            ids = sorted(list(self.labels.loc[:,'id']))

            dupes = [x for n, x in enumerate(ids) if x in ids[:n]]
            dupes = set(dupes)
            dupes = list(dupes)

            for d in dupes:
                temp = self.labels.index[self.labels['id'] == d].tolist()
                # if regression keep first session, if classification keep last session
                if task == 'regression':
                    temp2 = temp[1:]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

                elif task == 'classification':
                    temp2 = temp[:-1]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

            if task == 'classification':
                # print(len(self.labels))
                self.labels = self.labels[self.labels['scan_ga']>((35-40.4482070460464)/1.9935309236699883)]  #dont understand where this comes from
                # print(len(self.labels))

            # extract ids of certain class - keep only labels of one class
            if not(class_label is None):
                self.labels = self.labels[self.labels['is_prem'] == class_label]

            image_paths = sorted(os.listdir(self.img_path))
            remove_ind = []
            i=0
            # check which images are present in labels
            for img in image_paths:
                f = img.split('_')
                f1 = f[5]
                f1 = f1.split('-')
                subject = f1[3]
                if not(any(self.labels['id'].str.match(subject))):
                    remove_ind.append(i)
                i=i+1
            self.img_pref = f[0] + '_' + f[1] + '__' + f[3] + '_' + f[4] + '_-subj-'

            # print(len(image_paths))
            # remove images without labels
            image_paths = [i for j, i in enumerate(image_paths) if j not in remove_ind]
            # print(len(image_paths))


        self.img_pref = f[0] + '_' + f[1] + '__' + f[3] + '_' + f[4] + '_-subj-'
        # get unique id list (without duplicates)
        if data_type == 'train':
            id_list = []
            for img in image_paths:
                f = img.split('_')
                f1 = f[5]
                sess = f[6]
                f1 = f1.split('-')
                id_list.append('sub-' + f1[3] + '_' + sess)
            id_list = list(dict.fromkeys(id_list))

            # remove test subjects
            if test_subjects:
                id_list = [e for e in id_list if e not in test_subjects]
        elif data_type == 'test':
            id_list = test_subjects

            if not(class_label is None):
                id_list = sorted(id_list)
                remove_ind = []
                i=0
                # check which images are present in labels
                for img in id_list:
                    f = img.split('_')
                    f1 = f[0]
                    f1 = f1.split('-')
                    subject = f1[1]
                    if not(any(self.labels['id'].str.match(subject))):
                        remove_ind.append(i)
                    i=i+1
                id_list = [i for j, i in enumerate(id_list) if j not in remove_ind]


        labels_new = pd.DataFrame()
        # keep only labels with images
        for img in id_list:
            f = img.split('_')
            f1 = f[0]
            f1 = f1.split('-')
            subject = f1[1]
            if any(self.labels['id'].str.match(subject)):
                labels_new = labels_new.append(self.labels.loc[self.labels['id'] == subject])
        self.labels = labels_new

    
        #Added by Helena from Abdulah_DHCP (need to change this to preterm or term labels and not scan age > 37)
        if class_label == 0:
            self.labels = self.labels[self.labels[:, 1] >= 37]
        else:
            self.labels = self.labels[self.labels[:, 1] < 37]


        self.image_paths = sorted(image_paths)
        self.id_list = sorted(id_list)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx, plot=False):
        img_name = sorted(self.id_list)[idx]
        img_name = self.img_pref + img_name
        if self.img_path_2:
            rand_num = (np.random.uniform(0,1,1) > 0.5).astype(int)
            if rand_num == 1:
                image = np.float32(np.load(os.path.join(self.img_path, img_name)))
            elif rand_num == 0:
                image = np.float32(np.load(os.path.join(self.img_path_2, img_name)))
        else:
            image = np.float32(np.load(os.path.join(self.img_path, img_name)))

        if plot:
            fig = plt.figure()
            a = fig.add_subplot(2, 3, 1)
            imgplot = plt.imshow(image[:,:,0])
            a.axis('off')
            a.set_title('Before aug')
            a = fig.add_subplot(2, 3, 2)
            imgplot = plt.imshow(image[:,:,1])
            a.axis('off')
            a = fig.add_subplot(2, 3, 3)
            imgplot = plt.imshow(image[:,:,2])
            a.axis('off')


        if self.transform:
            image = self.transform(image.astype(np.float16))
            image = image.astype(np.float32)

        if plot:
            a = fig.add_subplot(2, 3, 4)
            imgplot = plt.imshow(image[:,:,0])
            a.axis('off')
            a.set_title('After aug')
            a = fig.add_subplot(2, 3, 5)
            imgplot = plt.imshow(image[:,:,1])
            a.axis('off')
            a = fig.add_subplot(2, 3, 6)
            imgplot = plt.imshow(image[:,:,2])
            a.axis('off')
            plt.show()

        image = torch.from_numpy(image.copy()).float()
        image = image.permute(2,0,1)

        if self.label_type == 'age':
            f = img_name.split('_')[5]
            f = f.split('-')
            subject = f[3]
            if self.task == 'regression':
                label = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label_id = 'sub-' + label['id'] + '_ses-' + str(label['session'])
                label = np.array(label['scan_ga'])
                
            elif self.task == 'classification':
                values = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label_id = 'sub-' + values['id'] + '_ses-' + str(values['session'])
                values = values['is_prem'].astype(int)
                label = np.zeros(self.num_classes)
                label[values] = 1

            label = torch.from_numpy(label).float()

            if self.output_id:
                label = [label, label_id]

        mask = torch.zeros((0))

        return [image, label, mask]

    def make_weights_for_balanced_classes(self):
        nclasses = self.num_classes
        count = [0] * nclasses

        for name in self.id_list:
            img_name = self.img_pref + name

            if self.label_type == 'age':
                f = img_name.split('_')[5]
                f = f.split('-')
                subject = f[3]
                values = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label = values['is_prem'].astype(int)

            elif self.label_type == 'cognitive':
                f = img_name.split('_')[5]
                f = f.split('-')
                subject = f[3]
                label = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label = (label['composite_score'] > 100).astype(int)

            count[label] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(self.id_list)
        for idx, name in enumerate(self.id_list):
            img_name = self.img_pref + name

            if self.label_type == 'age':
                f = img_name.split('_')[5]
                f = f.split('-')
                subject = f[3]
                values = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label = values['is_prem'].astype(int)

            elif self.label_type == 'cognitive':
                f = img_name.split('_')[5]
                f = f.split('-')
                subject = f[3]
                label = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label = (label['composite_score'] > 100).astype(int)

            weight[idx] = weight_per_class[label]
        return weight





class Dataset3D(torch.utils.data.Dataset):
    """3D MRI dataset loader"""

    def __init__(self, image_path='/home/cb19/Documents/3d_data_registered_Affine_NMI/masked_images',
                 label_path='/home/cb19/Github/MRI_analysis/ages_scores.pkl',
                 task='regression',
                 num_classes=1,
                 label_type='age',
                 output_id=False,
                 test_subjects=None,
                 data_type = 'train',
                 class_label = None,
                 transform=None):
        """
        Args:
            image_path (string): path to image folder.
            label_path (string): path to labels file.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.task = task
        self.num_classes = num_classes
        self.transform = transform
        self.img_dir = os.path.join(image_path)
        image_paths = sorted(os.listdir(self.img_dir))
        self.label_type = label_type
        self.output_id = output_id
        file = open(os.path.join(label_path), 'rb')


        if label_type == 'age':
            self.labels = pickle.load(file)

            ids = sorted(list(self.labels.loc[:,'id']))

            dupes = [x for n, x in enumerate(ids) if x in ids[:n]]
            dupes = set(dupes)
            dupes = list(dupes)

            for d in dupes:
                temp = self.labels.index[self.labels['id'] == d].tolist()
                # if regression keep first session, if classification keep last session
                if task == 'regression':
                    temp2 = temp[1:]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

                elif task == 'classification':
                    temp2 = temp[:-1]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

            if task == 'classification':
                # print(len(self.labels))
                self.labels = self.labels[self.labels['scan_ga']>((35-40.4482070460464)/1.9935309236699883)]
                # print(len(self.labels))

            # extract ids of certain class - keep only labels of one class
            if not(class_label is None):
                self.labels = self.labels[self.labels['is_prem'] == class_label]

            remove_ind = []
            i = 0
            # check which images are present in labels
            for img in image_paths:
                f = img.split('_')[0]
                f = f.split('-')
                subject = f[0]
                session = f[1]
                if not (any(self.labels['id'].str.match(subject))):
                    remove_ind.append(i)
                elif not (any(self.labels['session'] == int(session))):
                    remove_ind.append(i)
                elif any(self.labels['id'].str.match(subject)):
                    temp = self.labels.loc[self.labels['id'] == subject]
                    if not (any((temp['session'] == int(session)).values)):
                        remove_ind.append(i)
                i = i + 1

            # remove images without labels
            image_paths = [i for j, i in enumerate(image_paths) if j not in remove_ind]
            # self.img_pref = f[0] + '_' + f[1] + '__' + f[3] + '_' + f[4] + '_-subj-'


        elif label_type == 'cognitive':
            self.labels = pickle.load(file)
            # print(len(self.labels))

            # drop nans
            self.labels = self.labels[pd.notnull(self.labels['composite_score'])]
            self.labels = self.labels[pd.notnull(self.labels['IMD_score'])]
            self.labels = self.labels[self.labels['IMD_score']!=-998]
            # print(len(self.labels))

            ids = sorted(list(self.labels.loc[:, 'id']))

            dupes = [x for n, x in enumerate(ids) if x in ids[:n]]
            dupes = set(dupes)
            dupes = list(dupes)

            for d in dupes:
                temp = self.labels.index[self.labels['id'] == d].tolist()
                # if regression keep first session, if classification keep last session
                if task == 'regression':
                    temp2 = temp[1:]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

                elif task == 'classification':
                    temp2 = temp[:-1]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

            if task == 'classification':
                # print(len(self.labels))
                self.labels = self.labels[self.labels['scan_ga']>((35-40.4482070460464)/1.9935309236699883)]
                # print(len(self.labels))


            # extract ids of certain class - keep only labels of one class
            if not(class_label is None):
                self.labels = self.labels[self.labels['composite_score'] == class_label]

            image_paths = sorted(os.listdir(self.img_path))
            remove_ind = []
            i = 0
            # check which images are present in labels
            for img in image_paths:
                f = img.split('_')[0]
                f = f.split('-')
                subject = f[0]
                session = f[1]
                if not (any(self.labels['id'].str.match(subject))):
                    remove_ind.append(i)
                elif not (any(self.labels['session'] == int(session))):
                    remove_ind.append(i)
                elif any(self.labels['id'].str.match(subject)):
                    temp = self.labels.loc[self.labels['id'] == subject]
                    if not (any((temp['session'] == int(session)).values)):
                        remove_ind.append(i)
                i = i + 1

            # remove images without labels
            image_paths = [i for j, i in enumerate(image_paths) if j not in remove_ind]

            # self.img_pref = f[0] + '_' + f[1] + '__' + f[3] + '_' + f[4] + '_-subj-'
        # get unique id list (without duplicates)
        # if data_type == 'train':
        id_list = []
        for img in image_paths:
            f = img.split('_')[0]
            f = f.split('-')
            subject = f[0]
            session = f[1]
            id_list.append(subject + '-' + session)
        id_list = list(dict.fromkeys(id_list))

        if data_type == 'train':
            # remove test subjects
            if test_subjects:
                # print(len(id_list))
                id_list = [e for e in id_list if e not in test_subjects]
                # print(len(id_list))

        if data_type == 'test':
            id_list = [e for e in test_subjects if e not in id_list]

            if not(class_label is None):
                id_list = sorted(id_list)
                remove_ind = []
                i=0
                # check which images are present in labels
                for img in id_list:
                    # f = img.split('_')[0]
                    f = img.split('-')
                    subject = f[0]
                    session = f[1]
                    if not(any(self.labels['id'].str.match(subject))):
                        remove_ind.append(i)
                    i=i+1
                id_list = [i for j, i in enumerate(id_list) if j not in remove_ind]


        labels_new = pd.DataFrame()
        # keep only labels with images
        for img in id_list:
            # f = img.split('_')[0]
            f = img.split('-')
            subject = f[0]
            session = f[1]
            if any(self.labels['id'].str.match(subject)):
                labels_new = labels_new.append(self.labels.loc[self.labels['id'] == subject])

        # print(len(labels_new))
        # print(len(id_list))

        self.labels = labels_new
        self.image_paths = sorted(image_paths)
        self.id_list = sorted(id_list)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx, plot=False):
        img_name = sorted(self.id_list)[idx]
        image = np.load(os.path.join(self.img_dir, img_name +'_t1_t2_ratio_masked.npy'))

        if plot:
            ind = 50
            sag_slice = image[:, :, ind]
            cor_slice = image[:, ind, :]
            axi_slice = image[ind, :, :]
            fig = plt.figure()
            a = fig.add_subplot(2, 3, 1)
            imgplot = plt.imshow(np.rot90(sag_slice))
            a.axis('off')
            a.set_title('Before aug')
            a = fig.add_subplot(2, 3, 2)
            imgplot = plt.imshow(np.rot90(cor_slice))
            a.axis('off')
            a = fig.add_subplot(2, 3, 3)
            imgplot = plt.imshow(np.rot90(axi_slice))
            a.axis('off')

            print(np.min(image))
            print(np.max(image))
            print(np.mean(image))
            print(np.std(image))

        if self.transform:
            image = self.transform(image)


        if plot:
            print(np.min(image))
            print(np.max(image))
            print(np.mean(image))
            print(np.std(image))

            ind = 50
            sag_slice = image[:, :, ind]
            cor_slice = image[:, ind, :]
            axi_slice = image[ind, :, :]

            a = fig.add_subplot(2, 3, 4)
            imgplot = plt.imshow(np.rot90(sag_slice))
            a.axis('off')
            a.set_title('After aug')
            a = fig.add_subplot(2, 3, 5)
            imgplot = plt.imshow(np.rot90(cor_slice))
            a.axis('off')
            a = fig.add_subplot(2, 3, 6)
            imgplot = plt.imshow(np.rot90(axi_slice))
            a.axis('off')
            plt.show()

        image = torch.from_numpy(image.copy()).float()
        image = image.unsqueeze(0)

        # f = img_name.split('_')[0]
        f = img_name.split('-')
        subject = f[0]
        session = int(f[1])

        if self.task == 'regression':
            label = self.labels.loc[self.labels['id'] == subject]
            label = label.loc[label['session'] == session]
            label_id = str(label['id'].to_numpy()[0])+ '-' + str(label['session'].to_numpy()[0])
            label = label['scan_ga'].to_numpy()

        elif self.task == 'classification':
            values = self.labels.loc[self.labels['id'] == subject]
            values = values.loc[values['session'] == session]
            label_id = str(values['id'].to_numpy()[0])+ '-' + str(values['session'].to_numpy()[0])
            values = values['is_prem'].to_numpy().astype(int)
            label = np.zeros(self.num_classes)
            label[values] = 1

        label = torch.from_numpy(label).float()
        if self.output_id:
            label = [label, label_id]

        mask = torch.zeros((0))
        return [image, label, mask]


class DHCP(torch.utils.data.Dataset):
    """3D MRI dataset loader"""

    def __init__(self, image_path='/data/helena/dhcp-2d',
                 label_path='/data/helena/labels_ants_full.pkl',
                 task='classification',
                 num_classes=2,
                 get_id=False,
                 filter_labels=True,
                 class_label = None,
                 transform=None):
        """
        Args:
            image_path (string): path to image folder.
            label_path (string): path to labels file.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.task = task
        self.num_classes = num_classes
        self.transform = transform
        self.img_dir = os.path.join(image_path)
        image_paths = sorted(os.listdir(self.img_dir))
        self.image_id = None
        self.get_id = get_id
        file = open(os.path.join(label_path), 'rb')

        self.labels = pickle.load(file)
        # print(len(self.labels))
        if filter_labels:
            self.labels.drop(self.labels.loc[self.labels['scan_ga'] < 37].index, inplace=True)
        # print(len(self.labels))

        # extract ids of certain class - keep only labels of one class
        if not(class_label is None):
            # print(len(self.labels))
            self.labels = self.labels[self.labels['is_prem'] == class_label]
            # print(len(self.labels))

        # remove_ind = []
        # i = 0
        # # check which images are present in labels
        # for img in image_paths:
        #     #f = img.split('_')[0]
        #     f = img.split('_')[0]
        #     f = f.split('-')
        #     subject = f[0]
        #     #session = f[1].split('.')[0]
        #     session = f[1].split('_')[0]
        #     if not (any(self.labels['id'].str.match(subject))):
        #         remove_ind.append(i)
        #     elif not (any(self.labels['session'] == int(session))):
        #         remove_ind.append(i)
        #     elif any(self.labels['id'].str.match(subject)):
        #         temp = self.labels.loc[self.labels['id'] == subject]
        #         if not (any((temp['session'] == int(session)).values)):
        #             remove_ind.append(i)
        #     i = i + 1

        # remove images without labels
        # print(len(image_paths))
        # print(self.labels['id'].tolist())
        #image_paths = [i for j, i in enumerate(image_paths) if j not in remove_ind]
        # print(len(image_paths))

        print('Birth age stats: min, max, mean, std')
        print(self.labels['birth_ga'].min())
        print(self.labels['birth_ga'].max())
        print(self.labels['birth_ga'].mean())
        print(self.labels['birth_ga'].std())

        print('Scan age stats: min, max, mean, std')
        print(self.labels['scan_ga'].min())
        print(self.labels['scan_ga'].max())
        print(self.labels['scan_ga'].mean())
        print(self.labels['scan_ga'].std())

        self.image_paths = sorted(image_paths)
        
        img_name = sorted(self.image_paths)[0]
        image = np.float32(np.load(os.path.join(self.img_dir, img_name)))
        print('image loaded has size: ')
        print(image.shape)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx, plot=False):
        img_name = sorted(self.image_paths)[idx]
        image = np.float32(np.load(os.path.join(self.img_dir, img_name)))
        print('image loaded has size: ')
        print(image.shape)
        #image = nib.load(os.path.join(self.img_dir, img_name)) #FOR 3D IMAGES
        #image = np.float32(image.get_fdata())
        

        # if plot:
        #     ind = 50
        #     sag_slice = image[:, :, ind]
        #     cor_slice = image[:, ind, :]
        #     axi_slice = image[ind, :, :]
        #     fig = plt.figure()
        #     a = fig.add_subplot(2, 3, 1)
        #     imgplot = plt.imshow(np.rot90(sag_slice))
        #     a.axis('off')
        #     a.set_title('Before aug')
        #     a = fig.add_subplot(2, 3, 2)
        #     imgplot = plt.imshow(np.rot90(cor_slice))
        #     a.axis('off')
        #     a = fig.add_subplot(2, 3, 3)
        #     imgplot = plt.imshow(np.rot90(axi_slice))
        #     a.axis('off')

        #     print(np.min(image))
        #     print(np.max(image))
        #     print(np.mean(image))
        #     print(np.std(image))

        if self.transform:
            image = self.transform(image)


        # if plot:
        #     print(np.min(image))
        #     print(np.max(image))
        #     print(np.mean(image))
        #     print(np.std(image))

        #     ind = 50
        #     sag_slice = image[:, :, ind]
        #     cor_slice = image[:, ind, :]
        #     axi_slice = image[ind, :, :]

        #     a = fig.add_subplot(2, 3, 4)
        #     imgplot = plt.imshow(np.rot90(sag_slice))
        #     a.axis('off')
        #     a.set_title('After aug')
        #     a = fig.add_subplot(2, 3, 5)
        #     imgplot = plt.imshow(np.rot90(cor_slice))
        #     a.axis('off')
        #     a = fig.add_subplot(2, 3, 6)
        #     imgplot = plt.imshow(np.rot90(axi_slice))
        #     a.axis('off')
        #     plt.show()

        #image = torch.from_numpy(image.copy()).float()
        #image = image.unsqueeze(0)

        f = img_name.split('-')
        subject = f[0]
        session = int(f[1].split('_')[0]) # added '_' instead of '.' to remove slice id from subject

        label = self.labels.loc[self.labels['id'] == subject]
        label = label.loc[label['session'] == session]
        #label_id = str(label['id'].to_numpy()[0])+ '-' + str(label['session'].to_numpy()[0])
        label_id = str(label['id'].to_numpy())+ '-' + str(label['session'].to_numpy())
        label = label['birth_ga'].to_numpy()

        label = torch.from_numpy(label).float()
        if self.get_id:
            label = [label, label_id]

        mask = torch.zeros((0))
        return [image, label, mask]
    
class DHCP_2D(torch.utils.data.Dataset): # modified by Helena - to be used for classification term vs preterm 
    """3D MRI dataset loader"""

    def __init__(self, image_path='/data/helena/dhcp-2d',
                 label_path='/data/helena/labels_ants_full.pkl',
                 task='regression',
                 num_classes=2,
                 class_label = None,
                 transform=None):
        """
        Args:
            image_path (string): path to image folder.
            label_path (string): path to labels file.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.task = task
        self.num_classes = num_classes
        self.transform = transform
        self.img_dir = os.path.join(image_path)
        image_paths = sorted(os.listdir(self.img_dir))
        self.image_id = None
        #self.get_id = get_id
        file = open(os.path.join(label_path), 'rb')
        self.class_label = class_label
        self.labels = pickle.load(file)
        
        
        # choose if want preterm or term labels/images - filter images 
        if class_label == 1 :
           labels = self.labels[self.labels['is_prem'] == 1]
           self.labels = labels[labels['scan_ga'] <= 37]
        
        else:
            labels = self.labels[self.labels['is_prem'] == 0]
            self.labels = labels[labels['scan_ga'] > 37]
                    

        remove_ind = []
        i = 0
        # check which images are present in labels
        for img in image_paths:
            f = img.split('-')
            subject = 'CC' + f[0] # example: CC + 0056XX12 (to match the labels)
            session = f[1].split('_')[0] #example: 13823
            
            if not (any(self.labels['id'].str.match(subject))):
                remove_ind.append(i)
            elif not (any(self.labels['session'] == int(session))):
                remove_ind.append(i)
            elif any(self.labels['id'].str.match(subject)):
                temp = self.labels.loc[self.labels['id'] == subject]
                if not (any((temp['session'] == int(session)).values)):
                    remove_ind.append(i)
            i = i + 1

        #remove images without labels
        #print('length of image path before removing is ' + str(len(image_paths)))
        #print(self.labels['id'].tolist())
        
        image_paths = [i for j, i in enumerate(image_paths) if j not in remove_ind]


        self.image_paths = sorted(image_paths)
        #print('len of images: ' + str(len(self.image_paths)))
        print('len of scan age labels: ' + str(len(self.labels['scan_ga'])))
        

        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx, plot=False):
        
        # In the 2D case - load 3D image using nib
        img_name = sorted(self.image_paths)[idx]
        image = np.float32(np.load(os.path.join(self.img_dir, img_name)))
        
        #print('image loaded before transform has size: ' + str(image.shape) )
        
        # In the 3D case - load 3D image using nib
        #image = nib.load(os.path.join(self.img_dir, img_name)) #FOR 3D IMAGES use nib load
        #image = np.float32(image.get_fdata())
        

        if self.transform:
            image = self.transform(image)
            
        # unsqueeze image to add 1st channel 
        image = torch.from_numpy(image.copy()).float()
        image = image.unsqueeze(0)

        
        #modified from DATASET_3D
        f = img_name.split('-')
        subject = 'CC' +  f[0]
        session = int(f[1].split('_')[0]) # added '_' instead of '.' to remove slice id from subject (2D data)

        if self.task == 'regression':
            label = self.labels.loc[self.labels['id'] == subject]
            label = label.loc[label['session'] == session]
            label = label['scan_ga'].to_numpy() # to predict age of the scan 
            #label = label['birth_ga'].to_numpy() # to predict age at birth 

        elif self.task == 'classification':
            values = self.labels.loc[self.labels['id'] == subject]
            values = values.loc[values['session'] == session]
            values = values['is_prem'].to_numpy().astype(int)
            label = np.zeros(self.num_classes)
            label[values] = 1
            
            
        label = torch.from_numpy(label).float()
        mask = torch.zeros((0))
        print('label size:  ' + str(label.numpy()))
        #print('mask size:  ' + str(mask.numpy()))
        
        return [image, label, mask]


    
class DHCP_Abdulah(torch.utils.data.Dataset):
    """3D MRI dataset loader"""

    def __init__(self, image_path='/data/dHCP/ANTS_full/data_norm',
                 label_path='/data/dHCP/subjects_abdulah/birth_age/train.npy',
                 num_classes=2,
                 get_id=False,
                 class_label=0,
                 transform=None):
        """
        Args:
            image_path (string): path to image folder.
            label_path (string): path to labels file.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.num_classes = num_classes
        self.transform = transform
        self.img_dir = os.path.join(image_path)
        self.get_id = get_id
        self.labels = np.load(label_path, allow_pickle=True)

        if class_label == 0:
            self.labels = self.labels[self.labels[:, 1] >= 37]
        else:
            self.labels = self.labels[self.labels[:, 1] < 37]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx, plot=False):
        img_name = self.labels[idx, 0]
        label_id = img_name.split('_')[0] + '-' + img_name.split('_')[1]
        image = nib.load(os.path.join(self.img_dir, label_id + '.nii.gz'))
        image = np.float32(image.get_fdata())

        if self.transform:
            image = self.transform(image)

        image = torch.from_numpy(image.copy()).float()
        image = image.unsqueeze(0)

        label = np.array([self.labels[idx, 1]])

        label = torch.from_numpy(label).float()
        if self.get_id:
            label = [label, label_id]

        mask = torch.zeros((0))
        return [image, label, mask]


class DHCP_LESIONS(torch.utils.data.Dataset):
    """3D MRI dataset loader"""

    def __init__(self, image_path='/home/cb19/Documents/3d_data_registered_Affine_NMI/masked_images',
                 label_path='/data/dHCP/labels_ants_full.pkl',
                 mask_path='',
                 group='train',
                 task='classification',
                 num_classes=2,
                 get_id=False,
                 filter_labels=True,
                 class_label = None,
                 transform=None):
        """
        Args:
            image_path (string): path to image folder.
            label_path (string): path to labels file.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.task = task
        self.num_classes = num_classes
        self.transform = transform
        self.img_dir = os.path.join(image_path)
        self.mask_dir = mask_path
        image_paths = sorted(os.listdir(self.img_dir))
        self.image_id = None
        self.get_id = get_id
        file = open(os.path.join(label_path), 'rb')

        if group == 'test':
            image_paths = sorted(os.listdir(self.mask_dir))

        self.labels = pickle.load(file)
        # print(len(self.labels))
        if filter_labels and group != 'test':
            self.labels.drop(self.labels.loc[self.labels['scan_ga'] < 37].index, inplace=True)
        # print(len(self.labels))

        # extract ids of certain class - keep only labels of one class
        if not(class_label is None):
            # print(len(self.labels))
            self.labels = self.labels[self.labels['is_prem'] == class_label]
            # print(len(self.labels))

        remove_ind = []
        i = 0
        # check which images are present in labels
        for img in image_paths:
            f = img.split('_')[0]
            f = f.split('-')
            subject = f[0]
            session = f[1].split('.')[0]
            if not (any(self.labels['id'].str.match(subject))):
                remove_ind.append(i)
            elif not (any(self.labels['session'] == int(session))):
                remove_ind.append(i)
            elif any(self.labels['id'].str.match(subject)):
                temp = self.labels.loc[self.labels['id'] == subject]
                if not (any((temp['session'] == int(session)).values)):
                    remove_ind.append(i)
            i = i + 1

        # remove images without labels
        # print(len(image_paths))
        # print(self.labels['id'].tolist())
        image_paths = [i for j, i in enumerate(image_paths) if j not in remove_ind]
        # print(len(image_paths))

        # print('Birth age stats: min, max, mean, std')
        # print(self.labels['birth_ga'].min())
        # print(self.labels['birth_ga'].max())
        # print(self.labels['birth_ga'].mean())
        # print(self.labels['birth_ga'].std())
        #
        # print('Scan age stats: min, max, mean, std')
        # print(self.labels['scan_ga'].min())
        # print(self.labels['scan_ga'].max())
        # print(self.labels['scan_ga'].mean())
        # print(self.labels['scan_ga'].std())

        self.image_paths = sorted(image_paths)
        # print(len(self.image_paths))

        if group == 'train':
            temp_paths = sorted(os.listdir(self.img_dir))
            only_files = [f for f in listdir(self.mask_dir) if isfile(join(self.mask_dir, f))]
            test_ids = [f for f in only_files if f.endswith('nii.gz')]
            for im in temp_paths:
                image_name = im.split('.')[0]
                if any(image_name in s for s in test_ids):
                    try:
                        self.image_paths.remove(im)
                    except:
                        continue
        # print(len(self.image_paths))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx, plot=False):
        img_name = sorted(self.image_paths)[idx]
        image = nib.load(os.path.join(self.img_dir, img_name))
        image = np.float32(image.get_fdata())

        if not self.mask_dir is None:
            try:
                mask = nib.load(os.path.join(self.mask_dir, img_name))
                mask = np.float32(mask.get_fdata())
            except:
                mask = np.zeros((0))

        if plot:
            ind = 50
            sag_slice = image[:, :, ind]
            cor_slice = image[:, ind, :]
            axi_slice = image[ind, :, :]
            fig = plt.figure()
            a = fig.add_subplot(2, 3, 1)
            imgplot = plt.imshow(np.rot90(sag_slice))
            a.axis('off')
            a.set_title('Before aug')
            a = fig.add_subplot(2, 3, 2)
            imgplot = plt.imshow(np.rot90(cor_slice))
            a.axis('off')
            a = fig.add_subplot(2, 3, 3)
            imgplot = plt.imshow(np.rot90(axi_slice))
            a.axis('off')

            print(np.min(image))
            print(np.max(image))
            print(np.mean(image))
            print(np.std(image))

        if self.transform:
            image = self.transform(image)
            if not self.mask_dir is None and mask.shape[0] != 0:
                mask = self.transform(mask)
                mask = torch.from_numpy(mask.copy()).float()
                mask = mask.unsqueeze(0)

        if plot:
            print(np.min(image))
            print(np.max(image))
            print(np.mean(image))
            print(np.std(image))

            ind = 50
            sag_slice = image[:, :, ind]
            cor_slice = image[:, ind, :]
            axi_slice = image[ind, :, :]

            a = fig.add_subplot(2, 3, 4)
            imgplot = plt.imshow(np.rot90(sag_slice))
            a.axis('off')
            a.set_title('After aug')
            a = fig.add_subplot(2, 3, 5)
            imgplot = plt.imshow(np.rot90(cor_slice))
            a.axis('off')
            a = fig.add_subplot(2, 3, 6)
            imgplot = plt.imshow(np.rot90(axi_slice))
            a.axis('off')
            plt.show()

        image = torch.from_numpy(image.copy()).float()
        image = image.unsqueeze(0)

        f = img_name.split('-')
        subject = f[0]
        session = int(f[1].split('.')[0])

        label = self.labels.loc[self.labels['id'] == subject]
        label = label.loc[label['session'] == session]
        label_id = str(label['id'].to_numpy()[0])+ '-' + str(label['session'].to_numpy()[0])
        label = label['birth_ga'].to_numpy()

        label = torch.from_numpy(label).float()
        if self.get_id:
            label = [label, label_id]

        return [image, label, mask]
