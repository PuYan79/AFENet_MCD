import os
import cv2
import torch.utils.data
import numpy as np
import pandas as pd

def read_directory(directory_name,label=False):
    array_of_img = []
    files = os.listdir(directory_name)
    files.sort(key=lambda x: int(x[0:-4]))
    inx = 0
    for filename in files:
        img = cv2.imread(directory_name + "/" + filename)
        inx += 1
        if label:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        array_of_img.append(img)
    return array_of_img


def read_directory_reshape(directory_name,label=False,reshape=False,size=[256,256]):
    array_of_img = []
    files = os.listdir(directory_name)
    files.sort(key=lambda x: int(x[0:-4]))
    inx = 0
    for filename in files:
        if os.path.exists(directory_name + "/" + filename):
            img = cv2.imread(directory_name + "/" + filename)
            print('------------------' + str(inx) + '------------------')
            inx += 1
        else:
            print(f"Error: Image file not found at {directory_name + filename}")
        if img is None:
            print('----------None file name:', directory_name + "/" + filename)
        if label:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.where(img == 1, 1, 0).astype(np.uint8)
        if reshape:
            img = cv2.resize(img,size)
        array_of_img.append(img)
    return array_of_img


# Dataset for GZ
class GZ_Dataset(torch.utils.data.Dataset):
    def __init__(self, move='train', transform=None):
        super(GZ_Dataset, self).__init__()
        data_MT_CDD = 'Dataset/GZ'
        dataset_train_A = '/train/image1'
        dataset_train_B = '/train/image2'
        dataset_train_out = '/train/label'
        dataset_test_A = '/test/image1'
        dataset_test_B = '/test/image2'
        dataset_test_out = '/test/label'
        seq_img_A = []
        seq_img_B = []
        seq_out = []
        if move == 'train':
            seq_img_A = read_directory(data_MT_CDD + dataset_train_A)
            seq_img_B = read_directory(data_MT_CDD + dataset_train_B)
            seq_out = read_directory(data_MT_CDD + dataset_train_out,label=True)
        elif move == 'test':
            seq_img_A = read_directory(data_MT_CDD + dataset_test_A)
            seq_img_B = read_directory(data_MT_CDD + dataset_test_B)
            seq_out = read_directory(data_MT_CDD + dataset_test_out, label=True)
        self.seq_img_A = seq_img_A
        self.seq_img_B= seq_img_B
        self.seq_out = seq_out
        self.transform = transform

    def __getitem__(self, index):
        imgs_A = self.seq_img_A[index]
        imgs_B = self.seq_img_B[index]
        label = self.seq_out[index]
        if self.transform is not None:
            imgs_A = self.transform(imgs_A)
            imgs_B = self.transform(imgs_B)
            label = self.transform(label)
        return imgs_A, imgs_B, label

    def __len__(self):
        return len(self.seq_out)


# Dataset for MT-OSCD
class MT_OSCDDataset(torch.utils.data.Dataset):
    def __init__(self, move='train', transform=None):
        super(MT_OSCDDataset, self).__init__()
        data_MT_OSCD = '/Dataset/MT-OSCD'
        dataset_train_rgb = '/train/opt/t1'
        dataset_train_sar = '/train/sar/t2'
        dataset_train_label = '/train/mask'
        dataset_test_rgb = '/test/opt/t1'
        dataset_test_sar = '/test/sar/t2'
        dataset_test_label = '/test/mask'
        seq_img_rgb = []
        seq_img_sar = []
        seq_label = []
        if move == 'train':
            seq_img_rgb = read_directory(data_MT_OSCD + dataset_train_rgb)
            seq_img_sar = read_directory(data_MT_OSCD + dataset_train_sar)
            seq_label = read_directory(data_MT_OSCD + dataset_train_label,label=True)
        elif move == 'test':
            seq_img_rgb = read_directory(data_MT_OSCD + dataset_test_rgb)
            seq_img_sar = read_directory(data_MT_OSCD + dataset_test_sar)
            seq_label = read_directory(data_MT_OSCD + dataset_test_label, label=True)
        self.seq_img_rgb = seq_img_rgb
        self.seq_img_sar= seq_img_sar
        self.seq_label = seq_label
        self.transform = transform

    def __getitem__(self, index):
        imgs_rgb = self.seq_img_rgb[index]
        imgs_sar = self.seq_img_sar[index]
        label = self.seq_label[index]
        if self.transform is not None:
            imgs_rgb = self.transform(imgs_rgb)
            imgs_sar = self.transform(imgs_sar)
            label = self.transform(label)
        return imgs_rgb, imgs_sar, label

    def __len__(self):
        return len(self.seq_label)


# Dataset for CDD
class CDD_Dataset(torch.utils.data.Dataset):
    # img1-sat img2-uav
    def __init__(self, move='train' , transform=None):
        super(CDD_Dataset, self).__init__()
        self.dir = 'Dataset/CDD'
        if move == 'train':
            self.path = os.path.join(self.dir ,'train')
            self.images = os.listdir(os.path.join(self.path, 'A'))
            self.images.sort(key=lambda x: int(x[0:-4]))
        elif move == 'test':
            self.path = os.path.join(self.dir, 'test')
            self.images = os.listdir(os.path.join(self.path, 'A'))
            self.images.sort(key=lambda x: int(x[0:-4]))
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.images[idx]
        A_file = os.path.join(self.path, 'A', filename)
        B_file = os.path.join(self.path, 'B', filename)
        OUT_file = os.path.join(self.path, 'OUT', filename)
        img1 = cv2.imread(A_file)
        img2 = cv2.imread(B_file)
        lbl = cv2.imread(OUT_file, cv2.IMREAD_UNCHANGED)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            lbl = self.transform(lbl)
        return img1, img2, lbl

    def __len__(self):
        return len(self.images)


# Dataset for LEVIR
class LEVIR_Dataset(torch.utils.data.Dataset):
    def __init__(self, move='train', transform=None):
        super(LEVIR_Dataset, self).__init__()
        data_MT_CDD = 'Dataset/LEVIR'
        dataset_train_A = '/train/A'
        dataset_train_B = '/train/B'
        dataset_train_out = '/train/label'
        dataset_test_A = '/test/A'
        dataset_test_B = '/test/B'
        dataset_test_out = '/test/label'
        seq_img_A = []
        seq_img_B = []
        seq_out = []
        if move == 'train':
            seq_img_A = read_directory(data_MT_CDD + dataset_train_A)
            seq_img_B = read_directory(data_MT_CDD + dataset_train_B)
            seq_out = read_directory(data_MT_CDD + dataset_train_out,label=True)
        elif move == 'test':
            seq_img_A = read_directory(data_MT_CDD + dataset_test_A)
            seq_img_B = read_directory(data_MT_CDD + dataset_test_B)
            seq_out = read_directory(data_MT_CDD + dataset_test_out, label=True)
        self.seq_img_A = seq_img_A
        self.seq_img_B= seq_img_B
        self.seq_out = seq_out
        self.transform = transform

    def __getitem__(self, index):
        imgs_A = self.seq_img_A[index]
        imgs_B = self.seq_img_B[index]
        label = self.seq_out[index]
        if self.transform is not None:
            imgs_A = self.transform(imgs_A)
            imgs_B = self.transform(imgs_B)
            label = self.transform(label)
        return imgs_A, imgs_B, label

    def __len__(self):
        return len(self.seq_out)


# Dataset for MT-HTCD
class HTCD_CSV(torch.utils.data.Dataset):
    def __init__(self, move='train',transform=None):
        super(HTCD_CSV, self).__init__()
        self.path = 'Dataset/MT-HTCD'
        self.sat_mean = np.array([66, 71, 74], np.uint8)
        self.uav_mean = np.array([73, 81, 79], np.uint8)
        self.transform = transform
        if move == 'train':
            images_df = pd.read_csv('MT_HTCD_train.csv.csv')
            self.images = images_df['filename'].tolist()
        elif move == 'test':
            images_df = pd.read_csv('MT_HTCD_test.csv.csv')
            self.images = images_df['filename'].tolist()

    def __getitem__(self, idx):
        filename = self.images[idx]
        t1_file = os.path.join(self.path, 'sat', filename)
        t2_file = os.path.join(self.path, 'uav', filename)
        label_file = os.path.join(self.path, 'label', filename)
        img1 = cv2.imread(t1_file)
        if (img1 is None):
            print(idx)
            print(t1_file)
        img1 -= self.sat_mean
        img_size = img1.shape[:2]
        img2 = cv2.imread(t2_file)
        img2 = cv2.resize(img2, img_size)
        img2 -= self.uav_mean
        lbl = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
        lbl = cv2.resize(lbl, img_size)
        lbl = np.where(lbl==1,1,0).astype(np.uint8)
        is_binary_label = (lbl == 0) | (lbl == 1)
        if not is_binary_label.all():
            print(label_file)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            lbl = self.transform(lbl)*255


        return img1, img2, lbl

    def __len__(self):
        return len(self.images)


class MT_HTCDDataset_1(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super(MT_HTCDDataset_1, self).__init__()
        data_MT_HTCD = '/home/yan/Documents/Dataset/MT-HTCD'
        dataset_uav = '/uav'
        dataset_sat = '/sat'
        dataset_label = '/label'

        seq_label = read_directory_reshape(data_MT_HTCD + dataset_label, label=True, reshape=True)
        seq_img_uav = read_directory_reshape(data_MT_HTCD + dataset_uav,reshape=True,size=[256,256])
        seq_img_sat = read_directory_reshape(data_MT_HTCD + dataset_sat)


        self.seq_img_rgb = seq_img_uav
        self.seq_img_sar= seq_img_sat
        self.seq_label = seq_label
        # self.filenanme = filename
        self.transform = transform

    def __getitem__(self, index):
        imgs_rgb = self.seq_img_rgb[index]
        imgs_sar = self.seq_img_sar[index]
        label = self.seq_label[index]
        # filename = self.filenanme[index]
        if self.transform is not None:
            imgs_rgb = self.transform(imgs_rgb)
            imgs_sar = self.transform(imgs_sar)
            label = self.transform(label) * 255
        return imgs_rgb, imgs_sar, label

    def __len__(self):
        return len(self.seq_label)


# Dataset for WHU
class WHU_Dataset(torch.utils.data.Dataset):
    def __init__(self, move='train', transform=None):
        super(WHU_Dataset, self).__init__()
        data_MT_CDD = 'Dataset/WHU'
        dataset_train_A = '/train/A'
        dataset_train_B = '/train/B'
        dataset_train_out = '/train/label'
        dataset_test_A = '/test/A'
        dataset_test_B = '/test/B'
        dataset_test_out = '/test/label'
        seq_img_A = []
        seq_img_B = []
        seq_out = []
        if move == 'train':
            seq_img_A = read_directory(data_MT_CDD + dataset_train_A)
            seq_img_B = read_directory(data_MT_CDD + dataset_train_B)
            seq_out = read_directory(data_MT_CDD + dataset_train_out,label=True)
        elif move == 'test':
            seq_img_A = read_directory(data_MT_CDD + dataset_test_A)
            seq_img_B = read_directory(data_MT_CDD + dataset_test_B)
            seq_out = read_directory(data_MT_CDD + dataset_test_out, label=True)
        self.seq_img_A = seq_img_A
        self.seq_img_B= seq_img_B
        self.seq_out = seq_out
        self.transform = transform

    def __getitem__(self, index):
        imgs_A = self.seq_img_A[index]
        imgs_B = self.seq_img_B[index]
        label = self.seq_out[index]
        if self.transform is not None:
            imgs_A = self.transform(imgs_A)
            imgs_B = self.transform(imgs_B)
            label = self.transform(label)
        return imgs_A, imgs_B, label

    def __len__(self):
        return len(self.seq_out)


# Dataset for MT-Wuhan
class MT_WuHanDataset(torch.utils.data.Dataset):
    def __init__(self, move='train', transform=None):
        super(MT_WuHanDataset, self).__init__()
        data_MT_WuHan = 'Dataset/MT-WuHan'
        dataset_train_rgb = '/train/rgb'
        dataset_train_sar = '/train/sar'
        dataset_train_label = '/train/mask'
        dataset_test_rgb = '/test/rgb'
        dataset_test_sar = '/test/sar'
        dataset_test_label = '/test/mask'
        seq_img_rgb = []
        seq_img_sar = []
        seq_label = []
        if move == 'train':
            seq_img_rgb = read_directory(data_MT_WuHan + dataset_train_rgb)
            seq_img_sar = read_directory(data_MT_WuHan + dataset_train_sar)
            seq_label = read_directory(data_MT_WuHan + dataset_train_label,label=True)
        elif move == 'test':
            seq_img_rgb = read_directory(data_MT_WuHan + dataset_test_rgb)
            seq_img_sar = read_directory(data_MT_WuHan + dataset_test_sar)
            seq_label = read_directory(data_MT_WuHan + dataset_test_label, label=True)
        self.seq_img_rgb = seq_img_rgb
        self.seq_img_sar= seq_img_sar
        self.seq_label = seq_label
        self.transform = transform

    def __getitem__(self, index):
        imgs_rgb = self.seq_img_rgb[index]
        imgs_sar = self.seq_img_sar[index]
        label = self.seq_label[index]
        if self.transform is not None:
            imgs_rgb = self.transform(imgs_rgb)
            imgs_sar = self.transform(imgs_sar)
            label = self.transform(label)
        return imgs_rgb, imgs_sar, label

    def __len__(self):
        return len(self.seq_label)