import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from PIL import Image
import pandas as pd
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random


## NYUv2 detph dataset load

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}


def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile

    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list(
        (row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))

    from sklearn.utils import shuffle
    nyu2_train = shuffle(nyu2_train, random_state=0)

    # if True: nyu2_train = nyu2_train[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))
        sample = {'image': image, 'depth': depth}
        if self.transform: sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)


class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = self.to_tensor(image)

        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:
            depth = self.to_tensor(depth).float() * 1000

        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])


def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])


def getTrainingTestingData(batch_size):
    data, nyu2_train = loadZipToMem('./dataset/Nyudepthv2/nyu_data.zip')
    data, nyu2_test = loadZipToMem('./dataset/Nyudepthv2/nyu_test.zip')

    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = depthDatasetMemory(data, nyu2_test, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size,
                                                                                  shuffle=False)


# if __name__ == '__main__':
#     batch_size = 1
#
#     train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)
#
#     for i, sample_batched in enumerate(train_loader):
#         print(sample_batched['image'].size(), sample_batched['depth'].size())


#
# class KittiDataset(Dataset):
#     def __init__(self, dataset_folder, is_test=False):
#         self.is_test = is_test
#
#         self.transforms = A.Compose([
#             A.HorizontalFlip(),
#             A.RandomCrop(352, 704, True)
#         ], additional_targets={"label": "image"})
#
#         # This is a placeholder, you can simply put augmentations inside the list below to apply transformations to test
#         # set
#         self.test_transforms = A.Compose([
#         ], additional_targets={"label": "image"})
#
#         self.image_only_transforms = A.Compose([
#             A.RandomBrightnessContrast(),
#             A.RandomGamma(),
#             A.RGBShift(10),
#             A.Normalize()
#         ])
#
#         self.test_image_only_transforms = A.Compose([
#             A.Normalize(always_apply=True)
#         ])
#
#         self.train_drives = []
#         self.test_drives = []
#
#         self.inputs_path = os.path.join(dataset_folder, "inputs")
#         self.outputs_path = os.path.join(dataset_folder, "data_depth_annotated")
#         train_drives_path = os.path.join(self.outputs_path, "train")
#         print(train_drives_path)
#         test_drives_path = os.path.join(self.outputs_path, "val")
#
#         self.img_path = os.path.join("image_03", "data")
#         self.velodyne_path = "velodyne_points\data"
#
#         self.label_images_path = os.path.join("proj_depth", "groundtruth", "image_03")
#
#         # Extracts focal length in pixels of cameras used in drives
#         self.drive_focal_lengths = {}
#         for drive_name in os.listdir(self.inputs_path):
#             calib_path = os.path.join(self.inputs_path, drive_name, "calib_cam_to_cam.txt")
#             calib = {}
#             with open(calib_path, encoding="utf8") as f:
#                 for line in f:
#                     line_name, line_data = line.split(":")[:2]
#                     calib[line_name] = line_data.split(" ")
#             self.drive_focal_lengths[drive_name] = float(calib["P_rect_03"][1])
#
#         # Get folder names of drives that will be used in training
#         for drive in os.listdir(train_drives_path):
#             if ("drive" in drive):
#                 train_drive_images_path = os.path.join(train_drives_path, drive, self.label_images_path)
#                 train_drive_images = []
#                 for train_drive_image in os.listdir(train_drive_images_path):
#                     train_drive_images.append(train_drive_image)
#                 self.train_drives.append([len(train_drive_images), drive, train_drive_images])
#
#         # Get folder names of drives that will be used in testing
#         for drive in os.listdir(test_drives_path):
#             if ("drive" in drive):
#                 test_drive_images_path = os.path.join(test_drives_path, drive, self.label_images_path)
#                 test_drive_images = []
#                 for test_drive_image in os.listdir(test_drive_images_path):
#                     test_drive_images.append(test_drive_image)
#                 self.test_drives.append([len(test_drive_images), drive, test_drive_images])
#
#         self.total_len = 0
#         if (is_test):
#             self.drives = self.test_drives
#             self.drive_labels_path = test_drives_path
#             for test_drive_len, _, _ in self.test_drives:
#                 self.total_len += test_drive_len
#         else:
#             self.drives = self.train_drives
#             self.drive_labels_path = train_drives_path
#             for train_drive_len, _, _ in self.train_drives:
#                 self.total_len += train_drive_len
#
#     def load_label_img(self, drive_path, drive_img):
#         img_path = os.path.join(self.drive_labels_path, drive_path, self.label_images_path, drive_img)
#
#         depth_map = np.asarray(Image.open(img_path), np.float32)
#         depth_map = np.expand_dims(depth_map, axis=2) / 256.0
#
#         self.last_input_path = img_path
#         return depth_map
#
#     def load_input_img(self, drive_path, drive_img):
#         drive = drive_path.split("_drive_")[0]
#         img_path = os.path.join(self.inputs_path, drive, drive_path, self.img_path, drive_img)
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.array(img, dtype=np.uint8)
#
#         self.last_output_path = img_path
#         return img
#
#     def crop_img(self, img):
#         height, width, channels = img.shape
#         top, left = int(height - 352), int((width - 1216) / 2)
#         return img[top:top + 352, left:left + 1216]
#
#     def __getitem__(self, item):
#         for drive_len, drive_path, drive_image_paths in self.drives:
#             if (item < drive_len):
#                 label_img = self.crop_img(self.load_label_img(drive_path, drive_image_paths[item]))
#                 input_img = self.crop_img(self.load_input_img(drive_path, drive_image_paths[item]))
#                 data = {'image': input_img, 'label': label_img}
#
#                 if self.is_test:
#                     data = self.test_transforms(**data)
#                 else:
#                     data = self.transforms(**data)
#
#                 if self.is_test:
#                     data = self.test_image_only_transforms(**data)
#                 else:
#                     data = self.image_only_transforms(**data)
#
#                 data["image"] = torch.tensor(data["image"]).float().transpose(0, 2).transpose(1, 2)
#                 data["label"] = torch.tensor(data["label"]).float().transpose(0, 2).transpose(1, 2)
#
#                 drive = "_".join(drive_path.split("_")[:3])  # Extracts drive date from file path
#                 data["focal_length"] = torch.tensor(self.drive_focal_lengths[drive])
#                 return data
#             else:
#                 # Item isnt in this drive, search in next drive folder
#                 item -= drive_len
#
#     def __len__(self):
#         return self.total_len
#
#
# def KittiDataLoader(batch_size, dataset_folder, is_test=False):
#     dataset = KittiDataset(dataset_folder, is_test)
#     return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)