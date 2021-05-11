import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
from itertools import permutations
from zipfile import ZipFile
from sklearn.utils import shuffle
import os

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
    
class Resize(object):
    def __init__(self, size):

        self.size = size
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        image = image.resize(self.size)
        depth = depth.resize(self.size)

        return {'image': image, 'depth': depth}


class RandomChannelSwap(object):
    def __init__(self, probability):

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
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
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
        depth = self.to_tensor(depth)

        # if self.is_test:
        #     # depth normalize 0 - 1
        #     # 1000으로 나누거나 곱하는 이유 : 실제 real meter
        #     depth = self.to_tensor(depth).float() / 1000
        # else:
        #     depth = self.to_tensor(depth).float() * 1000

        # # put in expected range
        # depth = torch.clamp(depth, 10, 1000)

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


def getNoTransform(size, is_test=False):
    return transforms.Compose([
        Resize(size),
        ToTensor(is_test=is_test)
    ])


def getDefaultTrainTransform(size, p):
    return transforms.Compose([
        # RandomHorizontalFlip(),
        RandomChannelSwap(p),
        Resize(size),
        ToTensor()
    ])


def getTrainingTestingData(batch_size):
    data, nyu2_train = loadZipToMem(os.path.join(os.path.dirname(os.getcwd()), 'dataset/nyu_data.zip'))
    # data, nyu2_test = loadZipToMem('./dataset/nyu_test.zip')

    transformed_train = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_valid = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())
    # transformed_testing = depthDatasetMemory(data, nyu2_test, transform=getNoTransform())
    

    return DataLoader(transformed_train, batch_size, shuffle=True), DataLoader(transformed_valid, batch_size, shuffle=False)


def load_testloader(path, batch_size=1):
    data, nyu2_train = loadZipToMem(path)
    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())
    return DataLoader(transformed_testing, batch_size, shuffle=False)


if __name__ == '__main__':
    batch_size = 1

    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)
    

    for i, sample_batched in enumerate(train_loader):
        print(sample_batched['image'].size(), sample_batched['depth'].size())
        break