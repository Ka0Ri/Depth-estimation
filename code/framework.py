from numpy.lib.utils import deprecate
from pytorch_lightning.core import datamodule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule, Trainer
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ViTmodel import VisionTransformer, CONFIGS
from DataLoader import depthDatasetMemory, getDefaultTrainTransform, getNoTransform, loadZipToMem
from torch.utils.data import random_split, DataLoader
import os
import yaml

class NYUDataModule(LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_config = config["dataset"]

        size = eval(self.data_config["input_shape"])[:2]
        self.no_transform = getNoTransform(size=size)
        self.default_transform = getDefaultTrainTransform(size=size, p = self.data_config["channel_swap"])
        
    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        n_used_data = self.data_config["n_used_data"] # number of actual data used
        data, nyu2_train = loadZipToMem(os.path.join(os.path.dirname(os.getcwd()), self.data_config['path']))
        
        if stage == 'fit' or stage is None:
            transformed_train = depthDatasetMemory(data, nyu2_train[:n_used_data], transform=self.default_transform)
            n_train_data = int(n_used_data * 0.7)
            n_val_data = n_used_data - n_train_data
            self.train, self.val = random_split(transformed_train, lengths=[n_train_data, n_val_data])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            n_test_data = n_used_data + 5000
            self.test = depthDatasetMemory(data, nyu2_train[n_used_data:n_test_data], transform=self.default_transform)
            # data, nyu2_test = loadZipToMem('./dataset/nyu_test.zip')
           
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.config['batch_size'], num_workers=self.data_config['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.config['batch_size'], num_workers=self.data_config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.config['batch_size'], num_workers=self.data_config['num_workers'])



class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with=None):
        if concat_with is None:
            up_x = F.interpolate(x, size=[x.size(2) * 2, x.size(3) * 2], mode='bilinear', align_corners=True)
            return self.leakyreluB(self.convB(self.convA(up_x)))
        else:
            up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
            return self.leakyreluB(self.convB(self.convA(torch.cat([up_x, concat_with], dim=1))))

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        num_features = config["num_features"]
        decoder_width = config["decoder_width"]
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8,  output_features=features//16)
        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, features=None):
        if features is None:
            x_d0 = self.conv2(F.relu(x))
            x_d1 = self.up1(x_d0)
            x_d2 = self.up2(x_d1)
            x_d3 = self.up3(x_d2)
            x_d4 = self.up4(x_d3)
        else:
            x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
            x_d0 = self.conv2(F.relu(x_block4))
            x_d1 = self.up1(x_d0, x_block3)
            x_d2 = self.up2(x_d1, x_block2)
            x_d3 = self.up3(x_d2, x_block1)
            x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)


class ViTDepthEstimation(LightningModule):
    def __init__(self, config):
        super(ViTDepthEstimation, self).__init__()

        self.model_config = config["ViT-model"]
        self.training_config = config["training"]
        sz = eval(config['dataset']['input_shape'])[:2]
        
        # Encoder
        ViT_config = CONFIGS[self.model_config['ViTconfig']]
        model = VisionTransformer(config= ViT_config, img_size=sz, num_classes=1, zero_head=True, vis=self.model_config['vis'])

        if(self.model_config['pretrained'] is not None):
            model.load_from(np.load(self.model_config['pretrained']))
        self.encoder = model.transformer

        if(self.model_config['fine_tune'] == True):
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

        # Decoder
        self.decoder = Decoder(config=self.model_config["Decoder"])
        self.fine_tune = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x):
        h, att = self.encoder(x)
        
        # h size = [batchsize, 1 + gridsize (e.g 512 = 32x16 -> gridsize = 32x32), 768]
        h = h[:,1:] # remove class token
        gridsize = h.shape[1]
        h = h.reshape(-1, 768, 1, 1) #h size = [batchsize x gridsize, 768, 1, 1]
        
        up_samplings = self.decoder(h)
        up_samplings = up_samplings.reshape(-1, gridsize, 16, 16)
        up_samplings = up_samplings.reshape(-1, 1, 16 * int(gridsize ** 0.5), 16 * int(gridsize ** 0.5))

        depths = self.fine_tune(torch.cat([up_samplings, x], dim=1))

        return depths

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.training_config["lr"])

    def MSE_loss(self, predictions, groundtruths):
        return F.mse_loss(predictions, groundtruths)

    def Custom_loss(self, predictions, groundtruths):
        return

    def training_step(self, train_batch, batch_idx):
        x, groundtruths = train_batch['image'], train_batch['depth']
        predictions = self.forward(x)
        loss = self.MSE_loss(predictions, groundtruths)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, groundtruths = val_batch['image'], val_batch['depth']
        predictions = self.forward(x)
        loss = self.MSE_loss(predictions, groundtruths)
        self.log('val_loss', loss)


if __name__ == '__main__':

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    data_module = NYUDataModule(config=config)

    model = ViTDepthEstimation(config=config)

    trainer = Trainer(gpus=1, max_epochs=config["training"]["epochs"])
    trainer.fit(model, data_module)

    