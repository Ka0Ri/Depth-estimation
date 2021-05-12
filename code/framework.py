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
from torchvision.utils import make_grid, save_image
import os
import yaml
from utils import *
from torchvision import models



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
        
        #training and validating
        transformed_train = depthDatasetMemory(data, nyu2_train[:n_used_data], transform=self.default_transform)
        if stage == "fit" or None:
            n_train_data = int(n_used_data * 0.7)
            n_val_data = n_used_data - n_train_data
            self.train, self.val = random_split(transformed_train, lengths=[n_train_data, n_val_data])
        
        if stage == "test" or None:
            #testing
            n_test_data = n_used_data + self.data_config["n_test_data"]
            self.test = depthDatasetMemory(data, nyu2_train[n_used_data:n_test_data], transform=self.no_transform)
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
            up_x = F.interpolate(x, size=[x.size(2) * 2 , x.size(3) * 2], mode='bilinear', align_corners=True)
            return self.leakyreluB(self.convB(self.convA(up_x)))
        else:
            up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
            return self.leakyreluB(self.convB(self.convA(torch.cat([up_x, concat_with], dim=1))))

class DownSample(nn.Sequential):
    def __init__(self, skip_input, output_features, grid_size):
        super(DownSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=grid_size, stride=grid_size)
        self.leakyreluA = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leakyreluA((self.convA(x)))
   

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
        self.conv3 = nn.Conv2d(features//16 + 3, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, features=None):
       
        x_d0 = self.conv2(F.relu(x))
        x_d1 = self.up1(x_d0)
        x_d2 = self.up2(x_d1)
        x_d3 = self.up3(x_d2)
        x_d4 = self.up4(x_d3)
        if features is None:
            return self.conv3(x_d4)
        else:
            out = self.conv3(torch.cat([x_d4, features], dim=1))
            return out


class MultiScaleDecoder(nn.Module):
    def __init__(self, config):
        super(MultiScaleDecoder, self).__init__()

        num_features = config["num_features"]
        decoder_width = config["decoder_width"]
        base_grid = config["base_grid"]
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.down1 = DownSample(3, (features//1), grid_size=base_grid // 2)
        self.down2 = DownSample(3, (features//2), grid_size=base_grid // 4)
        self.down3 = DownSample(3, (features//4), grid_size=base_grid // 8)
        self.down4 = DownSample(3, (features//8), grid_size=base_grid // 16)

        self.up1 = UpSample(skip_input= 2*(features//1), output_features=features//2)
        self.up2 = UpSample(skip_input= 2*(features//2),  output_features=features//4)
        self.up3 = UpSample(skip_input= 2*(features//4),  output_features=features//8)
        self.up4 = UpSample(skip_input= 2*(features//8),  output_features=features//16)
        self.conv3 = nn.Conv2d(features//16 + 3, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, features=None):
       
        x_d0 = self.conv2(x)
        x_d1 = self.up1(x_d0, self.down1(features))
        x_d2 = self.up2(x_d1, self.down2(features))
        x_d3 = self.up3(x_d2, self.down3(features))
        x_d4 = self.up4(x_d3, self.down4(features))
        out = self.conv3(torch.cat([x_d4, features], dim=1))

        return out

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
        if(self.model_config["Decoder"]["multiScale"] == True):
            self.decoder = MultiScaleDecoder(config=self.model_config["Decoder"])
        else:
            self.decoder = Decoder(config=self.model_config["Decoder"])
        # self.fine_tune = nn.Conv2d(4, 1, kernel_size=3, padding=1)


    def forward(self, x):
        h, att = self.encoder(x)
        
        # h size = [batchsize, 1 + gridsize (e.g 512 = 32x16 -> gridsize = 32x32), 768]
        h = h[:,1:] # remove class token
        gridsize = h.shape[1]
        sq_gridsize = int(gridsize ** 0.5)
        h = h.transpose(1, 2).contiguous()
        h = h.reshape(-1, 768, sq_gridsize, sq_gridsize) #h size = [batchsize, 768, sq_gridsize, sq_gridsize]
        
        up_samplings = self.decoder(h, x) #up_samplings size = [batchsize, 1, 16 * sq_gridsize, 16 * sq_gridsize]
        
        depths = up_samplings
        ## sigmoid or tanh

        return torch.sigmoid(depths)

    # def forward(self, x):
    #     h, att = self.encoder(x)
        
    #     # h size = [batchsize, 1 + gridsize (e.g 512 = 32x16 -> gridsize = 32x32), 768]
    #     h = h[:,1:] # remove class token
    #     gridsize = h.shape[1]
    #     sq_gridsize = int(gridsize ** 0.5)
    #     h = h.reshape(-1, 768, 1, 1) #h size = [batchsize x gridsize, 768, 1, 1]
        
    #     up_samplings = self.decoder(h) #up_samplings size = [batchsize x gridsize, 1, 16, 16]
    #     up_samplings = up_samplings.reshape(-1, sq_gridsize, sq_gridsize, 16, 16)
    #     up_samplings = up_samplings.transpose(2, 3).contiguous() #up_samplings size = [batchsize, grid, 16, grid, 16]
    #     up_samplings = up_samplings.reshape(-1, 1, 16 * sq_gridsize, 16 * sq_gridsize)
        
    #     # depths = up_samplings

    #     depths = self.fine_tune(torch.cat([F.leaky_relu(up_samplings, 0.2), x], dim=1))
    #     ## sigmoid or tanh

    #     return torch.sigmoid(depths)

    def configure_optimizers(self):
        optimier = torch.optim.Adam(self.parameters(), lr=self.training_config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimier, step_size = self.training_config["step"],
                                                        gamma=self.training_config["gamma"])
        return [optimier], [scheduler]

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
        errors = compute_errors(groundtruths, predictions)
        return {'groundtruths': groundtruths, 'predictions': predictions, 'errors': errors}

    def validation_epoch_end(self, outs):
        errors = (0, 0, 0, 0, 0, 0, 0)
        n = 0
        for batch in outs:
            errors = tuple(map(lambda x, y: x + y, batch['errors'], errors))
            n += 1
        errors = tuple(map(lambda x: x/n, errors))
        print(errors)
        grid_images_inputs = make_grid(outs[0]['groundtruths'].cpu(), nrow=4)
        self.logger.experiment.add_image("inputs images", grid_images_inputs, 0, dataformats="CHW")
        save_image(grid_images_inputs, self.logger.log_dir + '/gt.png')
        grid_images_predictions = make_grid(outs[0]['predictions'].cpu(), nrow=4)
        self.logger.experiment.add_image("predicted images", grid_images_predictions, 0, dataformats="CHW")
        save_image(grid_images_predictions, self.logger.log_dir + '/predict.png')

    def test_step(self, test_batch, batch_idx):

        x, groundtruths = test_batch['image'], test_batch['depth']
        predictions = self.forward(x)
        errors = compute_errors(groundtruths, predictions)
        return {'groundtruths': groundtruths, 'predictions': predictions, 'errors': errors}

    def test_epoch_end(self, outs):
        errors = (0, 0, 0, 0, 0, 0, 0)
        n = 0
        for i, batch in enumerate(outs):
            errors = tuple(map(lambda x, y: x + y, batch['errors'], errors))
            n += 1

            colored_gt = colored_batch_depthmap(batch['groundtruths'])
            grid_images_inputs = make_grid(colored_gt, nrow=4)
            save_image(grid_images_inputs, self.logger.log_dir +  '/gt_{i}.png'.format(i = i))
            colored_predictions = colored_batch_depthmap(batch['predictions'])
            grid_images_predictions = make_grid(colored_predictions, nrow=4)
            save_image(grid_images_predictions, self.logger.log_dir + '/predict_{i}.png'.format(i = i))
        errors = tuple(map(lambda x: x/n, errors))
        print(errors)



class ResNetDepthEstimation(LightningModule):
    def __init__(self, config):
        super(ResNetDepthEstimation, self).__init__()

        self.model_config = config["ResNet-model"]
        self.training_config = config["training"]
        
        # Encoder
        original_model = models.resnet152(pretrained=self.model_config["pretrained"])
         
        

        if(self.model_config['fine_tune'] == True):
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

        # Decoder
        self.decoder = Decoder(config=self.model_config["Decoder"])
        # self.decoder = MultiScaleDecoder(config=self.model_config["Decoder"])


    def forward(self, x):
        h = self.encoder(x) #h size = [batchsize, 768, 14, 14]
        up_samplings = self.decoder(h, x) #up_samplings size = [batchsize, 1, 224, 224]
        
        depths = up_samplings

        # depths = self.fine_tune())
        ## sigmoid or tanh

        return torch.sigmoid(depths)

    def configure_optimizers(self):
        optimier = torch.optim.Adam(self.parameters(), lr=self.training_config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimier, step_size = self.training_config["step"],
                                                        gamma=self.training_config["gamma"])
        return [optimier], [scheduler]

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
        errors = compute_errors(groundtruths, predictions)
        return {'groundtruths': groundtruths, 'predictions': predictions, 'errors': errors}

    def validation_epoch_end(self, outs):
        errors = (0, 0, 0, 0, 0, 0, 0)
        n = 0
        for batch in outs:
            errors = tuple(map(lambda x, y: x + y, batch['errors'], errors))
            n += 1
        errors = tuple(map(lambda x: x/n, errors))
        print(errors)
        grid_images_inputs = make_grid(outs[0]['groundtruths'].cpu(), nrow=4)
        self.logger.experiment.add_image("inputs images", grid_images_inputs, 0, dataformats="CHW")
        save_image(grid_images_inputs, self.logger.log_dir + '/gt.png')
        grid_images_predictions = make_grid(outs[0]['predictions'].cpu(), nrow=4)
        self.logger.experiment.add_image("predicted images", grid_images_predictions, 0, dataformats="CHW")
        save_image(grid_images_predictions, self.logger.log_dir + '/predict.png')

    def test_step(self, test_batch, batch_idx):

        x, groundtruths = test_batch['image'], test_batch['depth']
        predictions = self.forward(x)
        errors = compute_errors(groundtruths, predictions)
        return {'groundtruths': groundtruths, 'predictions': predictions, 'errors': errors}

    def test_epoch_end(self, outs):
        errors = (0, 0, 0, 0, 0, 0, 0)
        n = 0
        for batch in outs:
            errors = tuple(map(lambda x, y: x + y, batch['errors'], errors))
            n += 1
        errors = tuple(map(lambda x: x/n, errors))
        print(errors)
        grid_images_inputs = make_grid(outs[0]['groundtruths'].cpu(), nrow=4)
        save_image(grid_images_inputs, 'test_result/gt.png')
        grid_images_predictions = make_grid(outs[0]['predictions'].cpu(), nrow=4)
        save_image(grid_images_predictions, 'test_result/predict.png')


if __name__ == '__main__':

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    data_module = NYUDataModule(config=config)
   
    if(config["model_name"] == 'ResNet'):
        model = ResNetDepthEstimation(config=config)
    else:
        model = ViTDepthEstimation(config=config)

    trainer = Trainer(gpus=1, max_epochs=config["training"]["epochs"])

    #for training
    # trainer.fit(model, data_module)

    #for testing
    model_test = ViTDepthEstimation.load_from_checkpoint('lightning_logs/version_4/checkpoints/epoch=46-step=20585.ckpt',config=config)
    trainer.test(model_test, datamodule=data_module)

    