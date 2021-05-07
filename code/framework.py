import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ViTmodel import VisionTransformer, CONFIGS



###########################-----Modifications from here----------------------------------------
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

        self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)
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

class ViTDepthEstimation(nn.Module):
    def __init__(self, config):
        super(ViTDepthEstimation, self).__init__()

        model_config = config["ViT-model"]
        sz = eval(config['dataset']['input_shape'])[0]
        self.vis = model_config['vis']
        self.k = model_config['n_patches']
        
        # Encoder
        ViT_config = CONFIGS[model_config['ViTconfig']]
        model = VisionTransformer(config= ViT_config, img_size=sz, num_classes=1, zero_head=True, vis=self.vis)

        if(model_config['pretrained'] is not None):
            model.load_from(np.load(model_config['pretrained']))
        self.encoder = model.transformer

        # Decoder
        self.decoder = Decoder(config=model_config["Decoder"])

    def forward(self, x):
     
        h, att = self.ft(x)
        
        
        return