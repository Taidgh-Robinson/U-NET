import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from logger_config import logger


""" 
Not something we train but we could if we wanted to.
Can also just load state dict from regular PetUNet, they have the layers
Used during demo to show data move through the model
"""


class PetUNetWithLogging(nn.Module):
    def _conv3x3(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3
        )

    def _upConvLayer(self, in_channels, out_channels):
        return nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2
        )

    def __init__(self):
        super(PetUNet, self).__init__()

        # Convolutional Layers
        # 3x3 Channel Upsampling
        self.conv3x3_1LayerTo64Layers = self._conv3x3(1, 64)
        self.conv3x3_64LayersTo128Layers = self._conv3x3(64, 128)
        self.conv3x3_128LayersTo256Layers = self._conv3x3(128, 256)
        self.conv3x3_256LayersTo512Layers = self._conv3x3(256, 512)
        self.conv3x3_512LayersTo1024Layers = self._conv3x3(512, 1024)

        # 3x3 No Channel Changes - Encoder
        self.conv3x3_64LayersTo64Layers_encoder = self._conv3x3(64, 64)
        self.conv3x3_128LayersTo128Layers_encoder = self._conv3x3(128, 128)
        self.conv3x3_256LayersTo256Layers_encoder = self._conv3x3(256, 256)
        self.conv3x3_512LayersTo512Layers_encoder = self._conv3x3(512, 512)

        # 3x3 No Channel Changes - Decoder
        self.conv3x3_64LayersTo64Layers_decoder = self._conv3x3(64, 64)
        self.conv3x3_128LayersTo128Layers_decoder = self._conv3x3(128, 128)
        self.conv3x3_256LayersTo256Layers_decoder = self._conv3x3(256, 256)
        self.conv3x3_512LayersTo512Layers_decoder = self._conv3x3(512, 512)

        # 3x3 No Channel Changes - Bottleneck
        self.conv3x3_1024LayersTo1024Layers = self._conv3x3(1024, 1024)

        # 3x3 Channel Downsampling
        self.conv3x3_1024LayersTo512Layers = self._conv3x3(1024, 512)
        self.conv3x3_512LayersTo256Layers = self._conv3x3(512, 256)
        self.conv3x3_256LayersTo128Layers = self._conv3x3(256, 128)
        self.conv3x3_128LayersTo64Layers = self._conv3x3(128, 64)

        # Up conv-layers
        self.conv2x2_1024To512Layers = self._upConvLayer(1024, 512)
        self.conv2x2_512LayersTo256Layers = self._upConvLayer(512, 256)
        self.conv2x2_256LayersTo128Layers = self._upConvLayer(256, 128)
        self.conv2x2_128LayersTo64Layers = self._upConvLayer(128, 64)
        # Max Pool layer
        self.maxpool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output layer
        self.conv1x1_64LayersTo2Layers = nn.Conv2d(
            in_channels=64, out_channels=2, kernel_size=1
        )

    def forward(self, x):
        """--- Encoder Section ---"""
        x = F.relu(self.conv3x3_1LayerTo64Layers(x))
        logger.debug(f"Size after pass 1: {x.size()}")
        x = F.relu(self.conv3x3_64LayersTo64Layers_encoder(x))
        logger.debug(f"Size after pass 2: {x.size()}")
        skip_layer_1 = x
        x = self.maxpool2x2(x)
        logger.debug(f"Size after Max Pool 1: {x.size()}")
        x = F.relu(self.conv3x3_64LayersTo128Layers(x))
        logger.debug(f"Size after pass 3: {x.size()}")
        x = F.relu(self.conv3x3_128LayersTo128Layers_encoder(x))
        logger.debug(f"Size after pass 4: {x.size()}")
        skip_layer_2 = x
        x = self.maxpool2x2(x)
        logger.debug(f"Size after Max Pool 2: {x.size()}")
        x = F.relu(self.conv3x3_128LayersTo256Layers(x))
        logger.debug(f"Size after pass 5: {x.size()}")
        x = F.relu(self.conv3x3_256LayersTo256Layers_encoder(x))
        logger.debug(f"Size after pass 6: {x.size()}")
        skip_layer_3 = x
        x = self.maxpool2x2(x)
        logger.debug(f"Size after Max Pool 3: {x.size()}")
        x = F.relu(self.conv3x3_256LayersTo512Layers(x))
        logger.debug(f"Size after pass 7: {x.size()}")
        x = F.relu(self.conv3x3_512LayersTo512Layers_encoder(x))
        logger.debug(f"Size after pass 8: {x.size()}")
        skip_layer_4 = x
        x = self.maxpool2x2(x)
        logger.debug(f"Size after Max Pool 4: {x.size()}")
        """ --- Bottleneck section --- """
        x = F.relu(self.conv3x3_512LayersTo1024Layers(x))
        logger.debug(f"Size after pass 9: {x.size()}")
        x = F.relu(self.conv3x3_1024LayersTo1024Layers(x))
        logger.debug(f"Size after pass 10: {x.size()}")
        x = self.conv2x2_1024To512Layers(x)
        logger.debug(f"Size after up conv 1: {x.size()}")
        """ --- Decoder Section --- """
        x = torch.cat([TF.center_crop(skip_layer_4, (56, 56)), x], dim=1)
        logger.debug(f"Size after adding skip layer 1: {x.size()}")
        x = F.relu(self.conv3x3_1024LayersTo512Layers(x))
        logger.debug(f"Size after pass 11: {x.size()}")
        x = F.relu(self.conv3x3_512LayersTo512Layers_decoder(x))
        logger.debug(f"Size after pass 12: {x.size()}")
        x = self.conv2x2_512LayersTo256Layers(x)
        logger.debug(f"Size after up conv 2: {x.size()}")
        x = torch.cat([TF.center_crop(skip_layer_3, (104, 104)), x], dim=1)
        logger.debug(f"Size after adding skip layer 2: {x.size()}")
        x = F.relu(self.conv3x3_512LayersTo256Layers(x))
        logger.debug(f"Size after pass 13: {x.size()}")
        x = F.relu(self.conv3x3_256LayersTo256Layers_decoder(x))
        logger.debug(f"Size after pass 14: {x.size()}")
        x = self.conv2x2_256LayersTo128Layers(x)
        logger.debug(f"Size after up conv 3: {x.size()}")
        x = torch.cat([TF.center_crop(skip_layer_2, (200, 200)), x], dim=1)
        logger.debug(f"Size after adding skip layer 3: {x.size()}")
        x = F.relu(self.conv3x3_256LayersTo128Layers(x))
        logger.debug(f"Size after pass 15: {x.size()}")
        x = F.relu(self.conv3x3_128LayersTo128Layers_decoder(x))
        logger.debug(f"Size after pass 16: {x.size()}")
        x = self.conv2x2_128LayersTo64Layers(x)
        logger.debug(f"Size after up conv 4: {x.size()}")
        x = torch.cat([TF.center_crop(skip_layer_1, (392, 392)), x], dim=1)
        logger.debug(f"Size after adding skip layer 4: {x.size()}")
        x = F.relu(self.conv3x3_128LayersTo64Layers(x))
        logger.debug(f"Size after pass 17: {x.size()}")
        x = F.relu(self.conv3x3_64LayersTo64Layers_decoder(x))
        logger.debug(f"Size after pass 18: {x.size()}")
        x = self.conv1x1_64LayersTo2Layers(x)
        return x


"""
UNet that is a 1:1 recreation of the model in the white paper
"""


class PetUNet(nn.Module):
    def _conv3x3(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3
        )

    def _upConvLayer(self, in_channels, out_channels):
        return nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2
        )

    def __init__(self):
        super(PetUNet, self).__init__()

        # Convolutional Layers
        # 3x3 Channel Upsampling
        self.conv3x3_1LayerTo64Layers = self._conv3x3(1, 64)
        self.conv3x3_64LayersTo128Layers = self._conv3x3(64, 128)
        self.conv3x3_128LayersTo256Layers = self._conv3x3(128, 256)
        self.conv3x3_256LayersTo512Layers = self._conv3x3(256, 512)
        self.conv3x3_512LayersTo1024Layers = self._conv3x3(512, 1024)

        # 3x3 No Channel Changes - Encoder
        self.conv3x3_64LayersTo64Layers_encoder = self._conv3x3(64, 64)
        self.conv3x3_128LayersTo128Layers_encoder = self._conv3x3(128, 128)
        self.conv3x3_256LayersTo256Layers_encoder = self._conv3x3(256, 256)
        self.conv3x3_512LayersTo512Layers_encoder = self._conv3x3(512, 512)

        # 3x3 No Channel Changes - Decoder
        self.conv3x3_64LayersTo64Layers_decoder = self._conv3x3(64, 64)
        self.conv3x3_128LayersTo128Layers_decoder = self._conv3x3(128, 128)
        self.conv3x3_256LayersTo256Layers_decoder = self._conv3x3(256, 256)
        self.conv3x3_512LayersTo512Layers_decoder = self._conv3x3(512, 512)

        # 3x3 No Channel Changes - Bottleneck
        self.conv3x3_1024LayersTo1024Layers = self._conv3x3(1024, 1024)

        # 3x3 Channel Downsampling
        self.conv3x3_1024LayersTo512Layers = self._conv3x3(1024, 512)
        self.conv3x3_512LayersTo256Layers = self._conv3x3(512, 256)
        self.conv3x3_256LayersTo128Layers = self._conv3x3(256, 128)
        self.conv3x3_128LayersTo64Layers = self._conv3x3(128, 64)

        # Up conv-layers
        self.conv2x2_1024To512Layers = self._upConvLayer(1024, 512)
        self.conv2x2_512LayersTo256Layers = self._upConvLayer(512, 256)
        self.conv2x2_256LayersTo128Layers = self._upConvLayer(256, 128)
        self.conv2x2_128LayersTo64Layers = self._upConvLayer(128, 64)
        # Max Pool layer
        self.maxpool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output layer
        self.conv1x1_64LayersTo2Layers = nn.Conv2d(
            in_channels=64, out_channels=2, kernel_size=1
        )

    def forward(self, x):
        """--- Encoder Section ---"""
        x = F.relu(self.conv3x3_1LayerTo64Layers(x))
        x = F.relu(self.conv3x3_64LayersTo64Layers_encoder(x))
        skip_layer_1 = x
        x = self.maxpool2x2(x)
        x = F.relu(self.conv3x3_64LayersTo128Layers(x))
        x = F.relu(self.conv3x3_128LayersTo128Layers_encoder(x))
        skip_layer_2 = x
        x = self.maxpool2x2(x)
        x = F.relu(self.conv3x3_128LayersTo256Layers(x))
        x = F.relu(self.conv3x3_256LayersTo256Layers_encoder(x))
        skip_layer_3 = x
        x = self.maxpool2x2(x)
        x = F.relu(self.conv3x3_256LayersTo512Layers(x))
        x = F.relu(self.conv3x3_512LayersTo512Layers_encoder(x))
        skip_layer_4 = x
        x = self.maxpool2x2(x)
        """ --- Bottleneck section --- """
        x = F.relu(self.conv3x3_512LayersTo1024Layers(x))
        x = F.relu(self.conv3x3_1024LayersTo1024Layers(x))
        x = self.conv2x2_1024To512Layers(x)
        """ --- Decoder Section --- """
        x = torch.cat([TF.center_crop(skip_layer_4, (56, 56)), x], dim=1)
        x = F.relu(self.conv3x3_1024LayersTo512Layers(x))
        x = F.relu(self.conv3x3_512LayersTo512Layers_decoder(x))
        x = self.conv2x2_512LayersTo256Layers(x)
        x = torch.cat([TF.center_crop(skip_layer_3, (104, 104)), x], dim=1)
        x = F.relu(self.conv3x3_512LayersTo256Layers(x))
        x = F.relu(self.conv3x3_256LayersTo256Layers_decoder(x))
        x = self.conv2x2_256LayersTo128Layers(x)
        x = torch.cat([TF.center_crop(skip_layer_2, (200, 200)), x], dim=1)
        x = F.relu(self.conv3x3_256LayersTo128Layers(x))
        x = F.relu(self.conv3x3_128LayersTo128Layers_decoder(x))
        x = self.conv2x2_128LayersTo64Layers(x)
        x = torch.cat([TF.center_crop(skip_layer_1, (392, 392)), x], dim=1)
        x = F.relu(self.conv3x3_128LayersTo64Layers(x))
        x = F.relu(self.conv3x3_64LayersTo64Layers_decoder(x))
        x = self.conv1x1_64LayersTo2Layers(x)
        return x


class PetUNetColor(nn.Module):
    def _conv3x3(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3
        )

    def _upConvLayer(self, in_channels, out_channels):
        return nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2
        )

    def __init__(self):
        super(PetUNetColor, self).__init__()

        # Convolutional Layers
        # 3x3 Channel Upsampling
        self.conv3x3_1LayerTo64Layers = self._conv3x3(3, 64)
        self.conv3x3_64LayersTo128Layers = self._conv3x3(64, 128)
        self.conv3x3_128LayersTo256Layers = self._conv3x3(128, 256)
        self.conv3x3_256LayersTo512Layers = self._conv3x3(256, 512)
        self.conv3x3_512LayersTo1024Layers = self._conv3x3(512, 1024)

        # 3x3 No Channel Changes - Encoder
        self.conv3x3_64LayersTo64Layers_encoder = self._conv3x3(64, 64)
        self.conv3x3_128LayersTo128Layers_encoder = self._conv3x3(128, 128)
        self.conv3x3_256LayersTo256Layers_encoder = self._conv3x3(256, 256)
        self.conv3x3_512LayersTo512Layers_encoder = self._conv3x3(512, 512)

        # 3x3 No Channel Changes - Decoder
        self.conv3x3_64LayersTo64Layers_decoder = self._conv3x3(64, 64)
        self.conv3x3_128LayersTo128Layers_decoder = self._conv3x3(128, 128)
        self.conv3x3_256LayersTo256Layers_decoder = self._conv3x3(256, 256)
        self.conv3x3_512LayersTo512Layers_decoder = self._conv3x3(512, 512)

        # 3x3 No Channel Changes - Bottleneck
        self.conv3x3_1024LayersTo1024Layers = self._conv3x3(1024, 1024)

        # 3x3 Channel Downsampling
        self.conv3x3_1024LayersTo512Layers = self._conv3x3(1024, 512)
        self.conv3x3_512LayersTo256Layers = self._conv3x3(512, 256)
        self.conv3x3_256LayersTo128Layers = self._conv3x3(256, 128)
        self.conv3x3_128LayersTo64Layers = self._conv3x3(128, 64)

        # Up conv-layers
        self.conv2x2_1024To512Layers = self._upConvLayer(1024, 512)
        self.conv2x2_512LayersTo256Layers = self._upConvLayer(512, 256)
        self.conv2x2_256LayersTo128Layers = self._upConvLayer(256, 128)
        self.conv2x2_128LayersTo64Layers = self._upConvLayer(128, 64)
        # Max Pool layer
        self.maxpool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output layer
        self.conv1x1_64LayersTo2Layers = nn.Conv2d(
            in_channels=64, out_channels=2, kernel_size=1
        )

    def forward(self, x):
        """--- Encoder Section ---"""
        x = F.relu(self.conv3x3_1LayerTo64Layers(x))
        x = F.relu(self.conv3x3_64LayersTo64Layers_encoder(x))
        skip_layer_1 = x
        x = self.maxpool2x2(x)
        x = F.relu(self.conv3x3_64LayersTo128Layers(x))
        x = F.relu(self.conv3x3_128LayersTo128Layers_encoder(x))
        skip_layer_2 = x
        x = self.maxpool2x2(x)
        x = F.relu(self.conv3x3_128LayersTo256Layers(x))
        x = F.relu(self.conv3x3_256LayersTo256Layers_encoder(x))
        skip_layer_3 = x
        x = self.maxpool2x2(x)
        x = F.relu(self.conv3x3_256LayersTo512Layers(x))
        x = F.relu(self.conv3x3_512LayersTo512Layers_encoder(x))
        skip_layer_4 = x
        x = self.maxpool2x2(x)
        """ --- Bottleneck section --- """
        x = F.relu(self.conv3x3_512LayersTo1024Layers(x))
        x = F.relu(self.conv3x3_1024LayersTo1024Layers(x))
        x = self.conv2x2_1024To512Layers(x)
        """ --- Decoder Section --- """
        x = torch.cat([TF.center_crop(skip_layer_4, (56, 56)), x], dim=1)
        x = F.relu(self.conv3x3_1024LayersTo512Layers(x))
        x = F.relu(self.conv3x3_512LayersTo512Layers_decoder(x))
        x = self.conv2x2_512LayersTo256Layers(x)
        x = torch.cat([TF.center_crop(skip_layer_3, (104, 104)), x], dim=1)
        x = F.relu(self.conv3x3_512LayersTo256Layers(x))
        x = F.relu(self.conv3x3_256LayersTo256Layers_decoder(x))
        x = self.conv2x2_256LayersTo128Layers(x)
        x = torch.cat([TF.center_crop(skip_layer_2, (200, 200)), x], dim=1)
        x = F.relu(self.conv3x3_256LayersTo128Layers(x))
        x = F.relu(self.conv3x3_128LayersTo128Layers_decoder(x))
        x = self.conv2x2_128LayersTo64Layers(x)
        x = torch.cat([TF.center_crop(skip_layer_1, (392, 392)), x], dim=1)
        x = F.relu(self.conv3x3_128LayersTo64Layers(x))
        x = F.relu(self.conv3x3_64LayersTo64Layers_decoder(x))
        x = self.conv1x1_64LayersTo2Layers(x)
        return x
