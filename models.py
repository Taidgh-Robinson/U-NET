import torch.nn as nn
import torch.nn.functional as F

class PetUNet(nn.Module): 
    def __init__(self):
        super(PetUNet, self).__init__()

        # Convolutional Layers
        # 3x3 Channel Upsampling
        self.conv3x3_1LayerTo64Layers = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv3x3_64LayersTo128Layers = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3x3_128LayersTo256Layers = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv3x3_256LayersTo512Layers = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv3x3_512LayersTo1024Layers = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)

        # 3x3 No Channel Changes
        self.conv3x3_64LayersTo64Layers = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv3x3_128LayersTo128Layers = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv3x3_256LayersTo256Layers = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv3x3_512LayersTo512Layers = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.conv3x3_1024LayersTo1024Layers = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)

        # 3x3 Channel Downsampling
        self.conv3x3_1024LayersTo512Layers = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)
        self.conv3x3_512LayersTo256Layers = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)
        self.conv3x3_256LayersTo128Layers = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.conv3x3_128LayersTo64Layers = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)

        #Up conv-layers
        self.conv2x2_1024To512Layers = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )

        #Max Pool layer
        self.maxpool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv3x3_1LayerTo64Layers(x))
        print(f'Size after pass 1: {x.size()}')
        x = F.relu(self.conv3x3_64LayersTo64Layers(x))
        print(f'Size after pass 2: {x.size()}')
        skip_layer_1 = x
        x = self.maxpool2x2(x)
        print(f'Size after Max Pool 1: {x.size()}')
        x = F.relu(self.conv3x3_64LayersTo128Layers(x))
        print(f'Size after pass 3: {x.size()}')
        x = F.relu(self.conv3x3_128LayersTo128Layers(x))
        print(f'Size after pass 4: {x.size()}')
        skip_layer_2 = x
        x = self.maxpool2x2(x)
        print(f'Size after Max Pool 2: {x.size()}')
        x = F.relu(self.conv3x3_128LayersTo256Layers(x))
        print(f'Size after pass 5: {x.size()}')
        x = F.relu(self.conv3x3_256LayersTo256Layers(x))
        print(f'Size after pass 6: {x.size()}')
        skip_layer_3 = x
        x = self.maxpool2x2(x)
        print(f'Size after Max Pool 3: {x.size()}')
        x = F.relu(self.conv3x3_256LayersTo512Layers(x))
        print(f'Size after pass 7: {x.size()}')
        x = F.relu(self.conv3x3_512LayersTo512Layers(x))
        print(f'Size after pass 8: {x.size()}')
        skip_layer_4 = x
        x = self.maxpool2x2(x)
        print(f'Size after Max Pool 4: {x.size()}')
        x = F.relu(self.conv3x3_512LayersTo1024Layers(x))
        print(f'Size after pass 9: {x.size()}')
        x = F.relu(self.conv3x3_1024LayersTo1024Layers(x))
        print(f'Size after pass 10: {x.size()}')
        x = self.conv2x2_1024To512Layers(x)
        print(f'Size after up conv 1: {x.size()}')
