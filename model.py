import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, VGG16_BN_Weights

__all__ = ["PixelLink2s"]

N_NEIGHBORS = 8


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)


def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv3x3_bn_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(
        conv3x3(in_channels, out_channels, stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = VGG16_BN_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.feat_extractor = vgg16_bn(weights=weights).features

        self.conv_stage1 = self.feat_extractor[0: 6] # 'conv1_1', 'conv1_2'
        self.conv_stage2 = self.feat_extractor[6: 13] # 'pool1', 'conv2_1', 'conv2_2'
        self.conv_stage3 = self.feat_extractor[13: 23] # 'pool2', 'conv3_1', 'conv3_2', 'conv3_3'
        self.conv_stage4 = self.feat_extractor[23: 33] # 'pool3', 'conv4_1', 'conv4_2', 'conv4_3'
        self.conv_stage5 = self.feat_extractor[33: 43] # 'pool4', 'conv5_1', 'conv5_2', 'conv5_3'

        # "All pooling layers except pool5 take a stride of $2$, and pool5 takes $1$."
        # Figure 3: Structure of PixelLink+VGG16 2s. fc6 and fc7 are converted into convolutional layers.
        self.block = nn.Sequential(
            conv3x3_bn_relu(512, 512),
            conv3x3_bn_relu(512, 512),
        ) # 'fc6', 'fc7'

    def forward(self, x): # `(b, 3, h, w)`
        x = self.conv_stage1(x)

        x1 = self.conv_stage2(x) # `(b, 3, h // 2, w // 2)`
        x2 = self.conv_stage3(x1) # `(b, 3, h // 4, w / 4)`
        x3 = self.conv_stage4(x2) # `(b, 3, h // 8, w // 8)`
        x4 = self.conv_stage5(x3) # `(b, 3, h // 16, w // 16)`
        x5 = self.block(x4) # `(b, 3, h // 16, w // 16)`
        return x1, x2, x3, x4, x5


# "Two settings of feature fusion layers are implemented:
# {'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3', 'fc_7'}, and {conv3_3, conv4_3, conv5_3, fc_7},
# denoted as 'PixelLink + VGG16 2s', and 'PixelLink + VGG16 4s', respectively. The resolution
# of '2s' predictions is a half of the original image, and '4s' is a quarter."
class PixelLink2s(nn.Module):
    def __init__(self, pretrained_vgg16=True):
        super().__init__()

        self.backbone = VGG16(pretrained_vgg16)

        # 'conv 1x1, 2(16)' stands for a 1x1 convolutional layer with 2 or 16 kernels,
        # for text/non-text prediction or link prediction individually."
        self.pixel_conv1 = conv1x1(128, 2)
        self.pixel_conv2 = conv1x1(256, 2)
        self.pixel_conv3 = conv1x1(512, 2)
        self.pixel_conv4 = conv1x1(512, 2)
        self.pixel_conv5 = conv1x1(512, 2)

        self.link_conv1 = conv1x1(128, 16)
        self.link_conv2 = conv1x1(256, 16)
        self.link_conv3 = conv1x1(512, 16)
        self.link_conv4 = conv1x1(512, 16)
        self.link_conv5 = conv1x1(512, 16)

    # "The upsampling operation is done through bilinear interpolation directly."
    def _upsample(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x
        
    def forward(self, x): # `(b, 3, h, w)`
        x1, x2, x3, x4, x5 = self.backbone(x)

        # "The size of 'fc7' is the same as 'conv5_3', and no upsampling is needed when adding scores
        # from these two layers.
        pixel = self.pixel_conv5(x5) + self.pixel_conv4(x4) # `(b, 2, h // 16, w // 16)`
        pixel = self._upsample(pixel) # `(b, 2, h // 8, w // 8)`
        pixel += self.pixel_conv3(x3) # `(b, 2, h // 8, w // 8)`
        pixel = self._upsample(pixel) # `(b, 2, h // 4, w // 4)`
        pixel += self.pixel_conv2(x2) # `(b, 2, h // 4, w // 4)`
        pixel = self._upsample(pixel) # `(b, 2, h // 2, w // 2)`
        pixel += self.pixel_conv1(x1) # `(b, 2, h // 2, w // 2)`
        # "Softmax is used in both."
        pixel = F.softmax(pixel, dim=1)
        # return pixel

        link = self.link_conv5(x5) + self.link_conv4(x4)  # `(b, 2, h // 16, w // 16)`
        link = self._upsample(link) # `(b, 2, h // 8, w // 8)`
        link += self.link_conv3(x3) # `(b, 2, h // 8, w // 8)`
        link = self._upsample(link) # `(b, 2, h // 4, w // 4)`
        link += self.link_conv2(x2) # `(b, 2, h // 4, w // 4)`
        link = self._upsample(link) # `(b, 2, h // 2, w // 2)`
        link += self.link_conv1(x1) # `(b, 2, h // 2, w // 2)`
        for i in range(0, N_NEIGHBORS * 2, 2):
            link[:, i: i + 2, ...] = F.softmax(link[:, i: i + 2, ...], dim=1)
        return pixel, link


if __name__ == "__main__":
    model = PixelLink2s()
    # x = torch.randn(2, 3, 3536, 2512)
    x = torch.randn(2, 3, 2512, 512)
    pixel_pred, link_pred = model(x)
    pixel_pred.shape

    # pixel_pred.sum(dim=1)
    # for i in range(0, N_NEIGHBORS * 2, 2):
    #     link_pred[:, i: i + 2, ...].sum(dim=1)
    # pixel_pred.shape, link_pred.shape

# "Instead of fine-tuning from an ImageNet-pretrained model, the VGG net is randomly initialized
# via the xavier method."
