from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import BasicBlock, conv1x1

class DoubleConv(nn.Module):
    def __init__(self, activation_function: nn.Module,
                 input_channel: int, intermediate_channel: int, output_channel: int,
                 kernel_size=3, stride=1, padding=0):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(input_channel,intermediate_channel,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn1 = nn.BatchNorm2d(intermediate_channel)
        self.act1 = activation_function()
        self.conv2 = nn.Conv2d(intermediate_channel,output_channel,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.act2 = activation_function()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, activation_function: nn.Module, 
                 input_channel: int, intermediate_channel: int, output_channel: int, 
                 kernel_size=3, stride=1, padding=0):
        super(DecoderBlock, self).__init__()

        self.conv_block = DoubleConv(activation_function, 
                                     input_channel, intermediate_channel, output_channel,
                                     kernel_size, stride, padding)
    
    def forward(self, x, skip):
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]

        x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class ResNet34(nn.Module):
    def __init__(self, activation_function: nn.Module):
        super(ResNet34, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = activation_function()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 3)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2, dilate=False)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2, dilate=False)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2, dilate=False)
    
    def _make_layer(
        self,
        block: BasicBlock,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        feat1 = self.maxpool(x)

        feat2 = self.layer1(feat1)
        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        
        return [feat1, feat2, feat3, feat4, feat5]

class SVNetEncoder(nn.Module):
    def __init__(self, activation_function: nn.Module):
        super(SVNetEncoder, self).__init__()
        self.encoder_conv_block = DoubleConv(activation_function, 3, 64, 64, 3, 1, 0)
        self.resnet34_encoder = ResNet34(activation_function)
    
    def forward(self, x):
        encoder_feat0 = self.encoder_conv_block(x)
        encoder_feat1, encoder_feat2, encoder_feat3, encoder_feat4, encoder_feat5 = self.resnet34_encoder(x)
        return encoder_feat0, encoder_feat1, encoder_feat2, encoder_feat3, encoder_feat4, encoder_feat5

class SVNetDecoder(nn.Module):
    def __init__(self, activation_function: nn.Module, final_output_channel=3):
        super(SVNetDecoder, self).__init__()
        self.decoder_conv_block5 = DoubleConv(activation_function, 512, 512, 256, 3, 1, 0)
        self.decoder_conv_block4 = DecoderBlock(activation_function, 512, 512, 128, 3, 1, 0)
        self.decoder_conv_block3 = DecoderBlock(activation_function, 256, 256, 64, 3, 1, 0)
        self.decoder_conv_block2 = DecoderBlock(activation_function, 128, 128, 64, 3, 1, 0)
        self.decoder_conv_block1 = DecoderBlock(activation_function, 128, 128, 64, 3, 1, 0)
        self.decoder_conv_block0 = DecoderBlock(activation_function, 128, 64, 64, 3, 1, 0)
        self.conv_out = nn.Conv2d(64, final_output_channel, 3, 1, 0)
    
    def forward(self, encoder_feats:List[torch.Tensor]):
        encoder_feat0, encoder_feat1, encoder_feat2, encoder_feat3, encoder_feat4, encoder_feat5 = encoder_feats
        
        decoder_out5 = self.decoder_conv_block5(encoder_feat5)
        decoder_out4 = self.decoder_conv_block4(decoder_out5, encoder_feat4)
        decoder_out3 = self.decoder_conv_block3(decoder_out4, encoder_feat3)
        decoder_out2 = self.decoder_conv_block2(decoder_out3, encoder_feat2)
        decoder_out1 = self.decoder_conv_block1(decoder_out2, encoder_feat1)        
        conv_block_out = self.decoder_conv_block0(decoder_out1, encoder_feat0)
        out = self.conv_out(conv_block_out)
        
        out = F.sigmoid(out)
        return out


class SVNet(nn.Module):
    def __init__(self, activation='relu'):
        super(SVNet, self).__init__()
        activation_function = nn.LeakyReLU if activation == 'relu' else nn.GELU

        self.svnet_encoder = SVNetEncoder(activation_function)
        self.base_color_decoder = SVNetDecoder(activation_function, final_output_channel=3)
        self.normal_decoder = SVNetDecoder(activation_function, final_output_channel=3)
        self.metallic_decoder = SVNetDecoder(activation_function, final_output_channel=1)
        self.roughness_decoder = SVNetDecoder(activation_function, final_output_channel=1)

    def forward(self, x):
        encoder_feats = self.svnet_encoder(x)

        base_color = self.base_color_decoder(encoder_feats)
        normal = self.normal_decoder(encoder_feats)
        metallic = self.metallic_decoder(encoder_feats)
        roughness = self.roughness_decoder(encoder_feats)

        return base_color, normal, metallic, roughness

if __name__ == '__main__':
    svnet = SVNet().cuda()

    dummy_input = torch.randn(1, 3, 512, 512).cuda()

    base_color, normal, metallic, roughness = svnet(dummy_input)
    print(base_color.size(), normal.size(), metallic.size(), roughness.size())