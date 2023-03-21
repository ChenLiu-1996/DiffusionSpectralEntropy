"""
Modification to ResNet encoder is adapted from
https://github.com/leftthomas/SimCLR/blob/master/model.py
"""

import torch
import torchvision
from typing import List


class ResNet50(torch.nn.Module):

    def __init__(self,
                 num_classes: int = 10,
                 small_image: bool = False,
                 hidden_dim: int = 512,
                 z_dim: int = 128) -> None:
        super(ResNet50, self).__init__()
        self.num_classes = num_classes

        # Isolate the ResNet model into an encoder and a linear classifier.

        # Get the correct dimensions of the classifer.
        self.encoder = torchvision.models.resnet50(
            num_classes=self.num_classes, weights=None)
        self.linear_in_features = self.encoder.fc.in_features
        self.linear_out_features = self.encoder.fc.out_features
        self.encoder.fc = torch.nn.Identity()

        if small_image:
            # Modify the encoder for small images (MNIST, CIFAR, etc.).
            del self.encoder
            self.encoder = []
            for name, module in torchvision.models.resnet50(
                    num_classes=self.num_classes,
                    weights=None).named_children():
                if name == 'conv1':
                    module = torch.nn.Conv2d(3,
                                             64,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bias=False)
                if not isinstance(module, torch.nn.Linear) and not isinstance(
                        module, torch.nn.MaxPool2d):
                    self.encoder.append(module)
            self.encoder.append(torch.nn.Flatten())
            self.encoder = torch.nn.Sequential(*self.encoder)

        # This is the linear classifier for fine-tuning and inference.
        self.linear = torch.nn.Linear(in_features=self.linear_in_features,
                                      out_features=self.linear_out_features)

        # This is the projection head g(.) for SimCLR training.
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.linear_in_features,
                            out_features=hidden_dim,
                            bias=False), torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=hidden_dim,
                            out_features=z_dim,
                            bias=True))

    def encode(self, x):
        return self.encoder(x)

    def project(self, x):
        return self.projection_head(self.encoder(x))

    def forward(self, x):
        return self.linear(self.encoder(x))

    def init_linear(self):
        torch.nn.init.constant_(self.linear.weight, 0.01)
        torch.nn.init.constant_(self.linear.bias, 0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(
                    m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                    torch.nn.init.constant_(m.bias, 0)


class AutoEncoder(torch.nn.Module):

    def __init__(self,
                num_classes: int = 10,
                small_image: bool = False,
                channels: List[int] = [1,32,64],
                code_dim: int = 20, # NOTE: 1024 instead of 2048, cuz 28x28 < 900.
                imsize: int = 28,
                hidden_dim: int = 10,
                z_dim: int = 128) -> None:
        super(AutoEncoder, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.code_dim = code_dim
        spatial_dim = imsize // (2**(len(self.channels)-1))
        # Encoder
        in_channels = self.channels[0]
        encoder_modules = []
        for i in range(1, len(self.channels)):
            h_dim = self.channels[i]
            encoder_modules.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    torch.nn.BatchNorm2d(h_dim),
                    torch.nn.ReLU())
            )
            in_channels = h_dim
        print(spatial_dim, h_dim, spatial_dim*spatial_dim*h_dim)

        encoder_modules.append(
            torch.nn.Sequential(
                torch.nn.Flatten(1),
                torch.nn.Linear(spatial_dim*spatial_dim*h_dim, code_dim)
            ))
        self.encoder = torch.nn.Sequential(*encoder_modules)

        # Decoder
        decoder_modules = []
        decoder_modules.append(
            torch.nn.Sequential(
                    torch.nn.Linear(code_dim, spatial_dim*spatial_dim*h_dim),
                    torch.nn.Unflatten(1, (h_dim,spatial_dim,spatial_dim)),
            )
        )

        for i in range(len(self.channels)-1, 0,-1):
            #print('?', self.hidden_channels)
            h_dim = self.channels[i]
            next_h_dim = self.channels[i-1]
            #print('h_dim: ', h_dim, 'next_h_dim:', next_h_dim)
            decoder_modules.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=h_dim, out_channels=next_h_dim,
                              kernel_size=3, stride=2, padding=1, output_padding=1),
                    torch.nn.BatchNorm2d(next_h_dim),
                    torch.nn.ReLU())
            )
            in_channels = h_dim
        decoder_modules.append(torch.nn.Sequential(
            torch.nn.Conv2d(self.channels[0], out_channels=self.channels[0],
                      kernel_size=3, padding=1),
            torch.nn.Sigmoid()))
        self.decoder = torch.nn.Sequential(*decoder_modules)
        
        #This is the linear classifier for fine-tuning and inference.
        self.linear = torch.nn.Linear(in_features=self.code_dim,
                                      out_features=self.num_classes)
    
    def encode(self, x):
        code = self.encoder(x)
        #print(code.shape)
        return code
    

    def decode(self, code):
        return self.decoder(code)

    def forward(self, x):
        code = self.encoder(x)
        code = code.view(code.shape[0], -1) # Flatten
        return self.linear(code)

    def init_linear(self):
        torch.nn.init.constant_(self.linear.weight, 0.01)
        torch.nn.init.constant_(self.linear.bias, 0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(
                    m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                    torch.nn.init.constant_(m.bias, 0)


class CAutoEncoder(torch.nn.Module):

    def __init__(self,
                num_classes: int = 10,
                small_image: bool = False,
                channels: List[int] = [3,32,64],
                code_dim: int = 392, # NOTE: 1024 instead of 2048, cuz 28x28 < 900.
                imsize: int = 28,
                hidden_dim: int = 10,
                z_dim: int = 128) -> None:
        super(CAutoEncoder, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.code_dim = code_dim
        spatial_dim = imsize // (2**(len(self.channels)-1))
        
        self.encoder = torch.nn.Sequential(
          torch.nn.Conv2d(1, 4, 3, 1, 1),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(2,2),
          torch.nn.Conv2d(4, 8, 3, 1, 1),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(2,2),
          torch.nn.Flatten(),
          torch.nn.Linear(392, 20),
          torch.nn.ReLU(),
          torch.nn.Linear(20, 2)
        )

        self.decoder = torch.nn.Sequential(
          torch.nn.Linear(2,20),
          torch.nn.ReLU(),
          torch.nn.Linear(20, 392),
          torch.nn.ReLU(),
          torch.nn.Unflatten(1, (8, 7,7)),
          torch.nn.Upsample(scale_factor=2),
          torch.nn.Conv2d(8, 4, 3, 1, 1),
          torch.nn.ReLU(),
          torch.nn.Upsample(scale_factor=2),
          torch.nn.Conv2d(4, 1, 3, 1, 1),
          torch.nn.Sigmoid()
        )
        
        #This is the linear classifier for fine-tuning and inference.
        self.linear = torch.nn.Linear(in_features=self.code_dim,
                                      out_features=self.num_classes)
    
    def encode(self, x):
        code = self.encoder(x)
        #print(code.shape)
        return code
    

    def decode(self, code):
        return self.decoder(code)

    def forward(self, x):
        code = self.encoder(x)
        code = code.view(code.shape[0], -1) # Flatten
        return self.linear(code)

    def init_linear(self):
        torch.nn.init.constant_(self.linear.weight, 0.01)
        torch.nn.init.constant_(self.linear.bias, 0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(
                    m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                    torch.nn.init.constant_(m.bias, 0)