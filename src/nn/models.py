"""
Modification to ResNet encoder is adapted from
https://github.com/leftthomas/SimCLR/blob/master/model.py
"""

import torch
import torchvision


def get_model(model_name: str, num_classes: int,
              small_image: bool) -> torch.nn.Module:
    if model_name == 'resnet50':
        model = ResNet50(num_classes=num_classes, small_image=small_image)
    elif model_name == 'wideresnet50':
        model = WideResNet50_2(num_classes=num_classes,
                               small_image=small_image)
    elif model_name == 'resnext50':
        model = ResNeXT50_32x4d(num_classes=num_classes,
                                small_image=small_image)
    else:
        raise ValueError('`get_model`: model_name (%s) not supported.' %
                         model_name)
    return model


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


class WideResNet50_2(torch.nn.Module):

    def __init__(self,
                 num_classes: int = 10,
                 small_image: bool = False,
                 hidden_dim: int = 512,
                 z_dim: int = 128) -> None:
        super(WideResNet50_2, self).__init__()
        self.num_classes = num_classes

        # Isolate the WideResNet50_2 model into an encoder and a linear classifier.

        # Get the correct dimensions of the classifer.
        self.encoder = torchvision.models.wide_resnet50_2(
            num_classes=self.num_classes, weights=None)
        self.linear_in_features = self.encoder.fc.in_features
        self.linear_out_features = self.encoder.fc.out_features
        self.encoder.fc = torch.nn.Identity()

        if small_image:
            # Modify the encoder for small images (MNIST, CIFAR, etc.).
            del self.encoder
            self.encoder = []
            for name, module in torchvision.models.wide_resnet50_2(
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


class ResNeXT50_32x4d(torch.nn.Module):

    def __init__(self,
                 num_classes: int = 10,
                 small_image: bool = False,
                 hidden_dim: int = 512,
                 z_dim: int = 128) -> None:
        super(ResNeXT50_32x4d, self).__init__()
        self.num_classes = num_classes

        # Isolate the ResNeXT50_32x4d model into an encoder and a linear classifier.

        # Get the correct dimensions of the classifer.
        self.encoder = torchvision.models.resnext50_32x4d(
            num_classes=self.num_classes, weights=None)
        self.linear_in_features = self.encoder.fc.in_features
        self.linear_out_features = self.encoder.fc.out_features
        self.encoder.fc = torch.nn.Identity()

        if small_image:
            # Modify the encoder for small images (MNIST, CIFAR, etc.).
            del self.encoder
            self.encoder = []
            for name, module in torchvision.models.resnext50_32x4d(
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
