import torch
import torchvision


class ResNet34(torch.nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super(ResNet34, self).__init__()
        self.num_classes = num_classes

        # Isolate the classification model
        # into an encoder and a linear classifier.
        self.encoder = torchvision.models.resnet34(
            num_classes=self.num_classes)
        self.linear_in_features = self.encoder.fc.in_features
        self.linear_out_features = self.encoder.fc.out_features
        self.encoder.fc = torch.nn.Identity()

        # This is the linear layer for fine-tuning and inference.
        self.linear = torch.nn.Linear(in_features=self.linear_in_features,
                                      out_features=self.linear_out_features)

        # This is the projection head g(.) for SimCLR training.
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.linear_in_features,
                            out_features=128), torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=128))

    def encode(self, x):
        return self.encoder(x)

    def project(self, x):
        return self.projection_head(self.encoder(x))

    def forward(self, x):
        return self.linear(self.encoder(x))

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def init_linear(self):
        self.linear.weight.data.fill_(0.01)
        self.linear.bias.data.fill_(0.01)
