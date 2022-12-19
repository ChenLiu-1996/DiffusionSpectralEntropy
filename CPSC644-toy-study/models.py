import torch
import torchvision


class ResNet50(torch.nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super(ResNet50, self).__init__()
        self.num_classes = num_classes

        # Isolate the classification model
        # into an encoder and a linear classifier.
        self.encoder = torchvision.models.resnet50(
            num_classes=self.num_classes)
        self.linear_in_features = self.encoder.fc.in_features
        self.linear_out_features = self.encoder.fc.out_features
        self.encoder.fc = torch.nn.Identity()
        self.linear = torch.nn.Linear(in_features=self.linear_in_features,
                                      out_features=self.linear_out_features)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.linear(self.encoder(x))

    def freeze_encoder(self):
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for _, param in self.encoder.named_parameters():
            param.requires_grad = True

    def freeze_linear(self):
        for _, param in self.linear.named_parameters():
            param.requires_grad = False

    def unfreeze_linear(self):
        for _, param in self.linear.named_parameters():
            param.requires_grad = True

    def init_linear(self):
        self.linear.weight.data.fill_(0.01)
        self.linear.bias.data.fill_(0.01)
