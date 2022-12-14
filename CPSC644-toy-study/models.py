import torch
import torchvision


class FlexibleResNet50(torch.nn.Module):

    def __init__(self,
                 contrastive: bool = False,
                 num_classes: int = 10) -> None:
        super(FlexibleResNet50, self).__init__()
        self.contrastive = contrastive
        self.num_classes = num_classes
        if not self.contrastive:
            self.model = torchvision.models.resnet50(
                num_classes=self.num_classes)
        else:
            self.encoder = torchvision.models.resnet50(
                num_classes=self.num_classes)
            in_features = self.encoder.fc.in_features
            out_features = self.encoder.fc.out_features
            del self.encoder.fc
            self.linear = torch.nn.Linear(in_features=in_features,
                                          out_features=out_features)

    def encode(self, x):
        if not self.contrastive:
            raise ValueError(
                '`FlexibleResNet50.encode` only works for `contrastive=True`.')
        else:
            return self.encoder(x)

    def forward(self, x):
        if not self.contrastive:
            return self.model(x)
        else:
            return self.linear(self.encoder(x))
