import timm
import torch


def build_timm_model(model_name: str,
                     num_classes: int,
                     pretrained: bool = False) -> torch.nn.Module:
    timm_model_name_map = {
        'resnet': 'timm/resnet50.a1_in1k',
        'resnext': 'timm/resnext50_32x4d.a1h_in1k',
        'mobilenet': 'timm/mobilenetv3_small_100.lamb_in1k',
        'convnext': 'timm/convnext_tiny.in12k_ft_in1k',
        'vit': 'timm/vit_tiny_patch16_224.augreg_in21k',
        'swin': 'timm/swin_tiny_patch4_window7_224.ms_in1k',
        'mobilevit': 'timm/mobilevitv2_050.cvnets_in1k',
        'maxvit': 'timm/maxvit_tiny_tf_224.in1k',
    }
    last_layer_name_map = {
        'resnet': 'fc',
        'resnext': 'fc',
        'mobilenet': 'classifier',
        'convnext': 'head.fc',
        'vit': 'head',
        'swin': 'head.fc',
        'mobilevit': 'head.fc',
        'maxvit': 'head.fc',
    }

    timm_model_name = timm_model_name_map[model_name]
    last_layer_name = last_layer_name_map[model_name]

    try:
        timm_model = timm.create_model(timm_model_name, num_classes=num_classes, pretrained=pretrained)
    except:
        raise ValueError('`build_timm_model`: model_name (%s) not supported.' %
                         model_name)

    model = SimCLRModel(timm_model=timm_model,
                        last_layer_name=last_layer_name,
                        num_classes=num_classes)

    return model


class SimCLRModel(torch.nn.Module):

    def __init__(self,
                 timm_model: torch.nn.Module,
                 last_layer_name: str,
                 num_classes: int = 10,
                 hidden_dim: int = 512,
                 z_dim: int = 128) -> None:
        super(SimCLRModel, self).__init__()
        self.num_classes = num_classes

        # Isolate the model into an encoder and a linear classifier.
        self.encoder = timm_model

        # Get the correct dimensions of the last linear layer and remove the linear layer.
        # NOTE: Currently not supporting many options...
        name_list = last_layer_name.split('.')
        assert any(n == name_list[0] for (n, _) in self.encoder.named_children())
        assert len(name_list) in [1, 2]
        if len(name_list) == 2:
            assert last_layer_name == 'head.fc'
            last_layer = self.encoder.head.fc
        else:
            last_layer = getattr(self.encoder, last_layer_name)

        self.linear_in_features = last_layer.in_features
        self.linear_out_features = last_layer.out_features

        if len(name_list) == 2:
            assert last_layer_name == 'head.fc'
            self.encoder.head.fc = torch.nn.Identity()
        else:
            setattr(self.encoder, last_layer_name, torch.nn.Identity())

        # This is the linear classifier for fine-tuning and inference.
        self.linear = torch.nn.Linear(in_features=self.linear_in_features,
                                      out_features=self.linear_out_features)

        # This is the projection head g(.) for SimCLR training.
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.linear_in_features,
                            out_features=hidden_dim,
                            bias=False),
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
