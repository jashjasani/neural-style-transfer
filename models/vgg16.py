import torch
import gc
from torchvision.models import vgg16
from collections import namedtuple

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, show_progress=False) -> None:
        super().__init__()
        vgg_pretrained_features = vgg16(weights='VGG16_Weights.DEFAULT', progress=show_progress).features
        self.layer_names = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        self.content_feature_maps_index = 1
        self.style_feature_maps_indices = list(range(len(self.layer_names)))
        
        self.slice1 = torch.nn.Sequential(*list(vgg_pretrained_features.children())[:4])
        self.slice2 = torch.nn.Sequential(*list(vgg_pretrained_features.children())[4:9])
        self.slice3 = torch.nn.Sequential(*list(vgg_pretrained_features.children())[9:16])
        self.slice4 = torch.nn.Sequential(*list(vgg_pretrained_features.children())[16:23])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        relu1_2 = self.slice1(x)
        relu2_2 = self.slice2(relu1_2)
        relu3_3 = self.slice3(relu2_2)
        relu4_3 = self.slice4(relu3_3)
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out