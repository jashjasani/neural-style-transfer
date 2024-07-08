from torchvision.models import vgg16
import torch
from collections import namedtuple


class Vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, show_progress=False) -> None:
        super().__init__()
        vgg_pretrained_features = vgg16(weights='VGG16_Weights.DEFAULT', progress=show_progress).features
        self.layer_names = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        self.content_feature_maps_index = 1
        self.style_feature_maps_indices = list(range(len(self.layer_names)))


        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()


        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out
    