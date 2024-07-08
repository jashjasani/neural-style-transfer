import os
import cv2 as cv
import numpy as np
from torchvision import transforms
from models.vgg16 import Vgg16
import matplotlib.pyplot as plt

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]




def get_image(path:str, device, target_shape=None):
    if not os.path.exists(path):
        raise Exception(f"Path does not exist {path}")
    
    img = cv.imread(path)[:,:, ::-1]

    
    if target_shape is not None:  # resize section

        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the height
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    
    
    trf = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = trf(img).to(device).unsqueeze(0)

    return img



def get_model():
    model = Vgg16()

    layer_names = model.layer_names
    content_fms_index = model.content_feature_maps_index
    style_fms_indices = model.style_feature_maps_indices


    content_fms_index_name = (content_fms_index, layer_names[content_fms_index])
    style_fms_indices_names = (style_fms_indices, layer_names)
    model.eval()


    return model, content_fms_index_name, style_fms_indices_names


def visualize_filter(content_representation, filter_index):
    # Extract the specified filter
    filter_map = content_representation[filter_index].squeeze(0).numpy()
    
    # Normalize the filter map to 0-1 range
    filter_map = (filter_map - filter_map.min()) / (filter_map.max() - filter_map.min())
    
    # Display the filter map in grayscale
    plt.figure(figsize=(10, 8))
    plt.imshow(filter_map, cmap='gray')
    plt.title(f"Filter {filter_index}")
    plt.axis('off')
    plt.show()



def gram_matrix():
    pass