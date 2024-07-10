from utils import get_image, get_model, gram_matrix, total_variation, save_image
import numpy as np 
from torch.autograd import Variable
import torch
import tqdm

C_W = 0.5
S_W = 0.5
T_W = 0.1

CONTENT_IMAGE_PATH = "/home/jcube/Downloads/intricate-explorer-OtfVkOWUbGQ-unsplash.jpg"
STYLE_IMAGE_PATH = "/home/jcube/Downloads/intricate-explorer-OtfVkOWUbGQ-unsplash.jpg"


device = "cpu"


style_image = get_image(STYLE_IMAGE_PATH, device, 512)
content_image = get_image(CONTENT_IMAGE_PATH, device, 512)


init_image = content_image

# init_image = np.random.normal(size=content_image.shape).astype(np.float32)
# init_image = torch.from_numpy(init_image).float().to(device)


image_to_optimize = Variable(init_image, requires_grad=True)


model, content_fms_index, style_fms_indices = get_model()

optimizer = torch.optim.AdamW((image_to_optimize,), lr=1e1)

pbar = tqdm.tqdm(range(1000))

for i in pbar:
    content_fms = model(content_image)
    style_fms = model(style_image)
    target_content_representation = content_fms[content_fms_index[0]].squeeze(axis=0)
    target_style_representation = [gram_matrix(x) for cnt,x in enumerate(style_fms) if cnt in style_fms_indices[0]]
    target_representations = [target_content_representation, target_style_representation]

    current_fms = model(image_to_optimize)
    current_content_representation = current_fms[content_fms_index[0]].squeeze(axis=0)
    current_style_representation = [gram_matrix(x) for cnt,x in enumerate(current_fms) if cnt in style_fms_indices[0]]


    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    for x,y in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction="sum")(x[0], y[0])

    style_loss /= len(target_style_representation)
    
    tv_loss = total_variation(image_to_optimize)


    total_loss = content_loss * C_W + style_loss * S_W + tv_loss * T_W

    total_loss.backward()


    optimizer.step()
    optimizer.zero_grad()

    
    with torch.no_grad():
        save_image(image_to_optimize, "./iterations", "image", i)
        pbar.set_postfix({'total_loss': total_loss.item()})


