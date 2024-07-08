from utils import get_image, get_model, visualize_filter
import numpy as np 



CONTENT_IMAGE_PATH = "/home/jcube/Downloads/intricate-explorer-OtfVkOWUbGQ-unsplash.jpg"
STYLE_IMAGE_PATH = "/home/jcube/Downloads/intricate-explorer-OtfVkOWUbGQ-unsplash.jpg"


device = "cpu"


style_image = get_image(STYLE_IMAGE_PATH, device, 512)
content_image = get_image(CONTENT_IMAGE_PATH, device, 512)


init_image = np.random.normal(size=content_image.shape).astype(np.float32)



model, content_fms_index, style_fms_indices = get_model()



content_representation = model(content_image)

visualize_filter(content_representation[0].squeeze(0), 2)