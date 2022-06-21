import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
image_path = "dataset/train/bees/17209602_fe5a5a746f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("train", img_array, 1, dataformats='HWC')
writer.close()
