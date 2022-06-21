from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
# To Tensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# Normalize
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)

# resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = tensor_trans(img_resize)
writer.add_image("resize", img_resize, 0)

# 使用compose来resize
trans_resize_2 = transforms.Resize(512)
# compose让两个方法合在一起 resize和tensor化 ↓
trans_com = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_com(img)
writer.add_image("resize", img_resize_2, 1)

# 随机裁剪
trans_random = transforms.RandomCrop(500, 1000)
trans_com_2 = transforms.Compose([trans_random, tensor_trans])
for i in range(10):
    img_crop = trans_com_2(img)
    writer.add_image("randomCrop", img_crop, i)

writer.add_image("norm_img", img_norm)
writer.add_image("Tensor_img", tensor_img)
writer.close()
