import torch
# 对应保存方式1
import torchvision

model = torch.load("vgg16_method1.pth")
# 对应保存方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
