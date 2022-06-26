import torchvision

# pretrained=False时，返回初始化模型参数
from torch import nn
from torch.nn import Linear

vgg16_false = torchvision.models.vgg16(pretrained=False)
# pretrained=True时，返回已经训练好的模型参数
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)
# 给现有模型增加一层线性层
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
# 修改现有模型
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
