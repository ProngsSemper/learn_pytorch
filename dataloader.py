import ssl

import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ssl._create_default_https_context = ssl._create_unverified_context
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor)
# 装载数据                 设置数据集           每次从数据集中取几个  是否打乱                   不足时是否舍弃（如：要取64个但是只剩16个数据）
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

writer = SummaryWriter("logs")
# 进行两轮取样shuffle=True时 两轮取样不同
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch:{}".format(epoch), imgs.step)
        step = step + 1
writer.close()
