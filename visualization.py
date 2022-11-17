import monai
import numpy as np
import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image
from easydict import EasyDict
from matplotlib import pyplot as plt
from monai.networks.nets import SegResNet
from monai.transforms import SaveImage

image_path = './images/ims23.png'

# 加载模型
config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
print('load model')
model = SegResNet(in_channels=config.model.in_channels, out_channels=config.model.out_channels, norm="", spatial_dims=2)
model.load_state_dict(torch.load('model/best/pytorch_model.bin', map_location='cpu'))

# 加载图片
image = Image.open(image_path).convert('L')
image = F.to_tensor(image)
val_transform = monai.transforms.Compose([
    monai.transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
])

image = val_transform(image)
image = torch.unsqueeze(image, dim=0)

# 模型预测
post_trans = monai.transforms.Compose([
    monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
])
saver = SaveImage(output_dir="./output", output_ext=".png", output_postfix="seg")
with torch.no_grad():
    logits = model(image)
    val_outputs = [post_trans(i).detach().cpu() for i in logits]

# 可视化图片
plt.figure("ori")
plt.imshow(image[0][0], cmap='gray')
val_outputs = val_outputs[0]
plt.figure("output")
for i in range(6):
    plt.subplot(1, 6, i + 1)
    plt.title(f"output {i}")
    plt.axis(False)
    plt.imshow(val_outputs[i, :, :], cmap='gray')
plt.show()

rgb = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0]]
change = [[0 for col in range(128)] for row in range(128)]
muscle = ['TA', 'PL', 'PT', 'MG', 'SL', 'LG']
tn = np.zeros((128, 128, 3))
for k in range(6):
    t = val_outputs[k, :, :]
    t0 = t.reshape(t.shape[0], t.shape[1], 1)
    a = np.concatenate([t0 * rgb[k][0], t0 * rgb[k][1], t0 * rgb[k][2]], axis=2)
    tn = np.add(tn, np.concatenate([t0 * rgb[k][0], t0 * rgb[k][1], t0 * rgb[k][2]], axis=2))
plt.figure("change")
plt.imshow(image[0][0], cmap='gray', alpha=0.6)
plt.imshow(tn, alpha=0.4)
La = 0
for color in list('rgcymb'):
    plt.scatter([], [], c=color, label=muscle[La])
    La += 1
plt.legend(facecolor='honeydew', fontsize=7)
plt.show()
