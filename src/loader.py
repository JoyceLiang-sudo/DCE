from typing import Tuple

import monai
import torch
import torchvision.transforms.functional as F
import tqdm
from PIL import Image
from easydict import EasyDict
from pycocotools.coco import COCO


def load_dataset(annotation_file: str, image_path: str):
    dataset = []
    coco = COCO(annotation_file)
    imgIds = coco.getImgIds()  # 图像ID列表
    for idx, imgId in tqdm.tqdm(enumerate(imgIds)):
        # 加载图片
        img_path = image_path + '/' + coco.loadImgs([imgId])[0]['file_name']
        image = Image.open(img_path).convert('L')
        annIds = coco.getAnnIds(imgIds=imgId)
        anns = coco.loadAnns(annIds)  # 获取所有注释信息
        masks = []  # 获得mask
        for ann_idx, ann in enumerate(anns):
            masks.append(coco.annToMask(ann))

        dataset.append({
            'image': F.to_tensor(image),
            'label': torch.Tensor(masks),
        })
    return dataset


def get_train_val_transforms(config: EasyDict) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose([
        # monai.transforms.EnsureChannelFirstd(keys="image"),
        # monai.transforms.EnsureTyped(keys=["image", "label"]),
        # monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        # monai.transforms.CenterSpatialCropD(keys=["image", "label"], roi_size=ensure_tuple_rep(config.model.image_size, 3)),
        # monai.transforms.Resized(keys=["image", "label"], spatial_size=ensure_tuple_rep(config.model.image_size, 3)),
        # # monai.transforms.RandSpatialCropd(keys=["image", "label"], roi_size=config.model.image_size, random_size=False),
        # monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        # monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        # monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  # 归一化，只对非零部分进行归一化
    ])
    val_transform = monai.transforms.Compose([
        # monai.transforms.EnsureChannelFirstd(keys="image"),
        # monai.transforms.EnsureTyped(keys=["image", "label"]),
        # monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        # monai.transforms.Resized(keys=["image", "label"], spatial_size=ensure_tuple_rep(config.model.image_size, 3)),
        monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    return train_transform, val_transform


def get_dataloader(config: EasyDict) -> torch.utils.data.DataLoader:
    images = load_dataset(config.trainer.annotations, config.trainer.images)
    train_transform, val_transform = get_train_val_transforms(config)

    train_dataset = monai.data.Dataset(data=images[:int(len(images) * config.trainer.train_ratio)], transform=train_transform)
    # train_dataset = monai.data.Dataset(data=[images[0]], transform=train_transform)
    val_dataset = monai.data.Dataset(data=images[int(len(images) * config.trainer.train_ratio):], transform=val_transform)
    # val_dataset = monai.data.Dataset(data=[images[1]], transform=val_transform)

    train_loader = monai.data.DataLoader(train_dataset, pin_memory=True, num_workers=config.trainer.num_workers, batch_size=config.trainer.batch_size, shuffle=True)
    val_loader = monai.data.DataLoader(val_dataset, num_workers=config.trainer.num_workers, batch_size=config.trainer.batch_size, shuffle=False)
    return train_loader, val_loader
