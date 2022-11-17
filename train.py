import os
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from monai.networks.nets import SegResNet
from timm.optim import optim_factory
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src import utils
from src.loader import get_dataloader
from src.utils import Logger


def train_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss], train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                    accelerator: Accelerator, step: int):
    # 训练
    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
    train_bar.set_description(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training')

    for image_batch in train_bar:
        seg_result = model(image_batch['image'])

        total_loss = 0
        for name in loss_functions:
            loss = loss_functions[name](seg_result, image_batch['label'])
            accelerator.log({name: float(loss)}, step=step)
            total_loss += loss

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.log({
            'Train Total Loss': float(total_loss),
        }, step=step)
        train_bar.set_postfix({'loss': f'{float(total_loss):1.5f}'})
        step += 1
    scheduler.step(epoch)
    return total_loss, step


def val_one_epoch(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader,
                  config: EasyDict, accelerator: Accelerator):
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])
    # 验证
    model.eval()
    val_bar = tqdm(val_loader, disable=not accelerator.is_local_main_process)
    val_bar.set_description(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation')
    dice_metric = monai.metrics.DiceMetric(include_background=True, reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True)
    for image_batch in val_bar:
        with torch.no_grad():
            logits = model(image_batch['image'])
            val_outputs = [post_trans(i) for i in logits]
            dice_metric(y_pred=val_outputs, y=image_batch['label'])

    batch_acc, batch_acc_not_nan = dice_metric.aggregate()
    if accelerator.num_processes > 1:
        batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
    mean_acc = batch_acc.mean()
    accelerator.log({
        '类别1 acc': float(batch_acc[0]),
        '类别2 acc': float(batch_acc[1]),
        '类别3 acc': float(batch_acc[2]),
        '类别4 acc': float(batch_acc[3]),
        '类别5 acc': float(batch_acc[4]),
        '类别6 acc': float(batch_acc[5]),
        '平均 acc ': float(mean_acc)
    }, step=step)
    return mean_acc, batch_acc


if __name__ == '__main__':
    # 读取配置
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(42)
    logging_dir = os.getcwd() + '/logs/' + str(datetime.now())
    accelerator = Accelerator(cpu=True, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(config)

    accelerator.print('加载数据集...')
    train_loader, val_loader = get_dataloader(config)

    # 初始化模型
    accelerator.print('加载模型...')
    model = SegResNet(in_channels=config.model.in_channels, out_channels=config.model.out_channels, norm="", spatial_dims=2)

    # 定义训练参数
    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.trainer.warmup, eta_min=1e-6)
    loss_functions = {
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True),
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
    }

    stale = 0
    best_acc = 0
    step = 0
    patience = config.trainer.num_epochs / 2

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)

    # 尝试继续训练
    starting_epoch = 0
    if config.trainer.resume:
        starting_epoch, step = utils.resume_train_state(config.trainer.checkpoint, train_loader, accelerator)
    # 开始训练
    accelerator.print("开始训练！")

    for epoch in range(starting_epoch, config.trainer.num_epochs):
        # 训练
        train_loss, step = train_one_epoch(model, loss_functions, train_loader, optimizer, scheduler, accelerator, step)
        # 验证
        mean_acc, class_acc = val_one_epoch(model, val_loader, config, accelerator)

        accelerator.print(f'mean acc: {100 * mean_acc:.5f}% , class acc: {class_acc.detach().cpu().numpy()}')
        accelerator.print(f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] lr = {scheduler.get_lr()}, loss = {float(train_loss):.5f}, acc = {100 * float(mean_acc):.5f} %")

        # 保存模型
        if mean_acc > best_acc:
            accelerator.save_state(output_dir=f"{os.getcwd()}/{config.trainer.checkpoint}/best")
            best_acc = mean_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                accelerator.print(f"连续的 {patience}  epochs 模型没有提升，停止训练")
                accelerator.end_training()
                break

    accelerator.print(f"最高acc: {best_acc:.5f}")
