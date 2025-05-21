import torch
import os
import yaml
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
from model import MiniUnet
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from rectified_flow import RectifiedFlow
from functools import partial
import numpy as np


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (batch, dim)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.sum(error ** 2, dim=-1)  # ||Δ||^2 per sample
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (w.detach() * loss).mean()


def train(config: str):
    """训练flow matching模型

    Args:
        config (str): yaml配置文件路径，包含以下参数：
            base_channels (int, optional): MiniUnet的基础通道数，默认值为16。
            epochs (int, optional): 训练轮数，默认值为10。
            batch_size (int, optional): 批大小，默认值为128。
            lr_adjust_epoch (int, optional): 学习率调整轮数，默认值为50。
            batch_print_interval (int, optional): batch打印信息间隔，默认值为100。
            checkpoint_save_interval (int, optional): checkpopint保存间隔(单位为epoch)，默认值为1。
            save_path (str, optional): 模型保存路径，默认值为'./checkpoints'。
            use_cfg (bool, optional): 是否使用Classifier-free Guidance训练条件生成模型，默认值为False。
            device (str, optional): 训练设备，默认值为'cuda'。

    """
    # 读取yaml配置文件
    config = yaml.load(open(config, 'rb'), Loader=yaml.FullLoader)
    # 解析参数数据，有默认值
    base_channels = config.get('base_channels', 16)
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 128)
    lr_adjust_epoch = config.get('lr_adjust_epoch', 50)
    batch_print_interval = config.get('batch_print_interval', 100)
    checkpoint_save_interval = config.get('checkpoint_save_interval', 1)
    save_path = config.get('save_path', './checkpoints')
    use_cfg = config.get('use_cfg', False)
    meanflow = config.get('meanflow', True)
    flow_ratio = config.get('flow_ratio', 0.5)
    device = config.get('device', 'cuda')

    # 打印训练参数
    print('Training config:')
    print(f'base_channels: {base_channels}')
    print(f'epochs: {epochs}')
    print(f'batch_size: {batch_size}')
    print(f'lr_adjust_epoch: {lr_adjust_epoch}')
    print(f'batch_print_interval: {batch_print_interval}')
    print(f'checkpoint_save_interval: {checkpoint_save_interval}')
    print(f'save_path: {save_path}')
    print(f'use_cfg: {use_cfg}')
    print(f'device: {device}')

    # 训练flow matching模型

    # 数据集加载
    # 把PIL转为tensor
    transform = Compose([ToTensor()])  # 变换成tensor + 变为[0, 1]

    dataset = MNIST(
        root='./data',
        train=True,  # 6w
        download=True,
        transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型加载
    model = MiniUnet(base_channels, meanflow=meanflow)
    model.to(device)

    # 优化器加载 Rectified Flow的论文里面有的用的就是AdamW
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # 学习率调整
    scheduler = StepLR(optimizer, step_size=lr_adjust_epoch, gamma=0.1)

    # RF加载
    rf = RectifiedFlow()

    # 记录训练时候每一轮的loss
    loss_list = []

    # 一些文件夹提前创建
    os.makedirs(save_path, exist_ok=True)

    # 训练循环
    for epoch in range(epochs):
        for batch, data in enumerate(dataloader):
            x_1, y = data  # x_1原始图像，y是标签，用于CFG
            batch_size = x_1.shape[0]

            # 随机采样时间步t和r（均匀分布）
            t_np = np.random.rand(batch_size).astype(np.float32)
            r_np = np.random.rand(batch_size).astype(np.float32)
            
            # 根据flow_ratio选择部分样本设置r = t
            num_selected = int(flow_ratio * batch_size)
            indices = np.random.permutation(batch_size)[:num_selected]
            r_np[indices] = t_np[indices]  # 确保部分r等于t，简化训练
            
            # 将numpy数组转换为Tensor并调整形状
            t = torch.tensor(t_np)
            r = torch.tensor(r_np)
            # 生成flow（实际上是一个点）
            x_t, x_0 = rf.create_flow(x_1, t)
            v = x_1 - x_0

            # 4090 大概占用显存3G
            x_t = x_t.to(device)
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)
            t = t.to(device)
            r = r.to(device)
            v = v.to(device)

            optimizer.zero_grad()

            # 这里我们要做一个数据的复制和拼接，复制原始x_1，把一半的y替换成-1表示无条件生成，这里也可以直接有条件、无条件累计两次计算两次loss的梯度
            # 一定的概率，把有条件生成换为无条件的 50%的概率 [x_t, x_t] [t, t]
            #if use_cfg:
            #    x_t = torch.cat([x_t, x_t.clone()], dim=0)
            #    t = torch.cat([t, t.clone()], dim=0)
            #    y = torch.cat([y, -torch.ones_like(y)], dim=0)
            #    x_1 = torch.cat([x_1, x_1.clone()], dim=0)
            #    x_0 = torch.cat([x_0, x_0.clone()], dim=0)
            #    y = y.to(device)
            #else:
            #    y = None
            # 这里换一种方式进行CFG，先计算uncond部分
            y = torch.tensor(y, device=device, dtype=torch.float32)
            if use_cfg:
                y_uncond = -torch.ones_like(y).to(device)
                with torch.no_grad():
                    u_t = model(x=x_t, t=t, y=y_uncond, d=r) # 借用shortcut model的d输入项
                v_hat = (v + u_t) * 0.5
            else:
                v_hat = v

            # v_pred = model(x=x_t, t=t, y=y)
            model_partial = partial(model, y=y)
            u, dudt = torch.autograd.functional.jvp(
                lambda x, t, d: model_partial(x, t, d),  # 模型函数
                (x_t, t, r),                               # 输入参数
                (v_hat, torch.ones_like(t), torch.zeros_like(r)),  # 切向量（用于JVP）
                create_graph=True  # 保留计算图以支持二阶导数
            )

            # 计算目标值u_tgt（MeanFlow Identity）
            delta=t - r
            delta = delta[:, None, None, None]  # [B, 1, 1, 1]
            u_target = v - delta * dudt

            # 损失计算（自适应L2损失）
            error = u - u_target.detach()
            loss = adaptive_l2_loss(error)     # 自适应加权L2损失

            loss.backward()
            optimizer.step()

            if batch % batch_print_interval == 0:
                print(f'[Epoch {epoch}] [batch {batch}] loss: {loss.item()}')

            loss_list.append(loss.item())

        scheduler.step()

        if epoch % checkpoint_save_interval == 0 or epoch == epochs - 1 or epoch == 0:
            # 第一轮也保存一下，快速测试用，大家可以删除
            # 保存模型
            print(f'Saving model {epoch} to {save_path}...')
            save_dict = dict(model=model.state_dict(),
                             optimizer=optimizer.state_dict(),
                             epoch=epoch,
                             loss_list=loss_list)
            torch.save(save_dict,
                       os.path.join(save_path, f'miniunet_{epoch}.pth'))


if __name__ == '__main__':
    train(config='./config/train_meanflow.yaml')
