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
    device = config.get('device', 'cuda')
    # cfm config
    consistency_only = config.get('consistency_only', 'false')
    consistency_delta_t = config.get('consistency_delta_t', '0.1')
    consistency_lambda_f = config.get('consistency_lambda_f', '1')
    consistency_lambda_v = config.get('consistency_lambda_v', '0.01')

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
    print(f'consistency_only: {consistency_only}')
    print(f'consistency_delta_t: {consistency_delta_t}')
    print(f'consistency_lambda_f: {consistency_lambda_f}')
    print(f'consistency_lambda_v: {consistency_lambda_v}')

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
    model = MiniUnet(base_channels)
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
            # 均匀采样[0, 1]的时间t randn 标准正态分布
            t = torch.rand(x_1.size(0))
            # 生成consistency需要的两个时间t_a和t_b
            dt = consistency_delta_t * torch.randn(x_1.size(0)).abs()
            t_a = torch.clip(t - 0.5 * dt, 0, 1) # 确保在0-1之间
            t_b = torch.clip(t + 0.5 * dt, 0, 1)

            # 生成flow（实际上是一个点）
            x_t_a, x_0 = rf.create_flow(x_1, t_a)
            x_t_b, _ = rf.create_flow(x_1, t_b, x_0) # 从同一个x_0出发

            # 4090 大概占用显存3G
            x_t_a = x_t_a.to(device)
            x_t_b = x_t_b.to(device)
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)
            t_a = t_a.to(device)
            t_b = t_b.to(device)

            optimizer.zero_grad()

            # 这里我们要做一个数据的复制和拼接，复制原始x_1，把一半的y替换成-1表示无条件生成，这里也可以直接有条件、无条件累计两次计算两次loss的梯度
            # 一定的概率，把有条件生成换为无条件的 50%的概率 [x_t, x_t] [t, t]
            if use_cfg:
                x_t_a = torch.cat([x_t_a, x_t_a.clone()], dim=0)
                x_t_b = torch.cat([x_t_b, x_t_b.clone()], dim=0)
                x_0 = torch.cat([x_0, x_0.clone()], dim=0)
                t_a = torch.cat([t_a, t_a.clone()], dim=0)
                t_b = torch.cat([t_b, t_b.clone()], dim=0)
                y = torch.cat([y, -torch.ones_like(y)], dim=0)
                x_1 = torch.cat([x_1, x_1.clone()], dim=0)

                y = y.to(device)
            else:
                y = None

            v_pred_a = model(x=x_t_a, t=t_a, y=y)
            v_pred_b = model(x=x_t_b, t=t_b, y=y.detach()).detach()

            f_pred_a = x_t_a + (1 - t_a[:, None, None, None]) * v_pred_a
            f_pred_b = x_t_b + (1 - t_b[:, None, None, None]) * v_pred_b

            # TrajectoryConsistencyLoss
            # 这里为了复用代码凑活下，实际应该改下
            f_loss = rf.mse_loss(f_pred_a, f_pred_a, 0) / consistency_delta_t ** 2
            # VelocityConsistencyLoss
            v_loss =  rf.mse_loss(v_pred_a, v_pred_b, 0)
            loss = f_loss + v_loss

            if not consistency_only:
                rf_loss = rf.mse_loss(v_pred_a, x_1, x_0)
                loss = loss + rf_loss

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
    train(config='./config/train_cfm_config.yaml')
