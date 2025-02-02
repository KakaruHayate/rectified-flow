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
    shortcut = config.get('shortcut', True)
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
    print(f'use_cfg: {shortcut}')
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
    model = MiniUnet(base_channels, shortcut=shortcut)
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
            b = x_1.shape[0]
            # 均匀采样[0, 1]的时间t randn 标准正态分布
            t = torch.rand(x_1.size(0))
            # 这里提前截取下来后面cfg方便一些
            t_rf = torch.clip(t[b // 4 :], 1e-7, 1-1e-7)
            t_sc = torch.clip(t[: b // 4], 0, 1-1e-3)
            d = 0.5 * (1.0 - t_sc) * torch.rand(b // 4)

            # 生成flow（实际上是一个点）
            x_1_rf = x_1[b // 4 :] # 这里提前截取下来后面cfg方便一些
            x_1_sc = x_1[: b // 4]
            y_rf = y[b // 4 :]
            y_sc = y[: b // 4]
            x_t, x_0 = rf.create_flow(x_1_rf, t_rf) # rf
            x_t_sc, x_0_sc = rf.create_flow(x_1_sc, t_sc)
            

            # 4090 大概占用显存3G
            x_t = x_t.to(device)
            x_0 = x_0.to(device)
            x_1_rf = x_1_rf.to(device)
            t_rf = t_rf.to(device)
            # SC
            d = d.to(device)
            x_t_sc = x_t_sc.to(device)
            x_0_sc = x_0_sc.to(device)
            x_1_sc = x_1_sc.to(device)
            t_sc = t_sc.to(device)

            optimizer.zero_grad()

            # 这里我们要做一个数据的复制和拼接，复制原始x_1，把一半的y替换成-1表示无条件生成，这里也可以直接有条件、无条件累计两次计算两次loss的梯度
            # 一定的概率，把有条件生成换为无条件的 50%的概率 [x_t, x_t] [t, t]
            if use_cfg:
                x_t = torch.cat([x_t, x_t.clone()], dim=0)
                x_t_sc = torch.cat([x_t_sc, x_t_sc.clone()], dim=0)
                y_rf = torch.cat([y_rf, -torch.ones_like(y_rf)], dim=0)
                y_sc = torch.cat([y_sc, -torch.ones_like(y_sc)], dim=0)
                x_1_rf = torch.cat([x_1_rf, x_1_rf.clone()], dim=0)
                x_1_sc = torch.cat([x_1_sc, x_1_sc.clone()], dim=0)
                x_0 = torch.cat([x_0, x_0.clone()], dim=0)
                y_rf = y_rf.to(device)
                y_sc = y_sc.to(device)
                t_rf = torch.cat([t_rf, t_rf.clone()], dim=0)
                t_sc = torch.cat([t_sc, t_sc.clone()], dim=0)
                d = torch.cat([d, d.clone()], dim=0)
            else:
                y = None

            v_pred = model(x=x_t, t=t_rf, y=y_rf, d=torch.zeros_like(t_rf))

            loss = rf.mse_loss(v_pred, x_1_rf, x_0)
            
            # SC
            v_pred_sc1 = model(x=x_t_sc, t=t_sc, y=y_sc, d=d)
            x_t2_sc = x_t_sc + v_pred_sc1 * d[:, None, None, None]
            v_pred_sc2 = model(x=x_t2_sc, t=t_sc+d, y=y_sc, d=d)
            v_pred_sc_mean = 0.5 *(v_pred_sc1 + v_pred_sc2).detach()
            v_pred_sc = model(x=x_t_sc, t=t_sc, y=y_sc, d=d*2)
            sc_loss = rf.mse_loss(v_pred_sc, v_pred_sc_mean, 0) # 复用
            
            loss = loss+sc_loss

            loss.backward()
            optimizer.step()

            if batch % batch_print_interval == 0:
                print(f'[Epoch {epoch}] [batch {batch}] loss: {loss.item()} sc_loss: {sc_loss.item()}')

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
    train(config='./config/train_shortcut.yaml')
