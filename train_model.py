import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm



def kl_divergence(pred, target, eps=1e-8):
    # pred, target: [B, 1, H, W]，已归一化
    return torch.sum(target * torch.log((target + eps) / (pred + eps)), dim=(1,2,3)).mean()


def correlation_coefficient(pred, target, eps=1e-8):
    # CC = cov / (std1 * std2)
    pred_mean = pred.mean(dim=(2,3), keepdim=True)
    target_mean = target.mean(dim=(2,3), keepdim=True)

    pred_std = torch.sqrt(((pred - pred_mean) ** 2).mean(dim=(2,3), keepdim=True) + eps)
    target_std = torch.sqrt(((target - target_mean) ** 2).mean(dim=(2,3), keepdim=True) + eps)

    cc = ((pred - pred_mean) * (target - target_mean)).mean(dim=(2,3), keepdim=True) / (pred_std * target_std)
    return cc.mean()



class SaliencyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # # 归一化为概率分布（KL 必须）
        # pred = nn.functional.relu(pred)
        # pred = pred / (pred.sum(dim=(2,3), keepdim=True) + 1e-8)
        # target = target / (target.sum(dim=(2,3), keepdim=True) + 1e-8)
        #
        # loss_kl = kl_divergence(pred, target)
        # loss_cc = 1 - correlation_coefficient(pred, target)
        # loss_mse = self.mse(pred, target)
        # 输入 pred: 模型的原始输出 logits (未经过 sigmoid)
        # 输入 target: 0-1 的连续显著性图

        # 1. 生成两个分支
        pred_sigmoid = torch.sigmoid(pred)  # 范围 0-1，用于 MSE/CC

        # 2. 计算 CC Loss (这是去中心化的关键)
        # 归一化为零均值单位方差 (Z-Score)，而不是概率分布
        pred_mean = pred_sigmoid.mean(dim=(2, 3), keepdim=True)
        target_mean = target.mean(dim=(2, 3), keepdim=True)
        pred_std = pred_sigmoid.std(dim=(2, 3), keepdim=True) + 1e-8
        target_std = target.std(dim=(2, 3), keepdim=True) + 1e-8

        pred_norm = (pred_sigmoid - pred_mean) / pred_std
        target_norm = (target - target_mean) / target_std

        cc = (pred_norm * target_norm).mean(dim=(2, 3)).mean()
        loss_cc = 1 - cc  # 我们希望 CC -> 1

        # 3. 计算 MSE Loss (在 0-1 空间计算，拒绝 Scale Collapse)
        # 这一步能让模型学会把背景（非显著区域）真的推向全黑
        loss_mse = nn.functional.mse_loss(pred_sigmoid, target)

        total_loss = (
            self.beta * loss_cc +
            self.gamma * loss_mse
        )
        return total_loss


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, save_path="best_model.pth"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()

    # criterion = SaliencyLoss(alpha=1.0, beta=2.0, gamma=5.0)
    criterion = SaliencyLoss(beta=10.0, gamma=2.0)
    # 学习率调度：每5轮降低一半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')

    print(f"开始训练，共 {epochs} 轮...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for imgs, masks, _, _, _ in pbar:
            # imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            # loss = criterion(outputs, masks)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks, _, _, _ in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)
                # loss = criterion(outputs, masks)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(epoch_val_loss)

        scheduler.step()
        print(f"Epoch {epoch + 1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)

    return history