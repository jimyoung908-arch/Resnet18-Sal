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


def nss(pred, fixation, eps=1e-8):
    # pred: 连续显著图
    # fixation: 二值注视点图
    pred_mean = pred.mean(dim=(2,3), keepdim=True)
    pred_std = torch.sqrt(((pred - pred_mean) ** 2).mean(dim=(2,3), keepdim=True) + eps)

    pred_norm = (pred - pred_mean) / (pred_std + eps)
    nss_score = (pred_norm * fixation).sum(dim=(1,2,3)) / (fixation.sum(dim=(1,2,3)) + eps)
    return nss_score.mean()


class SaliencyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target, fixation):
        # 归一化为概率分布（KL 必须）
        pred = nn.functional.relu(pred)
        pred = pred / (pred.sum(dim=(2,3), keepdim=True) + 1e-8)
        target = target / (target.sum(dim=(2,3), keepdim=True) + 1e-8)

        loss_kl = kl_divergence(pred, target)
        loss_cc = 1 - correlation_coefficient(pred, target)
        loss_nss = -nss(pred, fixation)

        total_loss = (
            self.alpha * loss_kl +
            self.beta * loss_cc +
            self.gamma * loss_nss
        )
        return total_loss


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, save_path="best_model.pth"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = SaliencyLoss(alpha=1.0, beta=0.5, gamma=1.0)
    # 学习率调度：每5轮降低一半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')

    print(f"开始训练，共 {epochs} 轮 (无预训练权重)...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for imgs, masks, fixations, _, _ in pbar:
            # imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            imgs, masks, fixations = imgs.to(DEVICE), masks.to(DEVICE), fixations.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            # loss = criterion(outputs, masks)
            loss = criterion(outputs, masks, fixations)
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
            for imgs, masks, fixations, _, _ in val_loader:
                imgs, masks, fixations = imgs.to(DEVICE), masks.to(DEVICE), fixations.to(DEVICE)
                outputs = model(imgs)
                # loss = criterion(outputs, masks)
                loss = criterion(outputs, masks, fixations)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(epoch_val_loss)

        scheduler.step()
        print(f"Epoch {epoch + 1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)

    return history