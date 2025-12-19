import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from cv_imread import cv_imread
from train_model  import train_model
from SaliencyDataset import SaliencyDataset
from ResNet18Saliency import ResNet18Saliency

# ================= 配置区域 =================
# 请确保路径没有多余的空格
TRAIN_ROOT = r"D:\直博\博1\课程\机器学习\机器学习实验课数据集\3-Saliency-TrainSet"
TEST_ROOT = r"D:\直博\博1\课程\机器学习\机器学习实验课数据集\3-Saliency-TestSet"

# 实验超参数
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
LR = 1e-4        # 从零训练建议使用较小的学习率
EPOCHS = 15      # 因为没有预训练权重，收敛较慢，建议适当增加轮数（可先设10测试）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(f"运行设备: {DEVICE}")
# print(f"训练集路径: {TRAIN_ROOT}")


def calc_cc_score(gtsAnn, resAnn):
    gtsAnn = gtsAnn.astype(float)
    resAnn = resAnn.astype(float)

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / (np.std(fixationMap) + 1e-7)

    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / (np.std(salMap) + 1e-7)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]


def calc_kl_score(gtsAnn, resAnn):
    gtsAnn = gtsAnn.astype(float)
    resAnn = resAnn.astype(float)

    # 归一化为概率分布 (Sum=1)
    gtsAnn = gtsAnn / (np.sum(gtsAnn) + 1e-7)
    resAnn = resAnn / (np.sum(resAnn) + 1e-7)

    eps = np.finfo(float).eps
    return np.sum(gtsAnn * np.log((gtsAnn + eps) / (resAnn + eps)))


def apply_heatmap(img_ori, saliency_map):
    """叠加热力图：红色代表高显著性"""
    saliency_map = (saliency_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_ori, 0.6, heatmap, 0.4, 0)
    return overlay


def plot_loss_curve(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training Loss Curve (No Pretrain)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


@torch.no_grad()
def evaluate_and_visualize(model_path, test_loader, output_dir="results_vis"):
    model = ResNet18Saliency(pretrained=True).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    category_metrics = defaultdict(lambda: {'cc': [], 'kl': []})
    saved_counts = defaultdict(int)

    print(f"正在生成结果，图片将保存至: {output_dir}")

    for imgs, _, (ori_h, ori_w), mask_ori_batch, img_paths in tqdm(test_loader):
        imgs = imgs.to(DEVICE)
        preds = model(imgs)
        pred_map = preds[0].squeeze().cpu().numpy()

        # 原始信息
        full_path = img_paths[0]
        category = os.path.basename(os.path.dirname(full_path))
        img_name = os.path.basename(full_path)

        # 还原尺寸
        # h, w = ori_h.item(), ori_w.item()
        # pred_ori = cv2.resize(pred_map, (w, h))

        # 修改
        pred_map = cv2.GaussianBlur(pred_map, (0, 0), sigmaX=10, sigmaY=10)
        h, w = ori_h.item(), ori_w.item()
        pred_ori = cv2.resize(pred_map, (w, h))
        # 修改结束

        mask_ori = mask_ori_batch[0].numpy()

        # 归一化 (0-1)
        pred_norm = (pred_ori - pred_ori.min()) / (pred_ori.max() - pred_ori.min() + 1e-7)

        # 指标计算
        cc = calc_cc_score(mask_ori, pred_norm)
        kl = calc_kl_score(mask_ori, pred_norm)
        category_metrics[category]['cc'].append(cc)
        category_metrics[category]['kl'].append(kl)

        # === 核心可视化：每类保存前2张 ===
        if saved_counts[category] < 2:
            # 读取原图 (BGR)
            img_ori = cv_imread(full_path, type=1)

            # Mask可视化
            gt_vis = (mask_ori).astype(np.uint8)
            gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)

            # 预测可视化 (灰度转BGR)
            pred_vis = (pred_norm * 255).astype(np.uint8)
            pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

            # 叠加图
            overlay = apply_heatmap(img_ori, pred_norm)

            # 拼接: 原图 | GT | 预测 | 叠加
            def add_text(img, text):
                # 在图片上写字
                img_c = img.copy()
                cv2.putText(img_c, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                return img_c

            combined = np.hstack([
                add_text(img_ori, "Original"),
                add_text(gt_vis, "Ground Truth"),
                add_text(pred_vis, f"Pred CC={cc:.2f}"),
                add_text(overlay, "Overlay")
            ])

            cv2.imwrite(os.path.join(output_dir, f"{category}_{img_name}"), combined)
            saved_counts[category] += 1

    # 生成 DataFrame
    summary_data = []
    for cat, metrics in category_metrics.items():
        summary_data.append({
            'Category': cat,
            'Count': len(metrics['cc']),
            'CC': np.mean(metrics['cc']),
            'KL': np.mean(metrics['kl'])
        })
    return pd.DataFrame(summary_data)

if __name__ == '__main__':
    train_dataset = SaliencyDataset(TRAIN_ROOT, img_size=IMG_SIZE, is_train=True)
    val_dataset = SaliencyDataset(TEST_ROOT, img_size=IMG_SIZE, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)  # Windows下设为0更安全
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 2. 实例化模型 (从零训练)
    model = ResNet18Saliency(pretrained=True).to(DEVICE)
    SAVE_PATH = "resnet18_scratch_best.pth"

    # 3. 训练
    history = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, save_path=SAVE_PATH)

    # 4. 绘制 Loss 曲线
    plot_loss_curve(history)

    # 5. 评估并生成对比图
    print("\n开始评估与可视化...")
    df_metrics = evaluate_and_visualize(SAVE_PATH, val_loader, output_dir="results_vis_scratch")

    # 6. 打印最终报表
    print("\n" + "=" * 40)
    print("分类别性能评估 (Sorted by CC)")
    print("=" * 40)
    print(df_metrics.sort_values(by='CC', ascending=False).to_string(index=False))
    print("-" * 40)
    print(f"总体平均 CC: {df_metrics['CC'].mean():.4f}")
    print(f"总体平均 kl: {df_metrics['KL'].mean():.4f}")
