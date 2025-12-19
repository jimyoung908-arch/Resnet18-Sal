import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
from cv_imread import cv_imread



class SaliencyDataset(Dataset):
    def __init__(self, root_dir, img_size=(256, 256), is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train
        self.img_paths = []
        self.mask_paths = []

        stimuli_dir = os.path.join(root_dir, "Stimuli")
        if not os.path.exists(stimuli_dir):
            raise FileNotFoundError(f"找不到目录: {stimuli_dir}")

        img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif")

        # 递归遍历所有子文件夹
        for root, _, files in os.walk(stimuli_dir):
            for file in files:
                if file.lower().endswith(img_extensions):
                    img_path = os.path.join(root, file)

                    # 匹配 Mask (将 Stimuli 替换为 FIXATIONMAPS)
                    mask_dir_part = root.replace("Stimuli", "FIXATIONMAPS")
                    base_name = os.path.splitext(file)[0]

                    found_mask = False
                    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                        mask_candidate = os.path.join(mask_dir_part, base_name + ext)
                        if os.path.exists(mask_candidate):
                            self.img_paths.append(img_path)
                            self.mask_paths.append(mask_candidate)
                            found_mask = True
                            break

        print(f"{'训练集' if is_train else '测试集'}加载完成，共 {len(self.img_paths)} 个样本")

        # 数据增强
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
        ]) if is_train else None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # 使用自定义的 cv_imread
        img_ori = cv_imread(img_path, type=1)
        mask_ori = cv_imread(mask_path, type=0)

        if img_ori is None or mask_ori is None:
            raise ValueError(f"数据读取失败: {img_path}")

        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img_ori.shape[:2]

        # Resize & Tensor转换
        img = cv2.resize(img_ori, self.img_size)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        mask = cv2.resize(mask_ori, self.img_size)
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        if self.transform and self.is_train:
            seed = torch.randint(0, 1000000, (1,)).item()
            torch.manual_seed(seed)
            img = self.transform(img)
            # Mask 同步翻转
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, [2])
                mask = torch.flip(mask, [2])

        return img, mask, (ori_h, ori_w), mask_ori, img_path