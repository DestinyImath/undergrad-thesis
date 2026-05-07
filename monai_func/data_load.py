import os
import glob
from sklearn.model_selection import train_test_split
import torch
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    ScaleIntensityd, 
    Resized, 
    RandRotate90d, 
    RandFlipd,
    RandZoomd
)
from monai.data import CacheDataset, DataLoader

def get_busi_data_dicts(data_dir, test_size=0.2, random_state=42):
    """
    步骤 1: 解析文件夹，生成 MONAI 标准的字典列表，并划分为训练集和验证集
    """
    data_dicts = []
    # 分割任务只使用包含肿瘤的类别
    categories = ['benign', 'malignant']
    
    for category in categories:
        cat_dir = os.path.join(data_dir, category)
        if not os.path.exists(cat_dir):
            continue
            
        # 找到所有原图 (不包含 _mask 的文件)
        base_images = [f for f in os.listdir(cat_dir) if '_mask' not in f and f.endswith('.png')]
        
        for base_img in base_images:
            img_path = os.path.join(cat_dir, base_img)
            # 找到我们刚刚清洗好的、唯一对应的 mask
            mask_name = base_img.replace('.png', '_mask.png')
            mask_path = os.path.join(cat_dir, mask_name)
            
            if os.path.exists(mask_path):
                data_dicts.append({"image": img_path, "label": mask_path})
    
    print(f"📦 共找到 {len(data_dicts)} 对有效的 (原图, 掩膜) 数据。")
    
    # 使用 sklearn 进行严格的随机切分，保证每次实验的可重复性
    train_files, val_files = train_test_split(data_dicts, test_size=test_size, random_state=random_state)
    print(f"📊 数据集划分 -> 训练集: {len(train_files)} 张, 验证集: {len(val_files)} 张。")
    
    return train_files, val_files

def get_transforms():
    """
    步骤 2: 定义 MONAI 字典数据变换 (预处理 + 数据增强)
    """
    # 训练集：包含数据预处理和数据增强
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=True), # 读取图像
        EnsureChannelFirstd(keys=["image", "label"]),         # 统一通道格式 (Channel, H, W)
        ScaleIntensityd(keys=["image", "label"]),             # 归一化到 0-1 之间
        Resized(keys=["image", "label"], spatial_size=(256, 256)), # 统一缩放到 256x256 (重要！)
        
        # --- 下面是医学专用的数据增强 ---
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)), # 随机旋转 90 度
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),          # 随机水平翻转
        RandZoomd(keys=["image", "label"], prob=0.3, min_zoom=0.9, max_zoom=1.1) # 随机缩放
    ])
    
    # 验证集：只做预处理，绝对不能做数据增强！
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image", "label"]),
        Resized(keys=["image", "label"], spatial_size=(256, 256))
    ])
    
    return train_transforms, val_transforms

def get_dataloaders(data_dir, batch_size=8, num_workers=4):
    """
    步骤 3: 封装成 PyTorch 可用的 DataLoader
    """
    train_files, val_files = get_busi_data_dicts(data_dir)
    train_transforms, val_transforms = get_transforms()
    
    print("\n⏳ 正在缓存训练集数据到内存中 (能成倍加快训练速度)...")
    # CacheDataset 是 MONAI 的神器，它会把预处理好的图存在内存里，省去每次读取硬盘的时间
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    print("⏳ 正在缓存验证集数据到内存中...")
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

# ================= 测试代码 =================
if __name__ == "__main__":
    # 指向我们刚刚清洗好的干净文件夹
    DATA_DIR = "../Dataset_BUSI_processed"
    
    # 试着获取 DataLoader (如果你在 Windows 下运行报错，请把 num_workers 改为 0)
    train_loader, val_loader = get_dataloaders(DATA_DIR, batch_size=4, num_workers=0)
    
    # 抓取一个 Batch 看看长什么样
    for batch_data in train_loader:
        images, labels = batch_data["image"], batch_data["label"]
        print(f"\n✅ 成功抓取一个 Batch!")
        print(f"   图像张量形状: {images.shape} (Batch, Channel, Height, Width)")
        print(f"   掩膜张量形状: {labels.shape} (Batch, Channel, Height, Width)")
        print(f"   图像像素范围: Min={images.min():.2f}, Max={images.max():.2f}")
        break