import torch
from monai.networks.nets import UNet
from monai.transforms import Compose, RandRotate90d, RandFlipd
from monai.data import Dataset, DataLoader

print("=== MONAI 极简 2D 分割测试启动 ===")

# 1. 构建字典数据格式 (Dictionary Format) —— 这是 MONAI 最核心的灵魂！
# 模拟两张 128x128 的单通道灰度图 (image) 和对应的二值化掩膜 (label)
data_dicts = [
    {"image": torch.randn(1, 128, 128), "label": torch.randint(0, 2, (1, 128, 128)).float()},
    {"image": torch.randn(1, 128, 128), "label": torch.randint(0, 2, (1, 128, 128)).float()}
]
print("1. 模拟数据生成完毕。")

# 2. 定义基于字典的数据增强流水线 (Dictionary Transforms)
# 注意末尾的 'd'，代表这是针对 dictionary 的操作，它会保证图像和标签同步翻转！
transforms = Compose([
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
])
print("2. 数据增强流水线组装完毕。")

# 3. 放入 Dataset 和 DataLoader
dataset = Dataset(data=data_dicts, transform=transforms)
dataloader = DataLoader(dataset, batch_size=2)
print("3. DataLoader 准备就绪。")

# 4. 初始化经典 U-Net 并塞入显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=2,          # 处理 2D 图像
    in_channels=1,           # 超声图像是单通道(黑白)
    out_channels=1,          # 输出单通道的预测 Mask
    channels=(16, 32, 64),   # 网络各层的厚度（这里设得很小，方便秒跑）
    strides=(2, 2)           # 下采样步长
).to(device)
print("4. U-Net 模型已加载至:", device)

# 5. 模拟一次前向传播 (Forward Pass)
for batch in dataloader:
    inputs = batch["image"].to(device)
    labels = batch["label"].to(device)
    
    # 将图像喂给模型
    outputs = model(inputs)
    
    print("\n--- 测试结果 ---")
    print(f"输入张量形状: {inputs.shape}  -> (Batch, Channel, Height, Width)")
    print(f"输出预测形状: {outputs.shape}  -> 完美匹配输入！")
    print("================================")
    print("🎉 恭喜！你的 MONAI 分割流水线已全链路跑通！")
    break