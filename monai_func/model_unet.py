import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm

# 导入 MONAI 核心原件
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU

# 导入你刚刚写好的数据加载器
from data_load import get_dataloaders

# ================= 1. 训练超参数与环境配置 =================
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DATA_DIR = "../Dataset_BUSI_processed"  # 清洗后的数据集路径

# --- 模型动态命名与路径管理 ---
MODEL_NAME = "unet"
RESULT_DIR = f"../result/{MODEL_NAME}"
os.makedirs(RESULT_DIR, exist_ok=True)  # 自动创建对应模型的独立文件夹

SAVE_PATH = os.path.join(RESULT_DIR, f"best_{MODEL_NAME}.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 MONAI 训练启动！当前模型: {MODEL_NAME.upper()} | 使用设备: {device}")
print(f"📁 本次实验的所有结果将保存在: {RESULT_DIR}")

# ================= 2. 数据与模型初始化 =================
# 获取数据 (注意 num_workers 根据你的电脑配置调整，Linux服务器可以直接用4或8)
train_loader, val_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, num_workers=4)

# 使用 U-Net
model = UNet(
    spatial_dims=2,          # 2D 图像
    in_channels=3,           # RGB 3通道输入 (对应你数据加载器里的 [Batch, 3, 256, 256])
    out_channels=1,          # 单类别分割（只需要输出肿瘤的 Mask）
    channels=(32, 64, 128, 256, 512),  # 网络深度与特征图通道数
    strides=(2, 2, 2, 2),    # 4次下采样
    # num_res_units=2          # 引入残差连接 (ResNet 机制)，加速收敛并防止梯度消失
).to(device)

# 定义 MONAI 专属的混合损失函数：Dice Loss + 交叉熵 (BCE)
# sigmoid=True 会自动在内部对模型的原始输出进行 sigmoid 激活
loss_function = DiceCELoss(sigmoid=True)
# 优化器
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ================= 3. 定义 MONAI 标准评估指标 =================
# reduction="mean" 表示对整个 Batch 取平均
dice_metric = DiceMetric(include_background=True, reduction="mean")
iou_metric = MeanIoU(include_background=True, reduction="mean")
history = {
    "train_loss": [],
    "val_loss": [],
    "val_dice": [],
    "val_iou": [],
    "val_acc": []
}

# ================= 4. 正式训练循环 =================
best_dice = 0.0

for epoch in range(1, EPOCHS + 1):
    print(f"\n[{epoch}/{EPOCHS}] " + "="*30)
    
    # ------------------- 训练阶段 -------------------
    model.train()
    epoch_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in train_pbar:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算损失并反向传播
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    avg_train_loss = epoch_loss / len(train_loader)
    
    # ------------------- 验证阶段 -------------------
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validating", leave=False)
        for batch in val_pbar:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(images)
            
            # 记录验证集损失
            val_loss += loss_function(outputs, labels).item()
            
            # --- 核心操作：模型输出后处理 ---
            # 1. 经过 Sigmoid 把输出压缩到 0~1 之间的概率
            val_outputs = torch.sigmoid(outputs)
            # 2. 阈值分割：大于 0.5 的认为是肿瘤 (1)，否则是背景 (0)
            val_outputs = (val_outputs > 0.5).float()
            
            # 将处理后的结果喂给 MONAI 评估器
            dice_metric(y_pred=val_outputs, y=labels)
            iou_metric(y_pred=val_outputs, y=labels)
            
            # 手动计算像素级准确率 (Accuracy)
            acc = (val_outputs == labels).float().mean()
            val_acc += acc.item()
            
    # 从 MONAI 评估器中提取最终的平均指标，并重置评估器状态
    avg_dice = dice_metric.aggregate().item()
    avg_iou = iou_metric.aggregate().item()
    dice_metric.reset()
    iou_metric.reset()
    
    avg_val_loss = val_loss / len(val_loader)
    avg_acc = val_acc / len(val_loader)

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_dice"].append(avg_dice)
    history["val_iou"].append(avg_iou)
    history["val_acc"].append(avg_acc)
    
    print(f"📉 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"📊 Val Metrics -> Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f} | Acc: {avg_acc:.4f}")
    
    # ------------------- 保存最优模型 -------------------
    if avg_dice > best_dice:
        print(f"🏆 发现新高 Dice: {best_dice:.4f} -> {avg_dice:.4f}！正在保存权重...")
        best_dice = avg_dice
        torch.save(model.state_dict(), SAVE_PATH)

print(f"\n🎉 训练全部结束！MONAI U-Net 最高 Dice 得分: {best_dice:.4f}")
print(f"📁 最佳权重已保存至: {SAVE_PATH}")

print("📉 正在生成训练曲线图与数据报表...")

# 1. 保存数据为 CSV 表格 (方便写论文时查阅数据)
df = pd.DataFrame(history)
df.index += 1  # 让索引从 1 开始代表 Epoch 1
csv_path = os.path.join(RESULT_DIR, f"{MODEL_NAME}_training_history.csv")
df.to_csv(csv_path, index_label="Epoch")

# 2. 绘制并保存曲线图
plt.figure(figsize=(12, 5))

# 子图 1: 损失曲线 (Loss)
plt.subplot(1, 2, 1)
plt.plot(df.index, df["train_loss"], label="Train Loss", marker='o', markersize=3)
plt.plot(df.index, df["val_loss"], label="Validation Loss", marker='o', markersize=3)
plt.title(f"{MODEL_NAME.upper()} - Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 子图 2: 评估指标曲线 (Metrics)
plt.subplot(1, 2, 2)
plt.plot(df.index, df["val_dice"], label="Dice Score", marker='s', markersize=3)
plt.plot(df.index, df["val_iou"], label="IoU Score", marker='^', markersize=3)
plt.plot(df.index, df["val_acc"], label="Accuracy", marker='d', markersize=3)
plt.title(f"{MODEL_NAME.upper()} - Validation Metrics")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.ylim(0, 1.05)  # 分数上限设定为 1
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plot_path = os.path.join(RESULT_DIR, f"{MODEL_NAME}_training_curves.png")
plt.savefig(plot_path, dpi=300)  # 保存为 300dpi 高清大图，论文必备
plt.close()

print(f"✅ 所有结果已成功保存至: {RESULT_DIR}")