import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
import math

from monai.metrics import DiceMetric, MeanIoU # 评估指标(Metrics)依然用 MONAI 的，这很规范，无需重写

from data_load import get_dataloaders
from function_lib import FlexibleUNet, CustomDiceCELoss, StandardSwinUnet

def load_swin_pretrained_encoder(custom_model, hf_model_name="./pretrained_weights/swin_v1"):
    import os
    from transformers import AutoModel
    
    print("-" * 50)
    print(f"📥 准备获取本地权重并执行“多头对齐”动态插值: {hf_model_name}")
    
    try:
        # 加载官方模型
        hf_model = AutoModel.from_pretrained(hf_model_name)
        hf_state_dict = hf_model.state_dict()
        custom_state_dict = custom_model.state_dict()
        
        matched_weights = {}
        hf_keys = list(hf_state_dict.keys())
        hf_values = list(hf_state_dict.values())
        
        success_count = 0
        interp_count = 0
        
        # 遍历我们自己的模型参数
        for custom_key, custom_tensor in custom_state_dict.items():
            # 跳过解码器
            if "up" in custom_key or "dec" in custom_key or "final" in custom_key:
                continue
                
            matched = False
            # 1. 尝试形状完全匹配（Linear层、MLP等）
            for i, hf_tensor in enumerate(hf_values):
                if hf_tensor.numel() > 0 and custom_tensor.shape == hf_tensor.shape:
                    matched_weights[custom_key] = hf_tensor
                    success_count += 1
                    hf_values[i] = torch.empty(0) # 标记已使用
                    matched = True
                    break
            
            # 2. 🌟 核心修复：位置偏置表插值（加入头数校验）
            if not matched and "relative_position_bias_table" in custom_key:
                custom_heads = custom_tensor.shape[1] # 获取我们模型的头数 (e.g., 24)
                
                for i, hf_key in enumerate(hf_keys):
                    # 必须满足：名字包含 bias_table 且 头数(维度1) 必须相等
                    if "relative_position_bias_table" in hf_key and \
                       hf_values[i].numel() > 0 and \
                       hf_values[i].shape[1] == custom_heads:
                        
                        hf_tensor = hf_values[i]
                        src_size = int(math.sqrt(hf_tensor.shape[0])) # 13
                        dst_size = int(math.sqrt(custom_tensor.shape[0])) # 15
                        
                        # 执行双三次插值
                        hf_tensor_spatial = hf_tensor.permute(1, 0).view(1, custom_heads, src_size, src_size)
                        hf_tensor_interp = F.interpolate(
                            hf_tensor_spatial, size=(dst_size, dst_size), mode='bicubic', align_corners=False
                        )
                        hf_tensor_interp = hf_tensor_interp.view(custom_heads, -1).permute(1, 0)
                        
                        matched_weights[custom_key] = hf_tensor_interp
                        interp_count += 1
                        hf_values[i] = torch.empty(0) # 标记已使用
                        matched = True
                        break
        
        # 注入权重
        custom_model.load_state_dict(matched_weights, strict=False)
        print(f"✅ 常规特征权重映射成功：{success_count} 个")
        print(f"🌟 空间位置编码插值成功：{interp_count} 个 (已严格校验多头对齐)")
        print("ℹ️ 解码器部分已保持随机初始化。")
        
    except Exception as e:
        print(f"⚠️ 权重迁移失败！错误原因: {e}")
        # 如果报错，建议检查模型定义中的头数分配
    print("-" * 50)

# ================= 1. 训练超参数与环境配置 =================
EPOCHS = 300
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DATA_DIR = "../Dataset_BUSI_processed"  # 清洗后的数据集路径

# --- 模型动态命名与路径管理 ---
MODEL_NAME = "swin_unet_pretrain" 
RESULT_DIR = f"../result/{MODEL_NAME}"
os.makedirs(RESULT_DIR, exist_ok=True)  # 自动创建对应模型的独立文件夹

SAVE_PATH = os.path.join(RESULT_DIR, f"best_{MODEL_NAME}.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 训练启动！当前模型: {MODEL_NAME.upper()} | 使用设备: {device}")
print(f"📁 本次实验的所有结果将保存在: {RESULT_DIR}")

# ================= 2. 数据与模型初始化 =================
train_loader, val_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, num_workers=4)
## 关闭数据增强
# train_loader, val_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, num_workers=4, use_augmentation=False)

# ================= 修改 3: 实例化你自己的 FlexibleUNet =================
# 现在你对网络的每一层都有了绝对的掌控权
pretrain = True
model = StandardSwinUnet(
    img_size=256,
    in_channels=3,
    out_channels=1,
    feature_size=96
).to(device)
if pretrain:
    load_swin_pretrained_encoder(model, hf_model_name="./pretrained_weights/swin_v1")

# ================= 修改 4: 使用自研的混合损失函数 =================
# 不再依赖 MONAI 的闭源损失计算，使用我们在 function_lib.py 中手写的 CustomDiceCELoss
loss_function = CustomDiceCELoss(sigmoid=True, weight_dice=1.0, weight_ce=1.0)

# 优化器保持不变
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ================= 3. 定义评估指标 (保持 MONAI 标准) =================
dice_metric = DiceMetric(include_background=True, reduction="mean")
iou_metric = MeanIoU(include_background=True, reduction="mean")
history = {
    "train_loss": [],
    "val_loss": [],
    "val_dice": [],
    "val_iou": [],
    "val_acc": []
}

# ================= 4. 正式训练循环 (核心逻辑保持原样) =================
best_dice = 0.0

for epoch in range(1, EPOCHS + 1):
    print(f"\n[{epoch}/{EPOCHS}] " + "="*30)
    
    # ------------------- 训练阶段 -------------------
    model.train()
    epoch_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in train_pbar:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        if images.shape[2:] != (256, 256) or labels.shape[2:] != (256, 256):
            print(f"🚨 捕获到异常尺寸！Image: {images.shape}, Label: {labels.shape}")
        
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
            
            # --- 模型输出后处理 ---
            # 经过 Sigmoid 压缩概率并进行二值化阈值分割
            val_outputs = torch.sigmoid(outputs)
            val_outputs = (val_outputs > 0.5).float()
            
            # 将处理后的结果喂给评估器
            dice_metric(y_pred=val_outputs, y=labels)
            iou_metric(y_pred=val_outputs, y=labels)
            
            # 手动计算像素级准确率 (Accuracy)
            acc = (val_outputs == labels).float().mean()
            val_acc += acc.item()
            
    # 提取最终的平均指标，并重置评估器状态
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

print(f"\n🎉 训练全部结束！最高 Dice 得分: {best_dice:.4f}")
print(f"📁 最佳权重已保存至: {SAVE_PATH}")

print("📉 正在生成训练曲线图与数据报表...")

# === 数据保存与绘图逻辑 ===
df = pd.DataFrame(history)
df.index += 1  
csv_path = os.path.join(RESULT_DIR, f"{MODEL_NAME}_training_history.csv")
df.to_csv(csv_path, index_label="Epoch")

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
plt.ylim(0, 1.05)  
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plot_path = os.path.join(RESULT_DIR, f"{MODEL_NAME}_training_curves.png")
plt.savefig(plot_path, dpi=300) 
plt.close()


print("🖼️ 正在分别生成固定样本与随机样本的对比图...")
# 1. 加载本次训练的最优权重
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()
# 2. 定义绘图函数，减少重复代码
def plot_and_save(idx, title_prefix, filename):
    with torch.no_grad():
        # 提取单张数据
        sample = val_loader.dataset[idx]
        img_tensor = sample["image"].unsqueeze(0).to(device)
        label_tensor = sample["label"].unsqueeze(0).to(device)
        
        # 模型推理
        output = model(img_tensor)
        pred_tensor = (torch.sigmoid(output) > 0.5).float()
        
        # 转为 Numpy (取第1通道)
        img_np = img_tensor.cpu().numpy()[0, 0, :, :] 
        label_np = label_tensor.cpu().numpy()[0, 0, :, :]
        pred_np = pred_tensor.cpu().numpy()[0, 0, :, :]
        
        # 创建独立画布
        plt.figure(figsize=(15, 5))
        
        # --- 1. 原始图 ---
        plt.subplot(1, 3, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title(f"{title_prefix}\nOriginal Image")
        plt.axis('off')
        
        # --- 2. Ground Truth ---
        plt.subplot(1, 3, 2)
        plt.imshow(label_np, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        # --- 3. Prediction ---
        plt.subplot(1, 3, 3)
        plt.imshow(pred_np, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
        
        plt.tight_layout()
        save_full_path = os.path.join(RESULT_DIR, filename)
        plt.savefig(save_full_path, dpi=300)
        plt.show()
        print(f"✅ 图片已保存至: {save_full_path}")
# --- 执行绘图 1: 固定样本 (Index 12) ---
fixed_idx = 42
plot_and_save(fixed_idx, f"Fixed Sample (Index {fixed_idx})", f"{MODEL_NAME}_fixed_sample.png")

# --- 执行绘图 2: 随机样本 ---
available_indices = [i for i in range(len(val_loader.dataset)) if i != fixed_idx]
random_idx = random.choice(available_indices)
plot_and_save(random_idx, f"Random Sample (Index {random_idx})", f"{MODEL_NAME}_random_sample.png")


print(f"✅ 所有结果已成功保存至: {RESULT_DIR}")