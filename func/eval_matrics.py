import os
import torch
import pandas as pd
from tqdm import tqdm

# ================= 导入模型与数据组件 =================
# 请确保从 function_lib.py 中正确导入以下类
from function_lib import (
    StandardSwinUnet, 
    FlexibleUNet, 
    UniversalDeepLabV3Plus, 
    ResNetBackbone,
    PureResNet50_FCN
)
from data_load import get_dataloaders

DATA_DIR = "../Dataset_BUSI_processed"  # 清洗后的数据集路径

def evaluate_single_model(model_name, backbone_name, model_instance, val_loader, device="cuda"):
    """
    读取特定模型的权重，评估并返回论文所需的所有指标。
    """
    RESULT_DIR = f"../result/{model_name}"
    SAVE_PATH = os.path.join(RESULT_DIR, f"best_{model_name}.pth")
    
    # 1. 检查权重是否存在
    if not os.path.exists(SAVE_PATH):
        print(f"⚠️ 警告: 找不到模型权重 {SAVE_PATH}，跳过该模型。")
        return None
        
    print(f"🔄 正在加载并评估: {model_name} ...")
    
    # 2. 加载权重并设置为验证模式 (weights_only=True 规避警告)
    # strict=False 允许略微的结构容错，但基于报错，我们已经完美对齐了结构
    model_instance.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
    model_instance.to(device)
    model_instance.eval()
    
    # 3. 计算参数量 Parameters (M)
    total_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
    params_m = total_params / 1e6
    
    # 4. 初始化混淆矩阵变量
    TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0
    
    # 5. 在验证集上进行推理
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating {model_name}", leave=False):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            
            outputs = model_instance(images)
            
            # 后处理：Sigmoid 压缩概率并二值化
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            labels = labels.float()
            
            # 严格计算像素级混淆矩阵元素
            TP += (preds * labels).sum().item()
            TN += ((1 - preds) * (1 - labels)).sum().item()
            FP += (preds * (1 - labels)).sum().item()
            FN += ((1 - preds) * labels).sum().item()

    # 6. 计算所有医学分割定量指标 (加入极小值 eps 防止除以 0)
    eps = 1e-6 
    
    dsc = (2.0 * TP + eps) / (2.0 * TP + FP + FN + eps)       # Dice 相似系数
    iou = (TP + eps) / (TP + FP + FN + eps)                   # 交并比
    acc = (TP + TN + eps) / (TP + TN + FP + FN + eps)         # 像素准确率
    recall = (TP + eps) / (TP + FN + eps)                     # 召回率 (敏感度)
    precision = (TP + eps) / (TP + FP + eps)                  # 精确率 (阳性预测值)
    
    # 返回格式化字典
    return {
        "Model": model_name.split('+')[0], # 简化的模型名称
        "Backbone": backbone_name,
        "Parameters (M)": f"{params_m:.2f}",
        "DSC": f"{dsc:.4f}",
        "IoU": f"{iou:.4f}",
        "Acc": f"{acc:.4f}",
        "Recall": f"{recall:.4f}",
        "Precision": f"{precision:.4f}"
    }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载数据 (只提取 val_loader 用于评估)
    _, val_loader = get_dataloaders(DATA_DIR, batch_size=8, num_workers=4)
    
    # 2. 定义论文所需评估的模型字典
    # ⚠️ 这里的参数严格对齐了你报错日志中暴露的权重维度！
    models_to_evaluate = {
        # ================= 1. 基础 U-Net 系列 =================
        "unet": ("None (Standard)", FlexibleUNet(use_aspp=False, use_attention=False)),
        "unet+small_data": ("small_data", FlexibleUNet(use_aspp=False, use_attention=False)),
        "unet+ASPP": ("ASPP", FlexibleUNet(use_aspp=True, use_attention=False)),

        # ================= 2. Attention U-Net 系列 =================
        "attention_unet": ("None (Standard)", FlexibleUNet(use_aspp=False, use_attention=True)),
        "attention_unet+small_data": ("small_data", FlexibleUNet(use_aspp=False, use_attention=True)),
        "attention_unet+ASPP": ("ASPP", FlexibleUNet(use_aspp=True, use_attention=True)),
        "attention_unet+ASPP+small_data": ("ASPP + small_data", FlexibleUNet(use_aspp=True, use_attention=True)),

        # ================= 3. DeepLabV3+ 系列 =================
        # 根据你的权重维度报错：low_level_channels是256，high_level_channels是1024
        "deeplab_res50": (
            "ResNet-50", 
            UniversalDeepLabV3Plus(
                backbone=ResNetBackbone("resnet50"),
                low_level_channels=256,   
                high_level_channels=1024, 
                out_channels=1,
                spatial_dims=2,
                aspp_out_channels=256,
                decoder_channels=256,
                use_aspp=True,
                use_decoder=True
            )
        ), 
        "deeplab_res50+no_aspp": (
            "no_aspp", 
            UniversalDeepLabV3Plus(
                backbone=ResNetBackbone("resnet50"),
                low_level_channels=256,   
                high_level_channels=1024, 
                out_channels=1,
                spatial_dims=2,
                aspp_out_channels=256,
                decoder_channels=256,
                use_aspp=False,
                use_decoder=True
            )
        ), 
        "deeplab_res50+no_decoder": (
            "no_decoder", 
            UniversalDeepLabV3Plus(
                backbone=ResNetBackbone("resnet50"),
                low_level_channels=256,   
                high_level_channels=1024, 
                out_channels=1,
                spatial_dims=2,
                aspp_out_channels=256,
                decoder_channels=256,
                use_aspp=True,
                use_decoder=False
            )
        ), 
        "deeplab_res50+small_data": (
            "small_data", 
            UniversalDeepLabV3Plus(
                backbone=ResNetBackbone("resnet50"),
                low_level_channels=256,   
                high_level_channels=1024, 
                out_channels=1,
                spatial_dims=2,
                aspp_out_channels=256,
                decoder_channels=256,
                use_aspp=True,
                use_decoder=True
            )
        ), 
        "res50": (
            "ResNet-50", 
            PureResNet50_FCN(out_channels=1, spatial_dims=2)
        ),
        "res50+small_data": (
            "small_data", 
            PureResNet50_FCN(out_channels=1, spatial_dims=2)
        ),
        
        # ================= 4. Swin-UNet 系列 =================
        "swin_unet": ("Swin-T (Window=8)", StandardSwinUnet(img_size=256, feature_size=96)),
        "swin_unet+100 epochs": ("Swin-T (Window=8)", StandardSwinUnet(img_size=256, feature_size=96)),
        "swin_unet+small_data": ("Swin-T (Window=8)", StandardSwinUnet(img_size=256, feature_size=96))
    }
    
    results = []
    
    # 3. 循环评估所有模型
    for save_name, (backbone_str, net_instance) in models_to_evaluate.items():
        res = evaluate_single_model(save_name, backbone_str, net_instance, val_loader, device)
        if res is not None:
            results.append(res)
            
    # 4. 使用 Pandas 生成表格并打印
    if results:
        df = pd.DataFrame(results)
        print("\n\n" + "="*90)
        print(" 表 4.1 多种主流深度学习分割架构在 BUSI 数据集上的定量性能评估 ".center(85))
        print("="*90)
        # 打印可以直接转换为 LaTeX 或 Markdown 的表格格式
        print(df.to_markdown(index=False, colalign=("center",)*len(df.columns)))
        print("="*90)
        
        # 自动保存为 CSV，方便后续导入 Excel
        os.makedirs("../result", exist_ok=True)
        df.to_csv("../result/Table_4_1_results.csv", index=False)
        print("\n✅ 表格数据已成功保存至 ../result/Table_4_1_results.csv")