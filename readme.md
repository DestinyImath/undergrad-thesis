# 毕业论文项目

本仓库是本科毕业设计 《基于深度学习的乳腺超声肿瘤多任务分割与良恶性分类模型研究》 的官方实现代码。项目基于 BUSI 数据集，构建了一套包含严格数据清洗、空间归一化、自研混合损失计算及多模型消融对比的完整医学影像分割框架。

## 🌟 项目亮点 (Highlights)

- 模块化多架构支持：集成并统一接口了基础 U-Net、Attention-UNet、DeepLabV3+ (ResNet-50) 以及 Swin-UNet，方便进行正交消融实验。
- 双模块协同增益：重点探索了 Attention-UNet + ASPP 架构，成功验证了 ASPP（宏观防漏诊）与 Attention（微观抑噪防误诊）在特征空间的 $1+1>2$ 协同效应。
- 严谨的数据管道：针对 BUSI 数据集编写了自动化脚本，完美解决“单图多掩膜 (Multi-mask)”的合并问题；
- 基于 sklearn 实现严格的随机种子切分，并利用 MONAI 的 CacheDataset 实现内存预取加速。
- 混合损失函数：采用 CustomDiceCELoss（Dice + CE 联合驱动），在处理肿瘤极度不规则边缘与类别不平衡时表现更优。

## 📂 目录结构 (Repository Structure)

paper
├── Dataset_BUSI_with_GT/       # 原始下载的 BUSI 数据集 (需自行解压至此)
├── Dataset_BUSI_processed/     # 清洗与合并多重掩膜后的纯净数据集 (由脚本自动生成)
├── result/                     # 自动生成的实验结果 (最佳权重、CSV报表、对比图等)
├── fig/                        # 论文或 README 中使用的各类架构图与说明图
├── model/                      # 存放模型定义的备用或核心组件文件夹
├── monai_func/                 # 基于 MONAI 的特定功能封装模块与部分函数
├── monai_result/               # 基于 MONAI 标准流程跑出的部分函数结果
│
├── func/                       # 🧠 核心功能模块 (解耦的模型与工具库)
│   ├── pretrained_weights/     # 预训练权重存放目录 (如 Swin-Transformer 官方权重)
│   ├── attention_unet.py       # Attention-UNet 模型架构定义
│   ├── data_load.py            # 数据划分、MONAI 数据增强与 DataLoader 构建
│   ├── deeplab_res50.py        # DeepLabV3+ (ResNet-50) 模型架构定义
│   ├── eval_matrices.py        # 评估指标计算逻辑 (手动严谨计算 DSC, IoU, Recall, Precision)
│   ├── function_lib.py         # 核心基础组件库 (ASPP, 自研损失函数 CustomDiceCELoss 等)
│   ├── res50.py                # 纯 ResNet-50 骨干网络基线定义
│   ├── swin_unet.py            # 纯 Transformer 架构 (Swin-UNet) 定义
│   └── unet.py                 # 基础 U-Net 架构定义
│
├── busi_data.py                # 数据集整体测试或主训练入口脚本
├── preprocess_mask.py          # 数据集清洗与多重 Mask 像素级合并脚本
├── .gitignore                  # Git 提交忽略配置
└── readme.md                   # 项目说明文档 (本文档)


## ⚙️ 环境依赖 (Dependencies)

建议使用 Anaconda 创建独立环境，并运行以下代码：

```
pip install requirements.txt
```

## 🚀 快速开始 (Quick Start)

1. 数据准备与清洗

请先将原始的 BUSI 数据集解压至项目根目录，命名为 Dataset_BUSI_with_GT。然后运行清洗脚本，系统将自动合并同一个肿瘤的多个局部掩膜，并剔除无病灶图片：

```
python preprocess_mask.py
```

输出：生成 Dataset_BUSI_processed 文件夹，内部包含纯净的 (原图, _mask.png) 对应数据。

2. 模型训练

通过运行项目的主入口脚本（可根据需要导入 func/ 目录下不同的网络架构如 deeplab_res50.py 或 attention_unet.py），调整当前函数的特殊模块，例如可以将 `use_aspp`、`use_decoder` 等设置为 `True` 或 `False`，还要注意调整函数`MODEL_NAME`的命名，系统会自动完成：

- 将图像 Resize 至 $256 \times 256$。
- 执行 MONAI 提供的几何仿射变换等数据增强。
- 在验证集上严格把控分类阈值（Sigmoid > 0.5）。
- 自动保存最佳 Dice 得分的模型权重至 result/ 目录，并生成定量评估报表与定性预测对比图。

运行下述代码即可（以 `attention_unet` 为例）：

```
python attention_unet.py
```

## 结果展示

在 BUSI 数据集的严格测试下，DeepLabV3+ (ResNet-50) 取得了最优的基线性能，而纯 Transformer 架构在小样本下存在明显的收敛瓶颈（详细机制分析请参阅论文正文）：

| 🤖 Model | 🧩 Spatial Treatment | 📦 Params (M) | 🎯 DSC | 📐 IoU | ⚖️ Acc | 🔍 Recall | 🎯 Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `unet` | None (Standard) | 7.85 | 0.8238 | 0.7004 | 0.9714 | 0.7871 | 0.8641 |
| `unet` | ASPP | 16.77 | 0.8310 | 0.7108 | 0.9719 | 0.8123 | 0.8506 |
| `attention_unet` | None (Standard) | 7.98 | 0.8160 | 0.6892 | 0.9706 | 0.7663 | **0.8725** |
| `attention_unet` | ASPP | 16.90 | 0.8367 | 0.7192 | 0.9724 | 0.8303 | 0.8431 |
| `deeplab_res50` | ResNet-50 | 18.37 | **0.8430** | **0.7286** | **0.9738** | 0.8273 | 0.8593 |
| `res50` | ResNet-50 | 23.51 | 0.8401 | 0.7242 | 0.9729 | **0.8372** | 0.8430 |
| `swin_unet` | None (Standard) | 27.17 | 0.7623 | 0.6158 | 0.9600 | 0.7532 | 0.7716 |
| `swin_unet` | 100 EPOCHS | 27.17 | 0.7260 | 0.5699 | 0.9552 | 0.6973 | 0.7571 |

> **注:** 加粗数据代表该列评估指标中的全局最优值。

## 📚 引用声明 (Acknowledgments)

本项目的顺利完成离不开以下开源框架与研究团队的支持：

* **数据集支持**: 特别感谢 [Dataset of Breast Ultrasound Images (BUSI)](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) 团队公开的高质量乳腺超声图像数据。
* **核心评估框架**: 感谢开源社区提供的 [MONAI (Medical Open Network for AI)](https://monai.io/)，本项目中的数据增强流水线与严格的定量评估指标（Dice, IoU 等）均基于该框架构建。