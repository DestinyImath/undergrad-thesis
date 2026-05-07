import os
import glob
import cv2
import numpy as np
import shutil
from tqdm import tqdm

def process_busi_dataset(src_dir, dst_dir):
    """
    清洗并合并 BUSI 数据集的多重 Mask
    """
    print(f"🚀 开始处理数据集...\n源目录: {src_dir}\n目标目录: {dst_dir}\n")
    
    # 支持的类别文件夹
    categories = ['benign', 'malignant', 'normal']
    
    # 确保目标总文件夹存在
    os.makedirs(dst_dir, exist_ok=True)
    
    # 记录处理了多少张图片
    total_processed = 0
    multi_mask_count = 0

    for category in categories:
        src_cat_dir = os.path.join(src_dir, category)
        dst_cat_dir = os.path.join(dst_dir, category)
        
        # 如果源文件夹不存在，跳过
        if not os.path.exists(src_cat_dir):
            print(f"⚠️ 找不到文件夹: {src_cat_dir}，已跳过。")
            continue
            
        # 为输出创建对应的类别文件夹
        os.makedirs(dst_cat_dir, exist_ok=True)
        
        # 找到当前类别下所有的 原图 (排除掉名字里带有 '_mask' 的文件)
        all_files = os.listdir(src_cat_dir)
        base_images = [f for f in all_files if '_mask' not in f and f.endswith('.png')]
        
        print(f"📁 正在处理 [{category}] 类别，共找到 {len(base_images)} 张原图...")
        
        # 使用 tqdm 增加进度条显示
        for base_img_name in tqdm(base_images, desc=category):
            # 原图的完整路径
            src_img_path = os.path.join(src_cat_dir, base_img_name)
            # 目标原图的完整路径 (直接复制，保持不变)
            dst_img_path = os.path.join(dst_cat_dir, base_img_name)
            shutil.copy2(src_img_path, dst_img_path)
            
            # --- 开始处理 Mask ---
            # 提取文件名前缀，例如 "benign (54)"
            base_name = os.path.splitext(base_img_name)[0]
            
            # 利用通配符查找所有相关的 Mask，例如 "benign (54)_mask*.png"
            mask_pattern = os.path.join(src_cat_dir, f"{base_name}_mask*.png")
            mask_paths = glob.glob(mask_pattern)
            
            if len(mask_paths) == 0:
                print(f"\n⚠️ 警告: 图片 {base_img_name} 找不到任何对应的 Mask！")
                continue
            
            if len(mask_paths) > 1:
                multi_mask_count += 1
                
            # 读取并合并 Mask
            merged_mask = None
            for mp in mask_paths:
                # 以灰度模式读取 Mask
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if merged_mask is None:
                    merged_mask = m
                else:
                    # 使用 numpy 的 maximum 合并。只要有白色(255)的地方，就会保留白色
                    merged_mask = np.maximum(merged_mask, m)
            
            # 将合并后的 Mask 保存到目标文件夹，统一命名为 "原名_mask.png"
            dst_mask_name = f"{base_name}_mask.png"
            dst_mask_path = os.path.join(dst_cat_dir, dst_mask_name)
            cv2.imwrite(dst_mask_path, merged_mask)
            
            total_processed += 1

    print("\n✅ 数据集清洗与合并完成！")
    print(f"总计处理原图: {total_processed} 张")
    print(f"其中包含多重 Mask 并成功合并的图片: {multi_mask_count} 张")
    print(f"纯净的数据集已保存在 -> {dst_dir}")

if __name__ == "__main__":
    # 请根据你的实际情况修改这两个路径
    SOURCE_DIRECTORY = "Dataset_BUSI_with_GT"  # 你截图中的原始文件夹名
    DESTINATION_DIRECTORY = "Dataset_BUSI_processed" # 新生成的干净文件夹名
    
    process_busi_dataset(SOURCE_DIRECTORY, DESTINATION_DIRECTORY)