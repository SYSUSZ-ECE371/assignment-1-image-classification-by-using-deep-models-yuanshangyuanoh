import os
import random
import shutil
from pathlib import Path

# 设置随机种子以确保可重复性
random.seed(42)

# 原始数据路径
raw_data_path = "data/flower_dataset_raw"
# 目标数据路径
target_data_path = "data/flower_dataset"

# 类别列表
classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# 创建目标目录结构
Path(target_data_path).mkdir(exist_ok=True)
Path(f"{target_data_path}/train").mkdir(exist_ok=True)
Path(f"{target_data_path}/val").mkdir(exist_ok=True)

for cls in classes:
    Path(f"{target_data_path}/train/{cls}").mkdir(exist_ok=True, parents=True)
    Path(f"{target_data_path}/val/{cls}").mkdir(exist_ok=True, parents=True)

# 创建classes.txt文件
with open(f"{target_data_path}/classes.txt", "w") as f:
    for cls in enumerate(classes):
        f.write(f"{cls}\n")

# 准备训练和验证集
train_lines = []
val_lines = []

for class_idx, cls in enumerate(classes):
    # 获取该类所有图像文件
    class_dir = f"{raw_data_path}/{cls}"
    images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 随机打乱
    random.shuffle(images)

    # 计算分割点 (80%训练，20%验证)
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # 复制文件并创建标注
    for img in train_images:
        src = f"{class_dir}/{img}"
        dst = f"{target_data_path}/train/{cls}/{img}"
        shutil.copy(src, dst)
        train_lines.append(f"{cls}/{img} {class_idx}\n")

    for img in val_images:
        src = f"{class_dir}/{img}"
        dst = f"{target_data_path}/val/{cls}/{img}"
        shutil.copy(src, dst)
        val_lines.append(f"{cls}/{img} {class_idx}\n")

# 写入标注文件
with open(f"{target_data_path}/train.txt", "w") as f:
    f.writelines(train_lines)

with open(f"{target_data_path}/val.txt", "w") as f:
    f.writelines(val_lines)

print(f"训练集样本数: {len(train_lines)}")
print(f"验证集样本数: {len(val_lines)}")