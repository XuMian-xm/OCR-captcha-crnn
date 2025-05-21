import os
import random
import shutil

# 设置路径
train_dir = 'train'     # 原始训练集目录
val_dir = 'val'         # 验证集保存目录

# 如果 val 目录不存在，则创建
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# 获取所有图片文件
image_files = [f for f in os.listdir(train_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 随机选取 100 张图片
selected_files = random.sample(image_files, 100)

# 移动或复制文件到 val 目录
for file_name in selected_files:
    src_path = os.path.join(train_dir, file_name)
    dst_path = os.path.join(val_dir, file_name)

    # 选择操作方式：移动 or 复制
    # 👇 如果只是测试，建议使用 copy；如果确认无误再用 move
    shutil.move(src_path, dst_path)   # 移动文件
    # shutil.copy(src_path, dst_path)  # 复制文件（保留原图）

print(f"✅ 已处理 {len(selected_files)} 张图片到 '{val_dir}' 目录。")