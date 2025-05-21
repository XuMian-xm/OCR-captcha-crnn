from PIL import Image
import numpy as np

def analyze_image_grayscale(image_path):
    # 打开图片并转换为灰度模式
    img = Image.open(image_path).convert('L')
    
    # 将图片数据转换为 NumPy 数组
    img_array = np.array(img)
    
    # 计算统计信息
    max_value = np.max(img_array)
    min_value = np.min(img_array)
    mean_value = np.mean(img_array)
    median_value = np.median(img_array)
    std_dev = np.std(img_array)
    
    # 打印统计信息
    print(f"图像路径: {image_path}")
    print(f"图像尺寸: {img.size}")
    print(f"灰度最高值: {max_value}")
    print(f"灰度最低值: {min_value}")
    print(f"灰度平均值: {mean_value:.2f}")
    print(f"灰度中位数: {median_value:.2f}")
    print(f"灰度标准差: {std_dev:.2f}")

# 示例使用
if __name__ == "__main__":
    image_path = "0Fm2HT_205.png"  # 替换为你的图片路径
    analyze_image_grayscale(image_path)