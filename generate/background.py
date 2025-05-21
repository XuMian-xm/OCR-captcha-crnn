import numpy as np
from PIL import Image
import noise
import random
from PIL import ImageFilter
def generate_noise_background(width, height):
    # 生成正态分布噪声
    noise = np.random.normal(230, 10, (height, width)).astype(np.uint8)
    # 截断像素值，确保不低于 220
    noise = np.clip(noise, 200, 255)
    # 转换为三通道 RGB 图像
    img = Image.fromarray(np.stack([noise]*3, axis=-1))
    return img

# 生成一个 200x200 的平滑背景图像
background = generate_noise_background(200, 80)
# background.save("smooth_background.png")
background.show()