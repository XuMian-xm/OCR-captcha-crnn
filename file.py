import torch
import torch.nn as nn
from torchsummary import summary
from ctc_model import OCR

gray_image = torch.randn(2, 1, 80, 200)  # 示例输入
 
# 将灰度图像转换为RGB图像
rgb_image = gray_image.repeat(1, 3, 1, 1)  # 复制灰度通道以创建3个通道

if __name__ == '__main__':
    ocr = OCR()
    summary(ocr.crnn, input_size=(3,80,200))  # 假设输入图像大小为 32x100


