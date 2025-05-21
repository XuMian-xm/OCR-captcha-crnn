# 六位验证码识别CRNN模型

## 项目简介
本项目是一个基于CRNN架构的验证码识别系统，专门用于识别六位数字或字母组合的验证码。
CRNN结合了卷积神经网络（CNN）和循环神经网络（RNN）的优势，能够有效地处理图像序列数据，适用于验证码等场景。

本项目同时包含了YOLO模型的训练和预测代码，用于比较CRNN模型与YOLO模型在验证码识别任务上的性能。

## 项目结构
- `data/`: 存放训练数据和测试数据。
- `weights/`: 存放模型文件。
- `generate/`: 用于额外生成数据。
- `YOLO/`: 用于训练YOLO模型。

## 环境要求
- Python 3.8+
- PyTorch 1.8+
- OpenCV
- NumPy
- Matplotlib
- torchvision
- Pillow

## 直接运行
```bash
python predict.py --input_dir ./data/test/ --output_file output.txt
```