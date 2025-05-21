import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from ctc_model import OCR, Dataset
import argparse

def batch_predict(ocr, input_dir, output_file, batch_size=8):
    """
    使用 OCR 模型对指定目录下的所有图片进行批量预测。
    
    参数:
        ocr: OCR 实例，包含已加载的模型
        input_dir: 存放待检测图片的目录
        output_file: 输出识别结果的 txt 文件路径
        batch_size: 批量大小，默认为 8
    """
    # 创建一个临时 Dataset 实例
    dataset = Dataset(input_dir, transform=transforms.Compose([
        transforms.Resize((32, 100)),  # 根据你的需求调整尺寸
        transforms.ToTensor(),
    ]))
    
    # 创建 DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sample_results = []
    ocr.crnn.eval()
    for i in range(len(dataset)):
        img, label = dataset.__getitem__(i)
        logits = ocr.predict(img.unsqueeze(0))
        pred_text = ocr.decode(logits.cpu())
        sample_results.append((label,pred_text))

    # 将预测结果写入文件
    with open(output_file, 'w') as f:
        for idx, (img_path, pred_text) in enumerate(sample_results, 1):
            f.write(f"{os.path.basename(img_path)}\t{pred_text}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量预测OCR')
    parser.add_argument('--input_dir', type=str, required=True, help='测试图像所在目录')
    parser.add_argument('--output_file', type=str, required=True, help='输出识别结果的 txt 文件')
    parser.add_argument('--model_path', type=str, default='ocr.pth', help='模型文件路径')
    
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 OCR 模型
    ocr = OCR()
    checkpoint = torch.load(args.model_path, map_location=device)
    ocr.crnn.load_state_dict(checkpoint['model_state_dict'])
    print("✅ 已加载模型权重:", args.model_path)

    # 开始批量预测
    batch_predict(ocr, args.input_dir, args.output_file)

    print(f"预测完成，结果已保存至 {args.output_file}")
