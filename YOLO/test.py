from ultralytics import YOLO
import os
import glob
char_to_idx = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
    'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18,
    't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, 'A': 26, 'B': 27,
    'C': 28, 'D': 29, 'E': 30, 'F': 31, 'G': 32, 'H': 33, 'I': 34, 'J': 35, 'K': 36,
    'L': 37, 'M': 38, 'N': 39, 'O': 40, 'P': 41, 'Q': 42, 'R': 43, 'S': 44, 'T': 45,
    'U': 46, 'V': 47, 'W': 48, 'X': 49, 'Y': 50, 'Z': 51, '0': 52, '1': 53, '2': 54,
    '3': 55, '4': 56, '5': 57, '6': 58, '7': 59, '8': 60, '9': 61
}
model = YOLO("best.pt")
# 如果需要反向映射（从索引到字符）
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

def predict(path):
    results = model([path], verbose=False)
    fin = ''
    
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # 获取置信度信息
        
        # 创建包含x坐标、类别和置信度的列表
        detections = []
        for index in range(len(xywh)):
            detections.append([
                xywh[index][0].item(),  # x坐标
                names[index],           # 类别名称
                confs[index].item()     # 置信度
            ])
        
        # 如果检测到的字符数量大于6个，按置信度排序并取前6个
        if len(detections) > 6:
            # 按置信度从高到低排序
            detections.sort(key=lambda x: x[2], reverse=True)
            # 只保留前6个
            detections = detections[:6]
        
        # 按x坐标从左到右排序
        detections.sort(key=lambda x: x[0])
        
        # 转换并拼接字符
        for det in detections:
            na = idx_to_char[int(det[1])]
            fin = str(fin) + na
            
    return fin

def get_image_labels(directory_path):
    """
    读取指定路径下的所有PNG文件，提取其文件名前缀作为标签
    
    参数:
        directory_path: 包含PNG文件的目录路径
        
    返回:
        一个列表，每个元素是一个元组 (文件路径, 标签)
    """
    result = []
    
    # 获取目录下所有的PNG文件
    png_files = glob.glob(os.path.join(directory_path, "*.png"))
    
    for file_path in png_files:
        # 获取文件名（不含路径）
        file_name = os.path.basename(file_path)
        
        # 从文件名中提取标签（下划线前的部分）
        if '_' in file_name:
            label = file_name.split('_')[0]
            result.append((file_path, label))
        else:
            print(f"警告: 文件名 {file_name} 不包含下划线，无法提取标签")
    
    return result

def batch_predict(directory_path, output_file="me.txt"):
    """
    批量预测目录下的图片并保存结果
    
    参数:
        directory_path: 包含图片的目录路径
        output_file: 输出文件名
    """
    # 获取目录下所有的图片文件（支持常见格式）
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))
    
    # 打开输出文件
    with open(output_file, 'w') as f:
        for image_path in image_files:
            # 获取文件名（不含路径和扩展名）
            file_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 预测图片
            prediction = predict(image_path)
            
            # 写入结果：编号\t预测结果
            f.write(f"{file_name}\t{prediction}\n")
def main():
    input_path = 'dataset/test/'
    batch_predict(input_path)
if __name__=='__main__':
    main()