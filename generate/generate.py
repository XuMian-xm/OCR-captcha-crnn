import os
from data_ge import generate_captcha
def batch_generate_captcha(num_images=10, output_dir="captcha_images"):
    """
    批量生成验证码图片并保存到指定目录。

    参数:
        num_images (int): 要生成的验证码图片数量，默认为 10。
        output_dir (str): 保存验证码图片的目录路径，默认为 "captcha_images"。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    begin = 100
    for i in range(begin,num_images):
        # 生成验证码图片和文本
        image, captcha_text = generate_captcha()
        
        # 构造文件名
        filename = os.path.join(output_dir, f"{captcha_text}_{i+1}.png")
        
        # 保存图片
        image.save(filename)
        print(f"已保存: {filename}，验证码内容: {captcha_text}")

# 示例使用
if __name__ == "__main__":
    # 生成 5 张验证码图片并保存到 "captcha_images" 目录
    batch_generate_captcha(num_images=10000, output_dir="E:/Major/机器学习/data/train")