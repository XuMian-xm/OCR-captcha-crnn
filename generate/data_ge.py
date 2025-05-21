from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import random
import string
import os
def random_text():
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=6))

def generate_noise_background(width, height):
    noise = np.random.normal(230, 10, (height, width)).astype(np.uint8)
    noise = np.clip(noise, 200, 255)
    img = Image.fromarray(np.stack([noise]*3, axis=-1))
    return img
def generate_textured_background(width, height):
    # 以灰底为主（约220-240），加上细腻噪声
    base = np.ones((height, width), dtype=np.uint8) * 235

    # 加入高频小幅度正态噪声
    noise = np.random.normal(0, 4, (height, width))  # 更低 std 更接近图中纹理
    texture = np.clip(base + noise, 0, 255).astype(np.uint8)

    # 将灰度转为RGB图像
    texture_rgb = np.stack([texture] * 3, axis=-1)
    return Image.fromarray(texture_rgb)

# 获取字体文件夹中的所有字体路径
def get_random_font_path(font_dir="fonts/"):
    font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith(".ttf")]
    return random.choice(font_files)
def draw_random_line(draw, width, height):
    # 随机决定线条位置类型
    mode = random.choice(['top', 'middle', 'bottom', 'through'])

    if mode == 'top':
        y1 = random.randint(5, 15)
        y2 = y1 + random.randint(-5, 5)
    elif mode == 'middle':
        y1 = height // 2 + random.randint(-5, 5)
        y2 = y1 + random.randint(-5, 5)
    elif mode == 'bottom':
        y1 = height - random.randint(20, 30)
        y2 = y1 + random.randint(-5, 5)
    else:  # 'through'
        y1 = random.randint(25, height - 25)
        y2 = y1 + random.randint(-5, 5)

    x1 = random.randint(5, 20)
    x2 = width - random.randint(5, 20)
    draw.line([(x1, y1), (x2, y2)], fill=(100, 100, 100), width=3)

def generate_captcha(text=None, width=200, height=80, font_size=50):
    if text is None:
        text = random_text()

    # image = generate_textured_background(width, height)
    image = generate_noise_background(width, height)
    draw = ImageDraw.Draw(image)

    font_path1 = get_random_font_path("fonts/")
    font_path2 = get_random_font_path("ttf_used/")
    if ('7' in text or 'z' in text or 'Z' in text):
        font_path = font_path1
    else:
        font_path = random.choice([font_path1, font_path2])
    font = ImageFont.truetype(font_path, font_size)

    char_width = width // len(text)
    padding = 10  # 左右边界的预留间距
    # char_width = int(width * 0.8 / len(text))  # 保留20%空白区域
    padding = int(width * 0.1)  # 边距从固定值改为比例值

    for i, char in enumerate(text):
        char_img = Image.new('RGBA', (char_width, height), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((10, 5), char, font=font, fill=(100, 100, 100))

        rotated = char_img.rotate(random.randint(-15, 15), resample=Image.BICUBIC, expand=1)
        x = padding + i * (char_width - padding // 2) + random.randint(-1, 1)
        x = i * (char_width) - 10
        y = random.randint(0, 10) - 10
        
        # 粘贴字符，确保不超出边界
        """
        if x + rotated.width <= width - padding:
            image.paste(rotated, (x, y), rotated)
        else:
            # 如果超出边界，调整位置
            image.paste(rotated, (width - padding - rotated.width, y), rotated)
        """
        image.paste(rotated, (x, y), rotated)

    draw_random_line(draw, width, height)

    image = image.filter(ImageFilter.GaussianBlur(radius=0.7))
    # print(font_path)
    return image, text

# 生成并保存
if __name__ == "__main__":
    image, captcha_text = generate_captcha()
    print(f"验证码内容: {captcha_text}")
    # image.save("captcha_random_line.png")
    image.show()