import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms

from ctc_model import OCR, Dataset  # 假设你的OCR模型定义在ocr_model.py中
from ce_model import OCR as CaptchaModel  # 假设你的OCR模型定义在ocr_model.py中
from main import test
TRAIN_DIR = './data/train'
VAL_DIR = './data/val'
TEST_DIR = './data/test'
class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.update(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update(val_loss, model)
            self.counter = 0
            
    def update(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        self.val_loss_min = val_loss

def get_data_loaders(dir, batch_size=32, shuffle=True):
    datasets = Dataset(dir)
    data_loaders = DataLoader(datasets, batch_size=batch_size, shuffle=shuffle)
    return data_loaders

def freeze_layers(model, freeze_layers=None):
    """冻结模型特定层并返回可微调的模型实例
    参数:
        model: OCR模型实例
        freeze_layers: 要冻结的层名称列表(如['convlayer','lstm_0'])
                      若为None则默认冻结除最后两层外的所有层
    """
    ocr = OCR() if model is None else model
    
    # 加载预训练权重
    start_epoch, opt_state, losses = ocr.load_model('weights/ocr_50epoch.pth')
    if start_epoch > 0:
        print(f"加载已有模型权重: Epoch {start_epoch}")
    else:
        print("未找到预训练模型，将从头开始训练")

    # 默认冻结策略：除lstm_1和out层外全部冻结
    if freeze_layers is None:
        # 冻结所有参数
        for param in ocr.crnn.parameters():
            param.requires_grad = False
        
        # 解冻最后两层
        for param in ocr.crnn.lstm_1.parameters():
            param.requires_grad = True
        for param in ocr.crnn.out.parameters():
            param.requires_grad = True
    else:
        # 自定义冻结策略
        for name, param in ocr.crnn.named_parameters():
            param.requires_grad = not any(freeze_layer in name for freeze_layer in freeze_layers)

    # 验证冻结效果
    print("\n参数冻结状态：")
    for name, param in ocr.crnn.named_parameters():
        print(f"{name.ljust(30)}: {'可训练' if param.requires_grad else '冻结'}")

    return ocr

def main():
    flag = input("请输入模型类型(1: ctc, 2: ce): ")
    flag = True if flag == '1' else False
    if flag:
        model = OCR()
    else:
        model = CaptchaModel(vocab_size=62, fixed_length=6)
    train_loader = get_data_loaders(VAL_DIR) 
    val_loader = get_data_loaders(TEST_DIR, shuffle=False)
    model.load_model('ocr.pth',weights_only=True)
    # 方案1：使用默认冻结策略（只训练最后两层）
    # frozen_model = freeze_layers(model)
    # 方案2：自定义冻结层（例如只冻结卷积部分）
    frozen_model = freeze_layers(model, freeze_layers=['convlayer'])
    # frozen_model = model
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, frozen_model.crnn.parameters()),
        lr=0.01,  # 微调时建议使用更小的学习率
        momentum=0.9,
        nesterov=True
    )
    
    epochs = 50
    train_loss, val_loss = frozen_model.train(
        num_epochs = epochs,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        print_every=1
    )
    # 保存微调后的模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': frozen_model.crnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': (train_loss, val_loss)
    }, 'captcha_model_finetuned.pth')
    print("模型微调完成并保存。")
    # 绘制损失曲线
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    # plt.show()
    test(frozen_model,'test')
    test(frozen_model,'val')

if __name__ == "__main__":
    main()
    