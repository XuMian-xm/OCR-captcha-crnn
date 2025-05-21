import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import numpy as np

# 定义常量
CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR2IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX2CHAR = CHARS
VOCAB_SIZE = len(CHARS)
CAPTCHA_LENGTH = 6

class Dataset(Dataset):
    def __init__(self, img_dir):
        self.img_list = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]
        self.transform = transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        path = self.img_list[idx]
        label = os.path.basename(path).split('_')[0]  # 根据你的文件命名规则修改
        assert len(label) == CAPTCHA_LENGTH, f"Label length error: {label}"
        
        img = Image.open(path).convert('RGB')
        img_tensor = self.transform(img)
        
        # 编码为固定长度的 tensor
        encoded = torch.tensor([CHAR2IDX[c] for c in label], dtype=torch.long)
        return img_tensor, encoded


class CNN(nn.Module):
    def __init__(self, vocab_size, dropout=0.5, fixed_length=6):
        super(CNN, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
 
        self.convlayer = nn.Sequential(
            # 如果预处理采用Grayscale 则 channel=1
            nn.Conv2d(3, 32, (3,3), stride=1, padding=1),
            # 激活函数，x小于0,y=0
            nn.ReLU(),
            nn.MaxPool2d((2,2), 2),
 
            nn.Conv2d(32, 64, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), 2),
 
            nn.Conv2d(64, 128, (3,3), stride=1, padding=1),
            nn.ReLU(),
 
            nn.Conv2d(128, 256, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1,2), 2),
 
            nn.Conv2d(256, 512, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
 
            nn.Conv2d(512, 512, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((1,2), 2),
 
            nn.Conv2d(512, 512, (2,2), stride=1, padding=0),
            self.dropout
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((fixed_length, 1))
        
        # 全连接分类层（每个字符位置独立预测）
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, vocab_size)
        )
 
    def forward(self, x):
        # CNN特征提取 [B, C, H, W]
        features = self.cnn(x)  
        
        # 自适应池化 [B, 512, fixed_length, 1]
        features = self.adaptive_pool(features)  
        
        # 调整维度 [B, fixed_length, 512]
        features = features.squeeze(-1).permute(0, 2, 1)  
        
        # 每个位置独立分类 [B, fixed_length, vocab_size]
        return torch.stack([self.fc(pos_feat) for pos_feat in features.unbind(1)], dim=1)


class OCR:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = CNN().to(self.device)
        print('Model loaded to ', self.device)

        self.critertion = nn.CrossEntropyLoss()

        self.idx2char = IDX2CHAR
        self.char2idx = CHAR2IDX

    def load_model(self, model_path='captcha.pth'):
        if os.path.exists(model_path):
            print(f"✅ 加载已有模型权重: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint.get('epoch', 0), checkpoint.get('optimizer_state_dict', None), checkpoint.get('losses', None)
        else:
            print("⚠️ 未找到模型权重文件，将从头开始训练")
            return 0, None, None

    def train_step(self, optimizer, images, labels):
        self.model.train()
        images = images.to(self.device)
        labels = labels.to(self.device)  # shape: [B, 6]

        logits = self.model(images)  # shape: [B, 6, 62]

        # 计算每个字符位置的独立损失
        loss = sum(self.criterion(logits[:,i], labels[:,i]) 
                  for i in range(CAPTCHA_LENGTH)) / CAPTCHA_LENGTH

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return logits, loss

    def val_step(self, images, labels):
        self.model.eval()
        images = images.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            logits = self.model(images)
            loss = 0
            for i in range(CAPTCHA_LENGTH):
                loss += self.critertion(logits[:, i, :], labels[:, i])
            loss /= CAPTCHA_LENGTH

        return logits, loss

    def predict(self, img):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(img.to(self.device))
            preds = logits.argmax(dim=2)  # shape: [B, 6]
        return preds

    def decode_batch(self, preds):
        return [''.join([IDX2CHAR[p.item()] for p in pred]) for pred in preds]

    def evaluate(self, val_loader):
        self.model.eval()
        correct_char = 0
        total_char = 0
        correct_seq = 0
        total_seq = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                preds = logits.argmax(dim=2)

                # 字符级准确率
                correct_char += (preds == labels).sum().item()
                total_char += preds.numel()

                # 序列级准确率
                for pred, label in zip(preds, labels):
                    if torch.equal(pred, label):
                        correct_seq += 1
                    total_seq += 1

        char_acc = correct_char / total_char
        seq_acc = correct_seq / total_seq

        return {
            'char_accuracy': char_acc,
            'sequence_accuracy': seq_acc
        }

    def train(self, num_epochs, optimizer, train_loader, val_loader, start_epoch=0, print_every=2):
        train_losses, valid_losses = [], []

        for epoch in range(start_epoch, num_epochs):
            tot_train_loss = 0
            self.model.train()

            for i, (images, labels) in enumerate(train_loader):
                _, loss = self.train_step(optimizer, images, labels)
                tot_train_loss += loss.item()

            with torch.no_grad():
                tot_val_loss = 0
                self.model.eval()

                for i, (images, labels) in enumerate(val_loader):
                    _, val_loss = self.val_step(images, labels)
                    tot_val_loss += val_loss.item()

                train_loss = tot_train_loss / len(train_loader)
                valid_loss = tot_val_loss / len(val_loader)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

            if epoch % print_every == 0 or epoch == num_epochs - 1:
                result = self.evaluate(val_loader)
                print(f'Epoch [{epoch+1}/{num_epochs}] | '
                      f'train loss {train_loss:.4f} | '
                      f'val loss {valid_loss:.4f} | '
                      f'Char Acc: {result["char_accuracy"]:.4f} | '
                      f'Seq Acc: {result["sequence_accuracy"]:.4f}')
                if print_every == 10:
                    # 可选：保存模型
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'losses': (train_losses, valid_losses)
                    }, 'captcha.pth')

        return train_losses, valid_losses
    def fine_tune(self, train_dir, val_dir, 
                 num_epochs=20, batch_size=8,
                 freeze_backbone=True, lr=0.001,
                 save_path='fine_tuned_model.pth'):
        """
        内置函数：对当前OCR实例进行微调
        
        参数:
            train_dir: 新训练集路径
            val_dir: 新验证集路径
            num_epochs: 微调轮数
            batch_size: 批大小
            freeze_backbone: 是否冻结卷积层
            lr: 学习率
            save_path: 模型保存路径
        """
        # 1. 冻结指定层
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'convlayer' in name:
                    param.requires_grad = False
            print("🔒 已冻结卷积层参数")

        # 2. 准备数据
        train_dataset = Dataset(train_dir)
        val_dataset = Dataset(val_dir)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4
        )

        # 3. 配置优化器（仅优化未冻结参数）
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, momentum=0.9, nesterov=True
        )

        # 4. 微调训练
        best_acc = 0
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            
            for images, labels in train_loader:
                _, loss = self.train_step(optimizer, images, labels)
                train_loss += loss.item()
            
            # 验证
            val_result = self.evaluate(val_loader)
            val_acc = val_result['sequence_accuracy']
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_acc
                }, save_path)
                print(f"🎯 保存最佳模型，验证集准确率: {val_acc:.4f}")
            
            # 打印进度
            print(f'Epoch [{epoch+1}/{num_epochs}] | '
                  f'Train Loss: {train_loss/len(train_loader):.4f} | '
                  f'Val Acc: {val_acc:.4f} | '
                  f'Best Acc: {best_acc:.4f}')
        
        print(f"微调完成，最佳验证准确率: {best_acc:.4f}")
        return best_acc

def main():
    # 实例化OCR模型
    ocr = OCR()
    # 分阶段微调（先冻结后解冻）
    phase1_acc = ocr.fine_tune(..., freeze_backbone=True, num_epochs=10)
    print(f"第一阶段微调准确率: {phase1_acc:.4f}")
    # 解冻部分层继续训练
    for name, param in ocr.model.named_parameters():
        if 'convlayer.6' in name:  # 解冻最后几个卷积层
            param.requires_grad = True
        
    phase2_acc = ocr.fine_tune(..., freeze_backbone=False, num_epochs=10)
    print(f"第二阶段微调准确率: {phase2_acc:.4f}")