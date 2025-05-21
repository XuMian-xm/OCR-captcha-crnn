import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
 
import numpy as np
import pickle as pkl

class Dataset(Dataset):
    def __init__(self, img_dir):
        path_list = os.listdir(img_dir)
        # 获取文件夹绝对路径
        abspath = os.path.abspath(img_dir)
        self.img_list = [os.path.join(abspath, path) for path in path_list]
        self.transform = transforms.Compose([
            # 灰度化，配合 卷积网络初始通过 1
            # transforms.Grayscale(),
            transforms.Resize((80, 200)),  # 固定宽高比
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.img_list)
 
    def __getitem__(self, idx):
        path = self.img_list[idx]
        label = os.path.basename(path).split('.')[0]
        label = label.split('_')[0]
        img = Image.open(path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, label

class CRNN(nn.Module):
    def __init__(self, vocab_size, dropout=0.5):
        super(CRNN, self).__init__()
        
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
 
        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 256),
            # nn.Linear(1024, 256)
            self.dropout
        )
 
        self.lstm_0 = nn.GRU(256, 256, bidirectional=True)
        self.lstm_1 = nn.GRU(512, 256, bidirectional=True)
 
        self.out = nn.Sequential(
            nn.Linear(512, vocab_size),
        )
 
    def forward(self, x):
        x = self.convlayer(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        
        x = self.mapSeq(x)
 
        x, _ = self.lstm_0(x)
        x, _ = self.lstm_1(x)
 
        x = self.out(x)
 
        return x.permute(1, 0, 2)
 
class OCR:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
 
        self.crnn = CRNN(VOCAB_SIZE).to(self.device)
        print('Model loaded to ', self.device)
 
        self.critertion = nn.CTCLoss(blank=0)
 
        self.char2idx, self.idx2char = self.char_idx()
    def load_model(self, model_path='ocr.pth', weights_only=False):
        """加载模型权重和优化器状态"""
        if os.path.exists(model_path):
            print(f"✅ 加载已有模型权重: {model_path}") 
            # 根据weights_only参数决定如何加载模型
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if not weights_only:
                # 如果不是仅加载权重，则尝试加载完整的checkpoint信息
                try:
                    # 加载模型权重
                    self.crnn.load_state_dict(checkpoint['model_state_dict'])                  
                    # 返回epoch、optimizer状态和losses
                    return checkpoint.get('epoch', 0), checkpoint.get('optimizer_state_dict', None), checkpoint.get('losses', None)
                except KeyError as e:
                    print(f"警告: 在checkpoint中找不到预期的键: {e}. 可能是因为使用了weights_only=True或checkpoint格式不匹配.")
                    raise
            else:
                # 如果是仅加载权重，则只加载模型权重
                try:
                    if checkpoint.get('model_state_dict') is None:
                        self.crnn.load_state_dict(checkpoint)
                    else:
                        self.crnn.load_state_dict(checkpoint['model_state_dict'])
                except RuntimeError as e:
                    print(f"错误: 无法加载模型权重 - {e}. 确保weights_only设置正确且checkpoint包含正确的权重数据.")
                    raise              
                # 返回默认值，因为weights_only模式下不会加载其他状态
                return 0, None, None
        else:
            print("⚠️ 未找到模型权重文件，将从头开始训练")
            return 0, None, None
    def char_idx(self):
        char2idx = {}
        idx2char = {}
 
        characters = CHARS + '-'
        for i, char in enumerate(characters):
            char2idx[char] = i + 1
            idx2char[i+1] = char
        return char2idx, idx2char
    
    def encode(self, labels):
        length_per_label = [len(label) for label in labels] 
        joined_label = ''.join(labels)
 
        joined_encoding = []
        for char in joined_label:
            joined_encoding.append(self.char2idx[char])
 
        return (torch.IntTensor(joined_encoding), torch.IntTensor(length_per_label)) 
 
    def decode(self, logits):
        tokens = logits.softmax(2).argmax(2).squeeze(1)
 
        tokens = ''.join([self.idx2char[token]
                          if token !=0 else '-'
                          for token in tokens.numpy()])
        tokens = tokens.split('-')
 
        text = [char 
                for batch_token in tokens
                for idx, char in enumerate(batch_token)
                if char != batch_token[idx-1] or len(batch_token) == 1]    
        
        text = ''.join(text)  
 
        return text
 
    def calculate_loss(self, logits, labels):
        encoded_labels, labels_len = self.encode(labels)
        # 确保标签也被移到了正确的设备上，标签需要在CPU上进行计算
        
        logits_lens = torch.full(
            size=(logits.size(1),),
            fill_value = logits.size(0),
            dtype = torch.int32
        )
 
        return self.critertion(
            logits.log_softmax(2), encoded_labels,
            logits_lens, labels_len
        )
    
    def train_step(self, optimizer, images, labels):
        logits = self.predict(images)
 
        optimizer.zero_grad()
        loss = self.calculate_loss(logits, labels)
        loss.backward()
        optimizer.step()
 
        return logits, loss
    
    def val_step(self, images, labels):
        logits = self.predict(images)
        loss = self.calculate_loss(logits, labels)
 
        return logits, loss
    
    def predict(self, img):
        return self.crnn(img.to(self.device))
    
    def train(self, num_epochs, optimizer, train_loader, val_loader, start_epoch=0, print_every = 2):
        train_losses, valid_losses = [],[]
        if hasattr(self, 'losses'):
            train_losses, valid_losses = self.losses
        best_val_loss = float('inf')
        # early_stopping = EarlyStopping(patience=5, verbose=True)
        for epoch in range(start_epoch,num_epochs):
            tot_train_loss = 0
            self.crnn.train()
 
            for i, (images, labels) in enumerate(train_loader):
                logits, train_loss = self.train_step(optimizer, images, labels)
                tot_train_loss += train_loss.item()
 
            with torch.no_grad():
                tot_val_loss = 0
                self.crnn.eval()
 
                for i, (images, labels) in enumerate(val_loader):
                    logits, val_loss = self.val_step(images, labels)
 
                    tot_val_loss += val_loss.item()
                
                train_loss = tot_train_loss / len(train_loader.dataset)
                valid_loss = tot_val_loss / len(val_loader.dataset)
 
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
            if epoch % print_every == 0:
                print('Epoch [{:5d}/{:5d}] | train loss {:6.4f} | val loss {:6.4f}'.format(
                    epoch + 1, num_epochs, train_loss, valid_loss
                )) 
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.crnn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': (train_losses, valid_losses),
                    'best_val_loss': best_val_loss
                }, 'best_model.pth')
                print(f'✅ 保存最佳模型 (val_loss: {valid_loss:.4f})')
            '''
            early_stopping(valid_loss, self.crnn)
            if early_stopping.early_stop:
                print("⚠️ 早停触发!")
                break
            '''       
        return train_losses, valid_losses
    def evaluate(self, val_loader):
        """
        在验证集上评估模型表现，包括：
        - 每个字符的准确率（Char Accuracy）
        - 完整字符串匹配的准确率（Seq Accuracy）
        """
        self.crnn.eval()
        total_char = 0
        correct_char = 0
        total_seq = 0
        correct_seq = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                logits = self.predict(images)  # shape: [T, B, V]
                preds = self.decode_batch(logits.cpu())  # 解码为字符串列表

                for pred, label in zip(preds, labels):
                    # 字符级别比较
                    min_len = min(len(pred), len(label))
                    correct_char += sum(pred[i] == label[i] for i in range(min_len))
                    total_char += max(len(pred), len(label))  # 长度不一致也统计为错误

                # 序列级别比较
                    if pred == label:
                        correct_seq += 1
                    total_seq += 1

        char_acc = correct_char / total_char
        seq_acc = correct_seq / total_seq

        return {
            'char_accuracy': char_acc,
            'sequence_accuracy': seq_acc
        }
    def decode_batch(self, batch_logits):
        """
        对一批次的 logits 进行解码，返回字符串列表
        """
        batch_logits=torch.split(batch_logits, 1, dim=1)
        batch_texts = []
        for logits in batch_logits:
            text = self.decode(logits)  # 解码为字符串
            batch_texts.append(text)
        return batch_texts
TRAIN_DIR = './data/train'
VAL_DIR = './data/val'

# batch_size lr 参数值训练，得到的结果较合适
BATCH_SIZE = 8
N_WORKERS = 0
EPOCHS = 50
 
CHARS ='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
VOCAB_SIZE = len(CHARS) + 1

lr = 0.02
# 权重衰减
weight_decay = 1e-5
# 下降幅度
momentum = 0.7
 
def main():
    train_dataset = Dataset(TRAIN_DIR)
    val_dataset = Dataset(VAL_DIR)
 
    train_loader = DataLoader(
        train_dataset, batch_size = BATCH_SIZE,
        num_workers = N_WORKERS, shuffle=True
    )
 
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=N_WORKERS, shuffle=False
    )
 
    ocr = OCR()
    start_epoch, opt_state, losses = ocr.load_model('ocr.pth')
    optimizer = optim.SGD(
        ocr.crnn.parameters(), lr =lr, nesterov=True,
        weight_decay=weight_decay, momentum=momentum
    ) 
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)

# 如果有 loss 历史记录
    if losses is not None:
        train_losses, val_losses = losses
    else:
        train_losses, val_losses = [], []
    train_losses, val_losses = ocr.train(EPOCHS, optimizer, train_loader, val_loader, 
                                        start_epoch = start_epoch, print_every=1)
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': ocr.crnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': (train_losses, val_losses)
    }, 'ocr.pth')
