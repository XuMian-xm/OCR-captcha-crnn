import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from ctc_model import OCR, Dataset
DIR_list = ['./data/train', './data/more', './data/archive']
TRAIN_DIR = DIR_list[2]
VAL_DIR = './data/val'
TEST_DIR = './data/test'
# batch_size lr 参数值训练，得到的结果较合适
BATCH_SIZE = 8
N_WORKERS = 0
EPOCHS = 20
 
CHARS ='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
VOCAB_SIZE = len(CHARS) + 1

lr = 0.02
# 权重衰减
weight_decay = 1e-5
# 下降幅度
momentum = 0.7
def test_view(ocr, name='val'):
    data_set = Dataset(VAL_DIR if name=='val' else TEST_DIR)
    data_loader = DataLoader(
        data_set, batch_size=BATCH_SIZE,
        num_workers=N_WORKERS, shuffle=False 
    )
    if ocr is None:
        ocr = OCR()
        start_epoch, opt_state, losses = ocr.load_model('ocr.pth')
        optimizer = optim.SGD(
            ocr.crnn.parameters(), lr =lr, nesterov=True,
            weight_decay=weight_decay, momentum=momentum
        )
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
    result = ocr.evaluate(data_loader)
    print("验证集结果：" if name=='val' else "测试集结果：")
    print(f"字符级准确率: {result['char_accuracy']:.4f}")
    print(f"序列级准确率: {result['sequence_accuracy']:.4f}")
    sample_result = []
    for i in range(10):
        idx = np.random.randint(len(data_set))
        img, label = data_set.__getitem__(idx)
        logits = ocr.predict(img.unsqueeze(0))
        pred_text = ocr.decode(logits.cpu())
        sample_result.append((img, label, pred_text))
    fig = plt.figure(figsize=(17,5))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        img, label, pred_text = sample_result[i]
        title = f'Truth: {label} | Pred: {pred_text}'
        ax.imshow(img.permute(1,2, 0))
        ax.set_title(title)
    plt.savefig('result.png')
    plt.show()

def test(ocr, name='test'):
    DIR = {'test': TEST_DIR, 'val': VAL_DIR, 'train': TRAIN_DIR}
    DIR = DIR.get(name, TEST_DIR)
    data_set = Dataset(DIR)
    data_loader = DataLoader(
        data_set, batch_size=BATCH_SIZE,
        num_workers=N_WORKERS, shuffle=False
    )
    if ocr is None:
        ocr = OCR()
        start_epoch, opt_state, losses = ocr.load_model('ocr.pth')
        optimizer = optim.SGD(
            ocr.crnn.parameters(), lr =lr, nesterov=True,
            weight_decay=weight_decay, momentum=momentum
        )
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
    correct = 0
    for i in range(len(data_set)):
        img, label = data_set.__getitem__(i)
        logits = ocr.predict(img.unsqueeze(0))
        pred_text = ocr.decode(logits.cpu())
        print(f"预测文本: {pred_text}, 真实文本: {label}")
        if pred_text == label:
            correct += 1
    print(f"Accuracy: {correct / len(data_set):.4f}")
    return correct / len(data_set)

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
    result = ocr.evaluate(train_loader)
    print("训练集结果：")
    print(f"字符级准确率: {result['char_accuracy']:.4f}")
    print(f"序列级准确率: {result['sequence_accuracy']:.4f}")
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': ocr.crnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': (train_losses, val_losses)
    }, 'ocr.pth')

    test_view(ocr,'val')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Valid Loss')
    plt.title('Loss stats')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()
if __name__ == '__main__':
    main()
    test(None, 'test')
    test(None, 'val')