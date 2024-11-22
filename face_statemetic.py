import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 1. データセットのパスとパラメータ設定
data_dir = './data'  # データセットのディレクトリパス
batch_size = 32
num_classes = 8
input_size = (128, 128)
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. データセットの前処理
transform = {
    "train": transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

# 3. データセットの読み込み
datasets = {
    "train": datasets.ImageFolder(root=f"{data_dir}/train", transform=transform["train"]),
    "val": datasets.ImageFolder(root=f"{data_dir}/val", transform=transform["val"]),
}
dataloaders = {
    "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
    "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
}

# 4. VGGモデルのロードと転移学習の設定
model = models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False  # 既存の特徴抽出部分は学習しない

# クラス分類用の全結合層をカスタマイズ
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model = model.to(device)

# 5. 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 6. 学習と評価
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # 各エポックでの学習と評価
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # モデルのベストウェイトを保存
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model

# モデルの学習
model = train_model(model, dataloaders, criterion, optimizer, num_epochs)

# モデルの保存
torch.save(model.state_dict(), "vgg16_transfer_learning.pth")
print("Model saved!")
