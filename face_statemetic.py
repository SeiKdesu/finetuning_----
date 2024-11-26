import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 1. データセットのパスとパラメータ設定
data_dir = './data'  # データセットのディレクトリパス
batch_size = 512
num_classes = 8
input_size = (128, 128)
num_epochs = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. データセットの前処理
transform = {
    "train": transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
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
# model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
# 全結合層 (classifier) にドロップアウトを追加
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),  # ドロップアウトを追加
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),  # ドロップアウトを追加
    nn.Linear(4096, num_classes),
    # nn.Softmax(dim=1)  # 出力層にSoftmaxを追加
)
model = model.to(device)
print(model)
# 5. 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.01, weight_decay=1e-4)  # L2正則化
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)  # 学習率スケジューリング

# 4. 学習と評価関数にスケジューラを組み込み
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

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

            if phase == "train":
                scheduler.step()  # 学習率スケジューリング
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()



    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history

# 修正した学習関数を実行
model, history = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs)

# 7. 学習履歴の可視化
def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # 損失のグラフ
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # 正解率のグラフ
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_acc.png')

# 学習履歴のプロット
plot_history(history)


# モデルの学習
# model = train_model(model, dataloaders, criterion, optimizer, num_epochs)

# # モデルの保存
# torch.save(model.state_dict(), "vgg16_transfer_learning.pth")
# print("Model saved!")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 混同行列を計算する関数
def evaluate_model(model, dataloader, class_names):
    model.eval()  # 評価モードに設定
    all_labels = []
    all_preds = []

    # 全てのデータを予測
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 混同行列を計算
    cm = confusion_matrix(all_labels, all_preds)
    
    # クラスごとの正解率を計算
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("Class-wise Accuracy:")
    for i, acc in enumerate(class_accuracy):
        print(f"Class {class_names[i]}: {acc:.2f}")
    
    return cm, class_accuracy

# 混同行列をプロットする関数
def plot_confusion_matrix(cm, class_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.title("Confusion Matrix")
    plt.savefig('matrix.png')

# クラス名を取得 (ImageFolderのクラス順に対応)
class_names = datasets["train"].classes

# 検証データで混同行列を生成
cm, class_accuracy = evaluate_model(model, dataloaders["val"], class_names)

# 混同行列をプロット
plot_confusion_matrix(cm, class_names)