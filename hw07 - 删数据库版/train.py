import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ====================== 1. 自动找路径，不用你改 ======================
print("当前脚本所在目录：", os.getcwd())

# 直接找当前目录下的 train/test 文件夹
train_dir = os.path.join(os.getcwd(), "train")
test_dir = os.path.join(os.getcwd(), "test")

print("训练集路径：", train_dir)
print("测试集路径：", test_dir)

# 检查文件夹是否存在
if not os.path.exists(train_dir):
    print("❌ 错误：找不到 train 文件夹！")
    exit()
if not os.path.exists(test_dir):
    print("❌ 错误：找不到 test 文件夹！")
    exit()

print("✅ 文件夹检查通过，开始加载数据...")

# ====================== 数据集类 ======================
class XrayDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ====================== 数据加载 ======================
img_size = 150
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 收集所有图片路径和标签
def load_data(data_dir):
    paths = []
    labels = []
    # 0:NORMAL  1:PNEUMONIA
    for label_idx, cls in enumerate(["NORMAL", "PNEUMONIA"]):
        cls_path = os.path.join(data_dir, cls)
        print("正在读取：", cls_path)
        if not os.path.exists(cls_path):
            print(f"❌ 错误：找不到 {cls_path} 文件夹！")
            exit()
        for img_name in os.listdir(cls_path):
            if img_name.endswith((".jpg", ".png", ".jpeg")):
                paths.append(os.path.join(cls_path, img_name))
                labels.append(label_idx)
    return paths, labels

print("加载训练集...")
train_paths, train_labels = load_data(train_dir)
print("加载测试集...")
test_paths, test_labels = load_data(test_dir)

# 划分训练集和验证集（8:2）
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

print(f"✅ 数据加载完成！训练集：{len(train_paths)} 张，验证集：{len(val_paths)} 张，测试集：{len(test_paths)} 张")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

train_dataset = XrayDataset(train_paths, train_labels, transform)
val_dataset = XrayDataset(val_paths, val_labels, transform)
test_dataset = XrayDataset(test_paths, test_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ====================== CNN模型 ======================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*(img_size//8)*(img_size//8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,1)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return torch.sigmoid(x)

model = CNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ====================== 训练 ======================
epochs = 10
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

print("🚀 开始训练...")
for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = correct / total

    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss:{avg_train_loss:.4f} Acc:{train_acc:.4f} "
          f"Val Loss:{avg_val_loss:.4f} Acc:{val_acc:.4f}")

# ====================== 画曲线 ======================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

# 确保 figures 文件夹存在
if not os.path.exists("./figures"):
    os.mkdir("./figures")

plt.tight_layout()
plt.savefig("./figures/loss_acc.png")
plt.show()

# ====================== 测试集评估 ======================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs).squeeze()
        preds = (outputs > 0.5).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
pre = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("\n===== 测试集结果 =====")
print(f"准确率 Accuracy: {acc:.4f}")
print(f"精确率 Precision: {pre:.4f}")
print(f"召回率 Recall: {rec:.4f}")
print(f"F1分数: {f1:.4f}")

# ====================== 混淆矩阵 ======================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["正常","肺炎"],
            yticklabels=["正常","肺炎"])
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("混淆矩阵")
plt.savefig("./figures/confusion.png")
plt.show()

print("\n✅ 所有步骤完成！图片已保存到 ./figures 文件夹")