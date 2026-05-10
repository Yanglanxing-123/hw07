import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ====================== 直接用现有结构，不复制文件 ======================
def get_class_name(path):
    filename = os.path.basename(path).lower()
    if "virus" in filename:
        return "VIRUS"
    elif "bacteria" in filename:
        return "BACTERIA"
    else:
        return "NORMAL"

# 超参数
img_size = 100
batch_size = 64
epochs = 5
device = torch.device("cpu")

# 数据加载（直接从原文件夹读取，不创建新目录）
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(root="./train", transform=transform)
test_data = datasets.ImageFolder(root="./test", transform=transform)

# 手动映射标签
def update_dataset_labels(dataset):
    new_classes = ["NORMAL", "BACTERIA", "VIRUS"]
    new_class_to_idx = {c: i for i, c in enumerate(new_classes)}
    new_samples = []
    for path, _ in dataset.samples:
        cls = get_class_name(path)
        new_samples.append((path, new_class_to_idx[cls]))
    dataset.samples = new_samples
    dataset.classes = new_classes
    dataset.class_to_idx = new_class_to_idx

update_dataset_labels(train_data)
update_dataset_labels(test_data)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

# 三分类模型
model = nn.Sequential(
    nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 23 * 23, 128), nn.ReLU(),
    nn.Linear(128, 3)
).to(device)

# 训练
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🚀 开始三分类训练...")
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} 完成")

# 测试
model.eval()
all_pred = []
all_true = []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        pred = model(x).argmax(dim=1).cpu()
        all_pred.extend(pred.numpy())
        all_true.extend(y.numpy())

# 输出结果
acc = sum(p == t for p, t in zip(all_pred, all_true)) / len(all_true)
print(f"\n🎉 三分类测试准确率 = {acc:.4f}")

# 保存混淆矩阵
os.makedirs("figures", exist_ok=True)
cm = confusion_matrix(all_true, all_pred)
labels = ["正常", "细菌", "病毒"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("三分类混淆矩阵")
plt.tight_layout()
plt.savefig("./figures/confusion_3class.png")
plt.close()

print("✅ 三分类混淆矩阵已保存到 figures/confusion_3class.png")