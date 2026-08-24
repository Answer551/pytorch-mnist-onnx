import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import onnx
import onnxruntime
import numpy as np

# ---------- 1. 定义模型 ----------
class ImageClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 修正：经过两次卷积+池化后，特征图尺寸为 4x4，因此输入特征数为 16*4*4 = 256
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------- 2. 训练 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = ImageClassifierModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True
)

model.train()
for epoch in range(3):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

# ---------- 3. 保存权重 ----------
torch.save(model.state_dict(), 'model_weights.pth')

# ---------- 4. 导出 ONNX ----------
model.eval()
dummy_input = torch.randn(1, 1, 28, 28).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=18,              # 使用较新版本
    do_constant_folding=True,
    input_names=['input'],         # ONNX 输入名称（可以自定义）
    output_names=['output'],       # ONNX 输出名称
    dynamic_shapes={'x': {0: 'batch_size'}}   # 关键：使用模型参数名 'x'
)

# ---------- 5. 验证 ONNX 模型 ----------
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型验证通过")

# ---------- 6. ONNX Runtime 推理验证 ----------
ort_session = onnxruntime.InferenceSession("model.onnx")
test_input_np = np.random.randn(1, 1, 28, 28).astype(np.float32)
ort_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: test_input_np})

with torch.no_grad():
    pt_input = torch.tensor(test_input_np).to(device)
    pt_output = model(pt_input)

print(f"最大差异: {np.max(np.abs(pt_output.cpu().numpy() - ort_outputs[0]))}")