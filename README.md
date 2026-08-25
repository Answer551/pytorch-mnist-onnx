# pytorch-mnist-onnx
使用 PyTorch 训练 MNIST 手写数字识别模型并导出 ONNX，打通训练到部署的基础链路。/PyTorch MNIST training with ONNX export. A minimal practice to understand the training-to-deployment pipeline.

当然可以！以下是一个适合放在 GitHub 仓库的 README 模板，内容清晰、结构完整，并包含了你这个项目的核心信息和使用说明。

```markdown
# MNIST 手写数字识别与 ONNX 导出

一个基于 PyTorch 的简单卷积神经网络（CNN）示例，用于识别 MNIST 手写数字，并将训练好的模型导出为 ONNX 格式，以便跨平台部署。

## ✨ 功能特点

- 使用 PyTorch 构建一个包含卷积层、池化层和全连接层的 CNN 模型。
- 在 MNIST 数据集上训练模型（默认 3 个 epoch）。
- 保存训练好的模型权重（`model_weights.pth`）。
- 导出 ONNX 模型（`model.onnx`），包含模型结构和参数。
- 验证 ONNX 模型的合法性。
- 对比 PyTorch 与 ONNX Runtime 的推理输出，确保导出精度无损。

## 🛠 环境要求

- Python 3.8+
- PyTorch 1.10+（推荐使用 CUDA 版本以加速训练）
- torchvision
- onnx
- onnxruntime
- numpy
- matplotlib（可选，用于可视化）

### 安装依赖

```bash
pip install torch torchvision onnx onnxruntime numpy
```

如果你有 NVIDIA GPU，建议安装 CUDA 版本的 PyTorch，参考 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取安装命令。

## 🚀 快速开始

1. 克隆仓库：

```bash
git clone: https://github.com/Answer551/pytorch-mnist-onnx
cd pytorch-mnist-onnx
```

2. 运行训练和导出脚本：

```bash
python train_export_onnx.py
```

脚本会自动下载 MNIST 数据集（首次运行），训练模型，保存权重，导出 ONNX，并进行验证。

3. 查看输出文件：

- `model_weights.pth`：PyTorch 模型权重。
- `model.onnx`：ONNX 模型文件。

控制台会输出类似以下信息：

```
✅ ONNX 模型验证通过
最大差异: 1.1920928955078125e-06
```

最大差异越小，说明 ONNX 导出越精确。

## 📁 文件结构

```
├── train_export_onnx.py   # 主脚本：训练、导出、验证
├── model_weights.pth      # 训练后的 PyTorch 权重（运行后生成）
├── model.onnx             # 导出的 ONNX 模型（运行后生成）
├── README.md
└── data/                  # MNIST 数据集（自动下载）
```

## 🧠 模型结构

```
ImageClassifierModel(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=256, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

输入尺寸：`[batch, 1, 28, 28]`（灰度图）  
输出尺寸：`[batch, 10]`（10 个类别的 logits）

## 📊 训练与验证

### 训练参数

- 批量大小：64
- 优化器：Adam（学习率 0.001）
- 损失函数：CrossEntropyLoss
- 训练轮数：3

### 模型准确率

在 MNIST 测试集上，该模型通常可以达到 **98% 以上** 的准确率（未在脚本中直接评估，可自行添加测试代码）。

### 导出 ONNX

ONNX 导出时使用动态 batch 维度，支持任意 batch size 的输入。

## 🔍 验证 ONNX 模型

脚本最后会对比 PyTorch 和 ONNX Runtime 的输出，最大差异通常在 `1e-6` 量级，证明导出无损。

你也可以使用 [Netron](https://netron.app/) 可视化 `model.onnx` 文件。

## 🖼 自定义测试

如果你想用自己的手写数字图片测试模型，可以参考以下步骤：

1. 准备一张 28×28 的灰度图片（或任意尺寸，代码会缩放）。
2. 使用以下代码进行推理：

```python
from PIL import Image
import torch
import onnxruntime
import numpy as np
from torchvision import transforms

# 加载图片并预处理
img = Image.open('my_digit.png').convert('L')
img = img.resize((28, 28))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img_tensor = transform(img).unsqueeze(0)  # 形状 [1,1,28,28]

# ONNX Runtime 推理
session = onnxruntime.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: img_tensor.numpy()})
pred = np.argmax(output[0])
print(f'预测数字: {pred}')
```

## ⚠️ 常见问题

### 1. 为什么 `torch.cuda.is_available()` 返回 `False`？

可能安装了 CPU 版本的 PyTorch，或者显卡驱动不兼容。请前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装 CUDA 版本，并确保 NVIDIA 驱动已正确安装。

### 2. 训练时间太长？

默认使用 CPU 训练，如果希望加速，请安装 CUDA 版 PyTorch 并使用 GPU。也可以适当减少 epoch 或增大 batch size。

### 3. 如何提高准确率？

- 增加训练轮数（但注意过拟合）。
- 调整学习率或优化器。
- 添加数据增强。
- 使用更深的网络。

## 📜 许可证

本项目采用 [MIT License](LICENSE)（如果你有的话）。你可以自由使用、修改和分发。

---

**如果觉得这个项目有帮助，请给个 ⭐ Star 支持一下！**
```
