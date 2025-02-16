# 525-DeepThinkers

CDS525-Handwriting Recognition Project

项目简介
本项目旨在实现一个基于深度学习的手写识别系统，使用KMNIST数据集进行训练和测试。KMNIST是一个包含日语假名字符的手写图像数据集，类似于经典的MNIST数据集。通过该项目，我们将探索卷积神经网络（CNN）在图像分类任务中的应用，并优化模型以实现高效准确的手写字符识别。

项目目标
• 使用KMNIST数据集完成手写字符的分类任务。
• 实现一个高效的卷积神经网络模型。
• 通过调整超参数、优化器和损失函数，提升模型性能。
• 可视化训练过程和测试结果，展示模型的准确性和泛化能力。

数据集
• 数据集来源：[KMNIST Dataset]()
KMNIST dataset: https://github.com/rois-codh/kmnist 
• 数据集描述：
• 包含70,000张28x28灰度图像，分为10个类别。
• 训练集：60,000张图像。
• 测试集：10,000张图像。
• 数据集已通过PyTorch的`torchvision.datasets`模块集成，方便直接加载和使用。

项目结构
```
Handwriting_Recognition_Project/
├── data/
│   ├── train/
│   ├── test/
├── src/
│   ├── model.py               # CNN模型定义
│   ├── train.py               # 训练脚本
│   ├── evaluate.py            # 测试和评估脚本
│   ├── visualize.py           # 性能可视化脚本
├── results/
│   ├── plots/                 # 训练和测试性能图表
│   ├── predictions/           # 测试集预测结果
├── README.md                  # 项目说明文档
├── requirements.txt           # 项目依赖文件
```

环境依赖
• Python 3.8+
• PyTorch 1.9+
• torchvision
• matplotlib
• numpy

安装依赖：
```bash
pip install -r requirements.txt
```

运行指南
1.数据集加载
确保安装了`torchvision`，并使用以下代码加载KMNIST数据集：
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
```

2.模型训练
运行`train.py`脚本开始训练模型：
```bash
python src/train.py
```

训练过程中的损失和准确率将保存在`results/plots`目录中。

3.模型评估
运行`evaluate.py`脚本评估模型性能：
```bash
python src/evaluate.py
```

测试集的预测结果将保存在`results/predictions`目录中。

4.性能可视化
运行`visualize.py`脚本生成性能可视化图表：
```bash
python src/visualize.py
```

图表将展示训练损失、训练准确率和测试准确率随训练轮数的变化。

模型设计
• 模型架构：使用卷积神经网络（CNN）作为基础模型。
• 损失函数：交叉熵损失（Cross-Entropy Loss）。
• 优化器：Adam优化器。
• 超参数：
• 学习率：0.001
• 批量大小：64
• 训练轮数：20

性能指标
• 训练准确率：预期达到95%以上。
• 测试准确率：预期达到90%以上。
• 训练损失：随着训练轮数逐渐降低。

项目成果
• 训练和测试性能图表：展示模型在不同训练轮数下的性能变化。
• 测试集预测结果：展示前100个测试样本的预测标签、输入图像和真实标签的对比。

联系方式
如有任何问题或建议，请通过以下方式联系：
• Email:lixinlin@ln.hk
• GitHub:3094973309@qq.com

参考文献
• [KMNIST Dataset]()
• [PyTorch Documentation]()
• [Torchvision Datasets]()
