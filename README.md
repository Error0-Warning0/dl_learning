# dl-learning

我的深度学习学习记录，使用 PyTorch。每个主题一个文件夹，按学习顺序编号，各自可以单独运行。

**长期目标：** 把超分辨率用在遥感图像的小目标检测上，能看懂并改进 RT-DETR、Super-DETR 这类模型。这个仓库只放打基础的练习，研究项目单独放在别的仓库。

## 学习进度

| 编号 | 主题 | 内容 | 状态 | 结果 |
|---|---|---|---|---|
| 01 | [MNIST 手写数字识别](01_mnist/) | 全连接网络（MLP）和 CNN | 已完成 | MLP 97.96%，CNN 99.20% |
| 02 | [CNN 进阶（CIFAR-10）](02_cifar10/) | 数据增强、BatchNorm、Dropout，自己写 ResNet，学习率调度、标签平滑 | 已完成 | 基线 72.17%，数据增强 78.19%，ResNet-56 + 余弦退火 + 标签平滑 89.08% |
| 03 | 超分辨率 | SRCNN、ESPCN、EDSR，PixelShuffle，MSE 和 SSIM 损失，PSNR 和 SSIM 指标 | 计划中 | |
| 04 | 目标检测基础 | 边界框、IoU、NMS、COCO AP、FPN、Faster R-CNN、FCOS | 计划中 | |
| 05 | Transformer | 注意力、多头注意力、位置编码；RNN/LSTM 只了解思路 | 计划中 | |
| 06 | DETR 和 RT-DETR | Object Query、匈牙利匹配、集合损失，写精简版 DETR，再读 RT-DETR | 计划中 | |
| 07 | 超分辨率 + 小目标检测 | AI-TOD 数据集、小目标指标、频域融合、联合训练，读懂 Super-DETR | 计划中 | |

每个主题的运行方式和详细结果，见各自文件夹里的 README。

## 环境

所有主题共用一个虚拟环境，依赖版本见 [requirements.txt](requirements.txt)。我用的是 Python 3.14。

```bash
python -m venv ~/venvs/pytorch
source ~/venvs/pytorch/bin/activate
pip install -r requirements.txt
```

检查环境是否装好：

```bash
python 01_mnist/check_env.py
```

## 仓库约定

- **各主题独立：** 每个文件夹都有自己的配置和训练代码，改一个主题不会影响其他主题。
- **不提交的内容：** 数据集、训练好的权重和生成的图片都不进仓库，规则写在 [.gitignore](.gitignore) 里，对所有子文件夹生效。
