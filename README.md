# dl-learning

我的深度学习学习记录，使用 PyTorch。每个主题一个文件夹，按学习顺序编号，各自可以单独运行。

## 学习进度

| 编号 | 主题 | 内容 | 状态 | 结果 |
|---|---|---|---|---|
| 01 | [MNIST 手写数字识别](01_mnist/) | 全连接网络（MLP），接下来加 CNN | 进行中 | MLP 测试准确率 97.96% |
| 02 | CIFAR-10 | 彩色图片、更深的 CNN、数据增强、BatchNorm | 计划中 | |
| 03 | RNN / LSTM | 序列数据：情感分类或时间序列预测 | 计划中 | |
| 04 | Seq2Seq + Attention | 注意力机制 | 计划中 | |
| 05 | Transformer | 自注意力、位置编码、字符级文本生成 | 计划中 | |

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
