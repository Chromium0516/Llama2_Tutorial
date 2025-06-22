# 🦙 Llama2-Chinese Fine-tuning Framework

> **By RuoChen from ZJU**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Model-Atom--7B-orange.svg" alt="Model">
  <img src="https://img.shields.io/badge/Framework-LoRA-green.svg" alt="LoRA">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## 📋 目录

- [快速开始](#-快速开始)
- [环境配置](#-环境配置)
- [模型下载](#-模型下载)
- [文件说明](#-文件说明)
- [使用指南](#-使用指南)
- [致谢](#-致谢)

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/FlagAlpha/Llama2-Chinese.git
```

### 2. 下载模型

```bash
cd models
git lfs clone https://huggingface.co/FlagAlpha/Atom-7B
```

### 3. 配置环境

```bash
conda create -n Llama2 python=3.9 -y
conda activate Llama2
pip install -r requirements.txt
```

### 4. 测试模型

```bash
python main.py
```

---

## 🛠 环境配置

### 系统要求
- Python 3.9+
- CUDA 11.0+ (推荐)

### 依赖安装

```bash
# 创建虚拟环境
conda create -n Llama2 python=3.9 -y
conda activate Llama2

# 安装依赖包
pip install -r requirements.txt
```

---

## 📥 模型下载

### 官方下载

在 `models` 目录下执行：

```bash
git lfs clone https://huggingface.co/FlagAlpha/Atom-7B
```

### 💡 国内下载方案

如遇下载问题，可使用网盘代替：

> **通过网盘分享的文件**：Llama2必要文件  
> 🔗 链接: https://pan.baidu.com/s/12w8SQwPTlK9jbJy-pUDIjQ?pwd=1234  
> 🔑 提取码: 1234

---

## 📂 文件说明

| 文件名 | 功能描述 |
|--------|---------|
| `main.py` | 主程序入口，包含模型加载和推理逻辑 |
| `LoRA_train.py` | LoRA训练脚本，用于微调模型 |
| `LoRA_val.py` | LoRA验证脚本，用于评估微调后的模型性能 |
| `dataset_create.py` | 数据集创建脚本，用于准备训练和验证数据 |

> 💡 **提示**：运行 `dataset_create.py` 脚本后，会生成 `training_dataset.json`

---

## 🎯 使用指南

### 数据准备

```bash
# 创建训练数据集
python dataset_create.py
# 生成 training_dataset.json
```

### 模型训练

```bash
# 使用LoRA进行模型微调
python LoRA_train.py
```

### 模型验证

```bash
# 评估微调后的模型
python LoRA_val.py
```

### 模型推理

```bash
# 运行主程序进行推理
python main.py
```

---

## 📁 项目结构

```
.
├── models/
│   └── Atom-7B/              # Atom-7B模型文件
├── main.py                   # 主程序入口
├── LoRA_train.py            # LoRA训练脚本
├── LoRA_val.py              # LoRA验证脚本
├── dataset_create.py        # 数据集创建脚本
├── training_dataset.json    # 生成的训练数据集
└── requirements.txt         # 项目依赖
```

---

## 🙏 致谢

<p align="center">
  <strong>This project would not be possible without the following codebases:</strong>
</p>

<p align="center">
  <a href="https://github.com/LlamaFamily/Llama-Chinese">Llama中文社区</a> • 
  <a href="https://github.com/FlagAlpha">AtomEcho</a> • 
  <a href="https://github.com/LlamaFamily/Llama-Chinese">Llama-Chinese</a>
</p>

---

<p align="center">
  <i>如有问题，欢迎提交 Issue 或 PR！</i>
</p>
