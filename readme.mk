<span style="color:rgb(0,0,255)">By RuoChen from ZJU</span>

---
# 运行步骤
git clone https://github.com/FlagAlpha/Llama2-Chinese.git

---
cd models
git lfs clone https://huggingface.co/FlagAlpha/Atom-7B

---
conda create -n Llama2 python=3.9 -y
conda activate Llama2
pip install -r requirements.txt
### 测试原模型
python main.py
# 国内下载问题可以用网盘代替
## 
---
# 文件介绍

### main.py: 主程序入口，包含模型加载和推理逻辑。
### LoRA_train.py: LoRA训练脚本，用于微调模型。
### LoRA_val.py: LoRA验证脚本，用于评估微调后的模型性能。
### dataset_create.py: 数据集创建脚本，用于准备训练和验证数据。
### 运行dataset_create.py脚本后，会生成training_dataset.json

---

# <span style="color:red">Acknowledgment</span>
## <span style="color:blue">This project is not possible without the following codebases.：</span>
Llama中文社区
AtomEcho
https://github.com/LlamaFamily/Llama-Chinese