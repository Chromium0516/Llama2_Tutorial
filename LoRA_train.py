import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    GenerationConfig  # 添加GenerationConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import os
import logging
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", message="The attention mask is not set and cannot be inferred")
warnings.filterwarnings("ignore", message="The `seen_tokens` attribute is deprecated")

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 1. 配置参数 ===
MODEL_PATH = './models/FlagAlpha/Atom-7B'
OUTPUT_DIR = './atom7b-lora-finetuned'
DATASET_PATH = 'training_dataset.json'

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 2. 加载模型和tokenizer ===
logger.info("正在加载模型和tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    load_in_8bit=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False
)

# 设置特殊token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

logger.info(f"词汇表大小: {len(tokenizer)}")
logger.info(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
logger.info(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

# === 3. 配置LoRA参数 ===
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

# === 4. 应用LoRA配置到模型 ===
logger.info("正在应用LoRA配置...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === 5. 改进的数据预处理函数 ===
def preprocess_function(examples):
    """将指令-响应对转换为模型训练所需的格式"""
    texts = []
    for i in range(len(examples["instruction"])):
        text = f"<s>Human: {examples['instruction'][i]}</s><s>Assistant: {examples['output'][i]}</s>"
        texts.append(text)
    
    # 显式返回attention_mask
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=384,
        return_tensors=None,
        add_special_tokens=False,
        return_attention_mask=True  # 确保返回attention_mask
    )
    
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized

# === 6. 使用内置数据整理器 ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# === 7. 加载和预处理数据集 ===
logger.info("正在加载数据集...")
def load_dataset_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_dict({
            "instruction": [item["instruction"] for item in data],
            "output": [item["output"] for item in data]
        })
    except Exception as e:
        logger.error(f"加载数据集时出错: {e}")
        return None

train_dataset = load_dataset_from_json(DATASET_PATH)

if train_dataset is None:
    logger.warning("使用示例数据集")
    sample_data = [
        {"instruction": "你好", "output": "你好！我是Atom-7B助手，很高兴为您服务！"},
        {"instruction": "介绍一下Python", "output": "Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。它广泛应用于Web开发、数据科学、人工智能等领域。"},
        {"instruction": "什么是机器学习？", "output": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习规律，无需明确编程。"},
        {"instruction": "如何学习编程？", "output": "学习编程建议从基础语法开始，选择合适的语言，通过实际项目练习，持续学习新技术。"},
        {"instruction": "解释深度学习", "output": "深度学习是机器学习的子领域，使用多层神经网络模拟人脑工作方式，在图像识别、语音识别等任务上表现出色。"},
        {"instruction": "什么是LoRA？", "output": "LoRA（Low-Rank Adaptation）是一种参数高效的微调方法，通过低秩矩阵分解来减少需要训练的参数数量。"},
        {"instruction": "Python有什么优势？", "output": "Python的优势包括：语法简洁易读、丰富的第三方库、强大的社区支持、跨平台兼容性好、适合快速原型开发。"},
        {"instruction": "如何提高代码质量？", "output": "提高代码质量的方法包括：遵循编码规范、编写清晰的注释、进行代码审查、编写单元测试、重构优化代码结构。"}
    ]
    train_dataset = Dataset.from_dict({
        "instruction": [item["instruction"] for item in sample_data],
        "output": [item["output"] for item in sample_data]
    })

logger.info(f"原始数据集大小: {len(train_dataset)}")

logger.info("正在预处理数据集...")
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="预处理数据"
)

logger.info(f"预处理后数据集大小: {len(train_dataset)}")

# === 8. 训练参数配置 ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    remove_unused_columns=True,
    dataloader_pin_memory=False,
    fp16=True,
    warmup_ratio=0.05,
    weight_decay=0.01,
    dataloader_num_workers=0,
    group_by_length=True,
    report_to="none",
    optim="paged_adamw_8bit",
)

# === 9. 创建训练器 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# === 10. 开始训练 ===
logger.info("开始LoRA微调训练...")
try:
    trainer.train()
    logger.info("训练完成！")
except Exception as e:
    logger.error(f"训练过程中出现错误: {e}")
    logger.info("建议: 尝试减小batch_size或max_length")

# === 11. 保存微调后的模型 ===
logger.info("正在保存微调后的模型...")
try:
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"模型保存成功: {OUTPUT_DIR}")
except Exception as e:
    logger.error(f"保存模型时出错: {e}")

# === 12. 改进的测试方法 ===
logger.info("\n快速测试微调后的模型:")
model.eval()

# 使用新的生成配置方法
generation_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    # 解决缓存问题
    use_cache=False
)

test_prompt = "<s>Human: 你好</s><s>Assistant: "
input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids

if hasattr(model, 'device'):
    device = model.device
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_ids = input_ids.to(device)

try:
    with torch.no_grad():
        # 使用新的生成方法
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant: " in response:
        assistant_response = response.split("Assistant: ")[-1]
    else:
        # 尝试从最后截取助手回复
        start_index = response.find("Assistant: ")
        if start_index != -1:
            assistant_response = response[start_index + len("Assistant: "):]
        else:
            assistant_response = response
    
    logger.info(f"模型回答: {assistant_response}")
except Exception as e:
    logger.error(f"测试时出错: {e}")
    # 尝试更简单的生成方法
    try:
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            predicted_token_ids = logits.argmax(-1)
            response = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
            logger.info(f"备选测试结果: {response}")
    except Exception as e2:
        logger.error(f"备选测试也失败: {e2}")

logger.info("LoRA微调流程完成！")