import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 指定本地模型路径
model_path = './models/FlagAlpha/Atom-7B'  # 替换为你的路径

# 2. 从本地加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False
)

# 修复警告的关键步骤：显式设置填充标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

# 3. 多轮对话循环
print("Atom-7B 本地对话模式已启动，输入 'exit' 退出对话。")

while True:
    user_input = input("\n[用户]: ")
    if user_input.lower() == 'exit':
        break
    
    # 构造 prompt（可自定义格式）
    prompt = f"<s>Human: {user_input}\n</s><s>Assistant: "
    
    # 编码输入 - 显式返回attention_mask
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=False,
        padding=True  # 确保返回attention_mask
    ).to('cuda')
    
    # 生成回答
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,  # 自动包含input_ids和attention_mask
            max_new_tokens=512,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id  # 使用设置的pad_token
        )
    
    # 解码并提取助手回答
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 更安全的提取方式
    if "Assistant: " in full_response:
        assistant_response = full_response.split("Assistant: ")[-1]
    else:
        assistant_response = full_response
    
    print(f"[Atom-7B]: {assistant_response}")