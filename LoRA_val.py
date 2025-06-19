import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# 配置路径
BASE_MODEL_PATH = 'D:/Llama2/models/FlagAlpha/Atom-7B'
LORA_MODEL_PATH = './atom7b-lora-finetuned'

def load_model_simple():
    """简化的模型加载"""
    print("正在加载模型...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    try:
        # 尝试加载GPU版本
        print("尝试GPU加载...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            load_in_8bit=True
        )
        
        # 加载LoRA权重
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
        device = "cuda"
        
    except Exception as e:
        print(f"GPU加载失败: {e}")
        print("切换到CPU加载...")
        
        # CPU版本
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True
        )
        
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
        device = "cpu"
    
    model.eval()
    print(f"模型加载完成，运行在: {device}")
    return model, tokenizer, device

def generate_response(model, tokenizer, device, prompt, max_new_tokens=100):
    """生成回复"""
    try:
        # 编码输入
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # 移动到设备
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成输出
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False  # 避免缓存问题
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取新生成的部分
        if "Assistant:" in response:
            generated_text = response.split("Assistant:")[-1].strip()
        else:
            # 移除原始prompt
            generated_text = response[len(prompt):].strip()
            
        return generated_text
        
    except Exception as e:
        return f"生成错误: {e}"

def chat_loop(model, tokenizer, device):
    """对话循环"""
    print("\n=== LoRA微调后的Atom-7B对话模式 ===")
    print("输入 'exit' 退出对话\n")
    
    while True:
        try:
            user_input = input("[用户]: ").strip()
            
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("再见！")
                break
            
            if not user_input:
                print("请输入有效问题")
                continue
            
            # 构造prompt
            prompt = f"Human: {user_input}\nAssistant:"
            
            # 生成回复
            print("[Atom-7B]: ", end="", flush=True)
            response = generate_response(model, tokenizer, device, prompt, max_new_tokens=150)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n对话被中断，再见！")
            break
        except Exception as e:
            print(f"\n对话过程中出错: {e}")
            continue

def test_model(model, tokenizer, device):
    """测试模型"""
    print("\n=== 模型测试 ===")
    
    test_questions = [
        "你好",
        "Python是什么？",
        "如何学习编程？"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        prompt = f"Human: {question}\nAssistant:"
        response = generate_response(model, tokenizer, device, prompt, max_new_tokens=100)
        print(f"回答: {response}")
        print("-" * 50)

def main():
    """主函数"""
    try:
        # 加载模型
        model, tokenizer, device = load_model_simple()
        
        # 选择模式
        print("\n请选择模式:")
        print("1. 对话模式")
        print("2. 测试模式")
        print("3. 两者都运行")
        
        choice = input("\n请输入选择 (1/2/3): ").strip()
        
        if choice == "1":
            chat_loop(model, tokenizer, device)
        elif choice == "2":
            test_model(model, tokenizer, device)
        elif choice == "3":
            test_model(model, tokenizer, device)
            chat_loop(model, tokenizer, device)
        else:
            print("无效选择，默认进入对话模式")
            chat_loop(model, tokenizer, device)
            
    except Exception as e:
        print(f"程序运行错误: {e}")
        print("\n可能的解决方案:")
        print("1. 检查模型路径是否正确")
        print("2. 确保LoRA训练已完成")
        print("3. 检查依赖库版本")
        print("4. 尝试重新安装transformers和peft")

if __name__ == "__main__":
    main()