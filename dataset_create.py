import json
import pandas as pd
from typing import List, Dict

def create_sample_dataset():
    """创建示例数据集"""
    sample_data = [
        {
            "instruction": "你好",
            "output": "你好！我是Atom-7B助手，很高兴为您服务！有什么可以帮助您的吗？"
        },
        {
            "instruction": "介绍一下Python编程语言",
            "output": "Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。它以简洁、易读的语法著称，广泛应用于Web开发、数据科学、人工智能、自动化脚本等领域。Python的设计哲学强调代码的可读性和简洁性。"
        },
        {
            "instruction": "什么是机器学习？",
            "output": "机器学习是人工智能的一个分支，它使用算法来让计算机系统能够从数据中自动学习和改进，而无需明确编程。主要包括监督学习、无监督学习和强化学习三种类型。机器学习广泛应用于图像识别、自然语言处理、推荐系统等领域。"
        },
        {
            "instruction": "如何学习编程？",
            "output": "学习编程建议按以下步骤：1. 选择一门适合初学者的语言（如Python）；2. 学习基础语法和概念；3. 通过实际项目练习；4. 阅读他人代码，学习最佳实践；5. 参与开源项目；6. 持续学习新技术。重要的是要保持耐心和持续练习。"
        },
        {
            "instruction": "解释一下深度学习",
            "output": "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的工作方式。它能够自动学习数据的特征表示，在图像识别、语音识别、自然语言处理等任务上取得了突破性进展。深度学习的核心是神经网络，通过反向传播算法进行训练。"
        }
    ]
    
    return sample_data

def save_dataset_to_json(data: List[Dict], filename: str):
    """保存数据集到JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据集已保存到: {filename}")

def load_dataset_from_csv(csv_file: str, instruction_col: str, output_col: str):
    """从CSV文件加载数据集"""
    df = pd.read_csv(csv_file)
    data = []
    for _, row in df.iterrows():
        data.append({
            "instruction": str(row[instruction_col]),
            "output": str(row[output_col])
        })
    return data

def convert_alpaca_format(alpaca_data: List[Dict]):
    """转换Alpaca格式数据"""
    converted_data = []
    for item in alpaca_data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if input_text:
            full_instruction = f"{instruction}\n{input_text}"
        else:
            full_instruction = instruction
            
        converted_data.append({
            "instruction": full_instruction,
            "output": output
        })
    
    return converted_data

def create_conversation_dataset():
    """创建对话格式数据集"""
    conversations = [
        {
            "instruction": "请介绍一下自己",
            "output": "我是Atom-7B，一个基于大语言模型的AI助手。我可以帮助您解答问题、提供建议、协助写作等。我会尽力为您提供准确、有用的信息。"
        },
        {
            "instruction": "你能做什么？",
            "output": "我可以帮助您：1. 回答各种知识性问题；2. 协助写作和编辑；3. 解释复杂概念；4. 提供学习建议；5. 帮助解决问题；6. 进行友好对话。我会根据您的需求提供个性化的帮助。"
        },
        {
            "instruction": "解释一下什么是人工智能",
            "output": "人工智能（AI）是指让机器具备类似人类智能的能力，包括学习、推理、感知、理解和决策等。AI技术包括机器学习、深度学习、自然语言处理、计算机视觉等。现代AI广泛应用于搜索引擎、推荐系统、自动驾驶、医疗诊断等领域，正在改变我们的生活和工作方式。"
        },
        {
            "instruction": "如何提高工作效率？",
            "output": "提高工作效率的方法包括：1. 制定清晰的目标和优先级；2. 使用时间管理技巧（如番茄工作法）；3. 减少干扰和多任务处理；4. 定期休息和保持健康；5. 学习使用效率工具；6. 持续学习新技能；7. 建立良好的工作习惯。关键是找到适合自己的方法并坚持执行。"
        },
        {
            "instruction": "Python和Java有什么区别？",
            "output": "Python和Java的主要区别：1. 语法：Python更简洁易读，Java更严格规范；2. 类型系统：Python是动态类型，Java是静态类型；3. 编译：Python是解释型语言，Java需要编译；4. 性能：Java通常运行更快，Python开发效率更高；5. 应用领域：Python多用于数据科学、AI，Java多用于企业级开发；6. 学习曲线：Python更容易入门。选择哪个取决于具体需求和项目特点。"
        }
    ]
    
    return conversations

def validate_dataset(data: List[Dict]):
    """验证数据集格式"""
    required_keys = ["instruction", "output"]
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"错误：第{i+1}条数据不是字典格式")
            return False
            
        for key in required_keys:
            if key not in item:
                print(f"错误：第{i+1}条数据缺少'{key}'字段")
                return False
                
            if not isinstance(item[key], str) or len(item[key].strip()) == 0:
                print(f"错误：第{i+1}条数据的'{key}'字段为空或不是字符串")
                return False
    
    print(f"数据集验证通过，共{len(data)}条有效数据")
    return True

def prepare_training_dataset():
    """准备训练数据集"""
    print("正在准备训练数据集...")
    
    # 创建基础数据
    sample_data = create_sample_dataset()
    conversation_data = create_conversation_dataset()
    
    # 合并数据
    all_data = sample_data + conversation_data
    
    # 验证数据
    if validate_dataset(all_data):
        # 保存数据集
        save_dataset_to_json(all_data, "training_dataset.json")
        
        # 显示数据集信息
        print(f"\n数据集统计:")
        print(f"总数据量: {len(all_data)}")
        print(f"平均指令长度: {sum(len(item['instruction']) for item in all_data) / len(all_data):.1f} 字符")
        print(f"平均回答长度: {sum(len(item['output']) for item in all_data) / len(all_data):.1f} 字符")
        
        # 显示示例数据
        print(f"\n示例数据:")
        for i, item in enumerate(all_data[:2]):
            print(f"样本 {i+1}:")
            print(f"指令: {item['instruction']}")
            print(f"回答: {item['output'][:100]}..." if len(item['output']) > 100 else f"回答: {item['output']}")
            print("-" * 50)
    
    return all_data

if __name__ == "__main__":
    # 准备训练数据集
    dataset = prepare_training_dataset()
    
    print("\n数据集准备完成！")
    print("现在你可以使用 'training_dataset.json' 文件进行LoRA微调。")
    print("如果需要添加更多数据，请编辑 training_dataset.json 文件。")