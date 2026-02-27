import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os


def load_and_test_model():
    # 1. 配置模型路径
    # 请确保该路径下包含 config.json, tokenizer.json, model.safetensors 等完整文件
    model_dir = "/home/ubuntu1/wangzihe/DeepSeek-V3"

    print(f"Checking path: {model_dir}")
    if not os.path.exists(model_dir):
        print(f"Error: Path {model_dir} does not exist!")
        return

    print("Loading tokenizer...")
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    print("Loading model (this may take time)...")
    try:
        # 加载模型
        # torch_dtype=torch.bfloat16: 使用半精度加载，节省显存（DeepSeek 推荐）
        # device_map="auto": 自动将模型分配到可用的 GPU 上
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # 加载生成配置
        # 如果本地文件夹中有 generation_config.json，这将应用默认的生成参数
        try:
            model.generation_config = GenerationConfig.from_pretrained(model_dir)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
        except Exception:
            print("Warning: No generation_config.json found, using default.")

    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Tip: Check if you have enough GPU memory (VRAM). DeepSeek-V3 is a very large model.")
        return

    print("\n=== Model Loaded Successfully! Starting Inference Test ===\n")

    # 2. 定义测试输入
    test_prompt = "你好，DeepSeek。请帮我写一段简单的 Python 代码来打印 'Hello World'。"

    # 构建对话格式 (如果是 Chat 版本模型，最好遵循对应的 template)
    # 这里使用简单的直接输入进行测试
    messages = [
        {"role": "user", "content": test_prompt}
    ]

    # 应用聊天模板 (如果 tokenizer 支持)
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
    except AttributeError:
        # 如果不支持 chat template，回退到普通文本拼接
        inputs = tokenizer(test_prompt, return_tensors="pt").input_ids.to(model.device)

    # 3. 生成回复
    # 设置生成参数：max_new_tokens 控制生成长度
    outputs = model.generate(
        inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # 4. 解码并打印输出
    # 截取掉输入部分的 token，只显示新生成的内容
    generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    print(f"Input: {test_prompt}")
    print("-" * 30)
    print(f"Output:\n{generated_text}")
    print("-" * 30)
    print("\nTest Finished.")


if __name__ == "__main__":
    load_and_test_model()

