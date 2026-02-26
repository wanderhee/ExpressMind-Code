"""
多模态大模型底层模型替换
"""

import argparse
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
)

try:

    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as Qwen3VLForConditionalGeneration

def get_args():
    parser = argparse.ArgumentParser(description="替换多模态大模型的底层 LLM 权重")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full"],
        required=True,
        help="选择模式: 微调 或 预训练",
    )
    parser.add_argument(
        "--qwen3vl_path",
        type=str,
        required=True,
        help="原始 Qwen3-VL 模型路径",
    )
    parser.add_argument(
        "--custom_path",
        type=str,
        required=True,
        help="针对 LoRA 模式指适配器目录和针对预训练模式指全量模型目录",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="最终生成的合并模型保存路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="处理设备默认 cpu",
    )
    return parser.parse_args()

def main():
    args = get_args()

    print("\n" + "="*60)
    print(f"模式: [{'LoRA合并' if args.mode == '微调训练' else '预训练权重训练'}]")
    print("="*60)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.qwen3vl_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.qwen3vl_path)

    if args.mode == "lora":
        try:
            from peft import PeftModel
        except ImportError:
            return

        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            args.custom_path
        )
        print("  - 挂载成功，正在执行 Merge & Unload...")
        model.language_model = model.language_model.merge_and_unload()
        print("  预训练权重已永久合并至 Qwen3-VL 的底层 LLM。")

    else:
        print(f"\n[2/4] 从本地加载微调后的全量 Qwen3 模型: {args.custom_path}")
        custom_llm = AutoModelForCausalLM.from_pretrained(
            args.custom_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True
        )
        
        print("  - 正在执行底层 LLM 权重替换...")

        model.language_model.model = custom_llm.model
        if hasattr(custom_llm, "lm_head"):
            model.language_model.lm_head = custom_llm.lm_head
        print("  - 权重替换完成。")
        del custom_llm


    print(f"\n[3/4] 正在保存合并后的标准 Qwen3-VL 模型到: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)
    processor.save_pretrained(args.output_path)

    print("\n" + "="*60)
    print("【操作完成】")
    print(f"生成的模型已就绪：{args.output_path}")
    print(f"提示：您可以直接使用 web_demo_mm.py 运行此目录下的模型。")
    print("="*60 + "\n")


    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
