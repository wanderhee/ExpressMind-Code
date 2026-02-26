
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
)

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as Qwen3VLForConditionalGeneration


# ==================== VPA模块定义 ====================
class VisualPriorAlignment(nn.Module):
    """
    Visual-Prior Alignment (VPA) Module
    用于增强多模态模型中视觉特征的贡献权重
    """
    def __init__(self, d_visual=2048, d_language=4096, d_hidden=512):
        super().__init__()
        self.d_visual = d_visual
        self.d_language = d_language
        self.d_hidden = d_hidden
        
        # 视觉特征投影层
        self.visual_projection = nn.Sequential(
            nn.Linear(d_visual, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_language)
        )
        
        # 跨模态注意力权重矩阵
        self.cross_attn_projection = nn.Linear(d_language, d_language, bias=False)
        
        # 视觉增强权重（可学习参数）
        self.visual_weight = nn.Parameter(torch.ones(1, 1, d_language) * 0.5)
        
    def forward(self, visual_features, language_embeddings=None):
        """
        前向传播
        
        Args:
            visual_features: 视觉特征 [batch_size, seq_len, d_visual]
            language_embeddings: 语言嵌入 [batch_size, seq_len_lang, d_language]
            
        Returns:
            对齐后的视觉特征 [batch_size, seq_len, d_language]
        """
        # 1. 视觉特征投影
        projected_visual = self.visual_projection(visual_features)  # [B, L_vis, D_lang]
        
        if language_embeddings is not None:
            # 2. 计算跨模态注意力（使用公式: attention = softmax(QK^T/√d) * V）
            Q = projected_visual  # [B, L_vis, D_lang]
            K = self.cross_attn_projection(language_embeddings)
            V = projected_visual
            
            # 计算注意力分数
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_language ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # 注意力加权
            aligned_visual = torch.matmul(attention_weights, V)
        else:
            aligned_visual = projected_visual
            
        # 3. 视觉特征增强
        aligned_visual = aligned_visual * self.visual_weight
        
        return aligned_visual


# ==================== VPA增强的Qwen3VL模型类 ====================
class Qwen3VLWithVPA(Qwen3VLForConditionalGeneration):
    """
    Qwen3-VL模型，集成了VPA机制
    """
    def __init__(self, config, vpa_config=None):
        super().__init__(config)
        
        # 获取维度信息
        if hasattr(self.visual_encoder, 'config'):
            d_visual = self.visual_encoder.config.hidden_size
        else:
            d_visual = 2048  # 默认值
            
        d_language = config.hidden_size
        
        # 如果提供了VPA配置，使用配置；否则使用默认值
        if vpa_config:
            d_hidden = vpa_config.get('d_hidden', min(d_visual, d_language) // 2)
        else:
            d_hidden = min(d_visual, d_language) // 2
        
        # 创建VPA模块
        self.vpa = VisualPriorAlignment(
            d_visual=d_visual,
            d_language=d_language,
            d_hidden=d_hidden
        )
        
        # 保存原始的前向传播函数
        self.original_forward = super().forward
        
    def forward(self, *args, **kwargs):
        """
        重写前向传播，集成VPA机制
        """
        # 提取图像输入（假设输入中有'images'或'pixel_values'）
        if 'images' in kwargs:
            images = kwargs['images']
        elif 'pixel_values' in kwargs:
            images = kwargs['pixel_values']
        else:
            # 如果无法提取图像，回退到原始前向传播
            return self.original_forward(*args, **kwargs)
        
        # 使用视觉编码器提取特征
        if hasattr(self, 'visual_encoder'):
            visual_features = self.visual_encoder(images)
            
            # 提取语言嵌入（如果有的话）
            if 'input_ids' in kwargs:
                input_ids = kwargs['input_ids']
                # 获取语言模型的词嵌入层
                if hasattr(self.language_model, 'get_input_embeddings'):
                    language_embeddings = self.language_model.get_input_embeddings()(input_ids)
                    
                    # 应用VPA模块
                    enhanced_visual = self.vpa(visual_features.last_hidden_state, language_embeddings)
                    
                    # 替换原始的视觉特征
                    visual_features.last_hidden_state = enhanced_visual
        
        # 继续原始的前向传播
        return self.original_forward(*args, **kwargs)


# ==================== 主程序 ====================
def get_args():
    parser = argparse.ArgumentParser(description="替换或合并 Qwen3-VL 底层 LLM 权重，并集成VPA机制")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full"],
        required=True,
        help="选择模式: 'lora' (合并 LoRA 适配器) 或 'full' (替换全量权重)",
    )
    parser.add_argument(
        "--qwen3vl_path",
        type=str,
        required=True,
        help="原始 Qwen3-VL 模型路径 (HF ID 或本地路径)",
    )
    parser.add_argument(
        "--custom_path",
        type=str,
        required=True,
        help="本文微调的模型路径 (针对 LoRA 模式指适配器目录，针对 Full 模式指全量模型目录)",
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
        help="处理设备 (默认 cpu, 推荐用 cpu 以防显存不足)",
    )
    parser.add_argument(
        "--enable_vpa",
        action="store_true",
        default=True,
        help="启用VPA机制 (默认启用)",
    )
    parser.add_argument(
        "--vpa_config",
        type=str,
        default=None,
        help="VPA配置JSON文件路径 (可选)",
    )
    return parser.parse_args()


def main():
    args = get_args()

    print("\n" + "="*60)
    print(f"模式: [{'LoRA 注入合并' if args.mode == 'lora' else '全量权重替换'}]")
    print(f"VPA机制: {'启用' if args.enable_vpa else '禁用'}")
    print("="*60)

    # 1. 加载原始 VLM
    print(f"\n[1/5] 正在加载 Qwen3-VL 基础模型: {args.qwen3vl_path}")
    
    # 如果启用VPA，使用自定义的模型类
    if args.enable_vpa:
        print("  - 使用VPA增强的模型架构")
        # 先加载配置
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.qwen3vl_path)
        
        # 读取VPA配置（如果有）
        vpa_config = None
        if args.vpa_config and os.path.exists(args.vpa_config):
            import json
            with open(args.vpa_config, 'r') as f:
                vpa_config = json.load(f)
        
        # 创建VPA增强的模型
        model = Qwen3VLWithVPA.from_pretrained(
            args.qwen3vl_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True,
            vpa_config=vpa_config
        )
    else:
        # 使用原始模型
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.qwen3vl_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True
        )
    
    processor = AutoProcessor.from_pretrained(args.qwen3vl_path)

    # 2. 处理 LLM 部分
    if args.mode == "lora":
        print(f"\n[2/5] 加载 LoRA 适配器并将权重合并到 language_model...")
        try:
            from peft import PeftModel
        except ImportError:
            print("错误: 请先安装 peft 库 (pip install peft)")
            return

        # 将 LoRA 挂载到 VL 模型的 language_model 部分
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            args.custom_path
        )
        print("  - 挂载成功，正在执行 Merge & Unload...")
        model.language_model = model.language_model.merge_and_unload()
        print("  - LoRA 权重已永久合并至 Qwen3-VL 的底层 LLM。")

    else:  # Full 模式
        print(f"\n[2/5] 从本地加载微调后的全量 Qwen3 模型: {args.custom_path}")
        custom_llm = AutoModelForCausalLM.from_pretrained(
            args.custom_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True
        )
        
        print("  - 正在执行底层 LLM 权重替换...")
        # 替换 VLM 内部的 language_model
        model.language_model.model = custom_llm.model
        if hasattr(custom_llm, "lm_head"):
            model.language_model.lm_head = custom_llm.lm_head
        print("  - 权重替换完成。")
        del custom_llm

    # 3. 如果启用了VPA，更新VPA模块的语言模型维度
    if args.enable_vpa and hasattr(model, 'vpa'):
        print(f"\n[3/5] 更新VPA模块以适应新的语言模型维度...")
        # 获取新的语言模型维度
        d_language = model.language_model.config.hidden_size
        
        # 获取视觉编码器维度
        if hasattr(model.visual_encoder, 'config'):
            d_visual = model.visual_encoder.config.hidden_size
        else:
            d_visual = model.vpa.d_visual
        
        # 重新初始化VPA模块
        model.vpa = VisualPriorAlignment(
            d_visual=d_visual,
            d_language=d_language,
            d_hidden=min(d_visual, d_language) // 2
        )
        print(f"  - VPA模块更新完成: d_visual={d_visual}, d_language={d_language}")

    # 4. 创建模型配置信息
    print(f"\n[4/5] 创建模型配置信息...")
    model_config = {
        "model_type": "qwen3-vl-with-vpa",
        "architecture": "Qwen3-VL with VPA and Custom LLM",
        "base_model": args.qwen3vl_path,
        "custom_llm": args.custom_path,
        "vpa_enabled": args.enable_vpa,
        "replacement_mode": args.mode,
        "device": args.device
    }
    
    if args.enable_vpa and hasattr(model, 'vpa'):
        model_config.update({
            "vpa_config": {
                "d_visual": model.vpa.d_visual,
                "d_language": model.vpa.d_language,
                "d_hidden": model.vpa.d_hidden
            }
        })

    # 5. 保存新模型
    print(f"\n[5/5] 正在保存增强后的 Qwen3-VL 模型到: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    
    # 保存模型权重
    model.save_pretrained(args.output_path)
    processor.save_pretrained(args.output_path)
    
    # 保存配置信息
    import json
    config_path = os.path.join(args.output_path, "model_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)
    
    print("  - 模型权重和处理器已保存")
    print(f"  - 模型配置已保存到: {config_path}")

    print("\n" + "="*60)
    print("【操作完成】")
    print(f"生成的模型已就绪：{args.output_path}")
    print(f"模型配置：{config_path}")
    print("\n模型特性:")
    print(f"  1. 底层LLM替换: {args.mode}模式")
    print(f"  2. VPA机制: {'已启用' if args.enable_vpa else '已禁用'}")
    if args.enable_vpa:
        print(f"      - 视觉特征维度: {model.vpa.d_visual}")
        print(f"      - 语言模型维度: {model.vpa.d_language}")
        print(f"      - 隐藏层维度: {model.vpa.d_hidden}")
    print(f"  3. 设备兼容性: {args.device}")
    print("\n使用提示：")
    print(f"  python3 web_demo_mm.py --checkpoint-path {args.output_path}")
    print("="*60 + "\n")

    # 释放资源
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

