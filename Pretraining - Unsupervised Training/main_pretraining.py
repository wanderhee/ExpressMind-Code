
"""
ExpressMind 预训练主程序
适用于 conda 环境 pyt39
"""

import os
import sys
import yaml
import torch
from pathlib import Path
import logging
import argparse
import transformers
import sys as sys_module
from types import ModuleType

class _BeamSearchScorerPlaceholder:
    """兼容性占位符类 - BeamSearchScorer 在新版 transformers 中已移除"""
    def __init__(self, *args, **kwargs):
        pass

def fix_transformers_compatibility():
    """修复 transformers_stream_generator 与新版 transformers 的兼容性问题"""
    try:
        try:
            from transformers.generation.beam_search import BeamSearchScorer
            return True
        except ImportError:
            pass
        
        if not hasattr(transformers, 'generation'):
            generation_module = ModuleType('generation')
            sys_module.modules['transformers.generation'] = generation_module
            setattr(transformers, 'generation', generation_module)
        else:
            generation_module = transformers.generation
        
        if not hasattr(generation_module, 'beam_search'):
            beam_search_module = ModuleType('beam_search')
            sys_module.modules['transformers.generation.beam_search'] = beam_search_module
            setattr(generation_module, 'beam_search', beam_search_module)
        else:
            beam_search_module = generation_module.beam_search
        
        BeamSearchScorer = _BeamSearchScorerPlaceholder
        transformers.__dict__['BeamSearchScorer'] = BeamSearchScorer
        
        if 'BeamSearchScorer' not in transformers.__dict__:
            sys_module.modules['transformers'].__dict__['BeamSearchScorer'] = BeamSearchScorer
        
        beam_search_module.BeamSearchScorer = BeamSearchScorer
        setattr(beam_search_module, 'BeamSearchScorer', BeamSearchScorer)
        setattr(generation_module, 'BeamSearchScorer', BeamSearchScorer)
        setattr(transformers, 'BeamSearchScorer', BeamSearchScorer)
        
        beam_search_module.__dict__['BeamSearchScorer'] = BeamSearchScorer
        generation_module.__dict__['BeamSearchScorer'] = BeamSearchScorer
        
        if hasattr(transformers, '__all__'):
            if 'BeamSearchScorer' not in transformers.__all__:
                transformers.__all__.append('BeamSearchScorer')
        
        original_getattr = getattr(transformers, '__getattr__', None)
        def __getattr__(name):
            if name == 'BeamSearchScorer':
                return BeamSearchScorer
            if original_getattr:
                return original_getattr(name)
            raise AttributeError(f"module '{transformers.__name__}' has no attribute '{name}'")
        
        transformers.__getattr__ = __getattr__
        
        return True
    except Exception as e:
        return False


fix_transformers_compatibility()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_processor import PDFProcessor
from data_preprocessor import DataPreprocessor
from trainer_pretraining import QwenTrainerFullFinetune
from gpu_utils import GPUManager


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pretraining.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path="config_pretraining.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def process_pdfs(config):
    """预处理无监督训练数据PDF文件"""
    data_config = config["data"]
    
    logger.info("=" * 60)
    logger.info("步骤1: 无监督训练数据处理PDF文件")
    logger.info("=" * 60)
    
    processor = PDFProcessor(
        pdf_dir=data_config["pdf_dir"],
        output_dir=data_config["processed_data_dir"],
        min_text_length=data_config["min_text_length"]
    )
    
    results = processor.process_all_pdfs(clean=data_config["clean_text"])
    processor.save_processed_texts(results)
    
    logger.info("无监督训练数据处理PDF处理完成\n")


def prepare_dataset(config):
    """训练数据集"""
    data_config = config["data"]
    
    logger.info("=" * 60)
    logger.info("步骤2: 训练数据集")
    logger.info("=" * 60)
    
    preprocessor = DataPreprocessor(
        processed_data_dir=data_config["processed_data_dir"],
        chunk_size=data_config["chunk_size"],
        chunk_overlap=data_config["chunk_overlap"],
        train_split=data_config["train_split"]
    )
    
    dataset = preprocessor.create_dataset()
    
    logger.info("数据集完成\n")
    return dataset


def train_model(config):
    """全量无监督训练"""
    logger.info("=" * 60)
    logger.info("步骤3: 全量无监督训练")
    logger.info("=" * 60)
    
    trainer = QwenTrainerFullFinetune(config_path="config_pretraining.yaml")
    metrics = trainer.train()
    
    logger.info("训练完成\n")
    return metrics


def check_gpu():
    """GPU状态"""
    logger.info("=" * 60)
    logger.info("GPU状态")
    logger.info("=" * 60)
    
    GPUManager.print_gpu_info()
    GPUManager.is_h20_or_4090()


def evaluate_model(config_path: str = "config_pretraining.yaml", model_path: str = None):
    """
    评估全量无监督训练模型
    
    Args:
        config_path: 配置文件路径
        model_path: 模型路径
    
    Returns:
        评估指标字典
    """
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if model_path is None:
        model_path = str(Path(config["training"]["output_dir"]) / "final_model")
    
    logger.info("=" * 60)
    logger.info("步骤4: 评估无监督训练模型")
    logger.info("=" * 60)
    logger.info(f"模型路径: {model_path}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        from datasets import load_from_disk
        from tqdm import tqdm
        import numpy as np
        
     
        logger.info("加载无监督训练模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if config["training"].get("bf16", False) else torch.float16
        )
        
    
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        
        if hasattr(model, 'config'):
            model.config.use_cache = True
        
        model.eval()
        logger.info("评估模型")
        
      
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
       
        logger.info("加载验证数据集...")
        dataset_path = Path(config["data"]["processed_data_dir"]) / "dataset"
        
        if not dataset_path.exists():
            logger.warning(f"无数据集: {dataset_path}")
            logger.info("请先运行数据准备步骤: python main_pretraining.py --step prepare")
            return None
        
        dataset = load_from_disk(str(dataset_path))
        eval_dataset = dataset["validation"]
        
        logger.info(f"验证集样本数: {len(eval_dataset)}")
        logger.info(f"验证集特征: {eval_dataset.features}")
        
 
        if "input_ids" not in eval_dataset.column_names:
            logger.warning("验证集未tokenized，需要进行tokenize...")
            
            max_length = config["model"]["max_length"]
            
            def tokenize_for_eval(examples):
                tokenized = tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors=None,
                )
                tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
                return tokenized
            
            eval_dataset = eval_dataset.map(
                tokenize_for_eval,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing validation set"
            )
            
            logger.info(f"✓ Tokenize完成，特征: {eval_dataset.features}")
        
        # 计算困惑度
        logger.info("计算困惑度...")
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for sample in tqdm(eval_dataset, desc="评估中"):
                input_ids = torch.tensor([sample["input_ids"]]).to(model.device)
                labels = torch.tensor([sample["labels"]]).to(model.device)
                
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item() * len(sample["input_ids"])
                total_tokens += len(sample["input_ids"])
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        metrics = {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
            "eval_samples": len(eval_dataset),
            "total_tokens": total_tokens
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 评估无监督训练模型指标")
        logger.info("=" * 60)
        logger.info(f"Loss: {avg_loss:.4f}")
        logger.info(f"Perplexity: {perplexity:.2f}")
        logger.info(f"样本数: {len(eval_dataset)}")
        logger.info(f"token: {total_tokens:,}")
        logger.info("=" * 60 + "\n")
        
        return metrics
        
    except Exception as e:
        logger.error(f"评估失败: {str(e)}", exc_info=True)
        return None


def inference_model(config_path: str = "config_pretraining.yaml", model_path: str = None, prompts: list = None):
    """
    使用无监督训练模型进行推理测试
    
    Args:
        config_path: 配置文件路径
        model_path: 模型路径
        prompts: 测试提示列表
    
    Returns:
        推理结果列表
    """
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if model_path is None:
        model_path = str(Path(config["training"]["output_dir"]) / "final_model")
    
    logger.info("=" * 60)
    logger.info("步骤5: 无监督训练模型")
    logger.info("=" * 60)
    logger.info(f"模型路径: {model_path}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        torch_dtype = torch.bfloat16 if config["training"].get("bf16", False) else torch.float16
        
        
        logger.info("\n" + "=" * 60)
        logger.info("加载无监督训练模型")
        logger.info("=" * 60)
        logger.info(f"模型路径: {model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
        
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        
        if hasattr(model, 'config'):
            model.config.use_cache = True
        
        model.eval()
        logger.info("✓ 模型加载完成并设置为推理模式")
        
        logger.info("\n" + "=" * 60)
        logger.info("加载分词器")
        logger.info("=" * 60)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
        )

        logger.info(f"原始分词器配置 - pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}")

        eos_token_id = None
        if tokenizer.eos_token is not None:
            eos_token_id = tokenizer.eos_token_id
        elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            eos_token_id = tokenizer.eos_token_id
        elif hasattr(tokenizer, 'model') and hasattr(tokenizer.model, 'eos_token_id'):
            eos_token_id = tokenizer.model.eos_token_id

        eos_token_str = None
        if tokenizer.eos_token is not None:
            eos_token_str = tokenizer.eos_token
        elif hasattr(tokenizer, 'special_tokens_map') and 'eos_token' in tokenizer.special_tokens_map:
            eos_token_str = tokenizer.special_tokens_map['eos_token']

        if tokenizer.pad_token is None:
            if eos_token_str is not None:
                tokenizer.pad_token = eos_token_str
                tokenizer.pad_token_id = eos_token_id if eos_token_id is not None else tokenizer.convert_tokens_to_ids(eos_token_str)
                logger.info(f"✓ 已设置 pad_token = eos_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
            else:
                if hasattr(tokenizer, 'convert_tokens_to_ids'):
                    for potential_eos in ['<|endoftext|>', '</s>', '<|im_end|>']:
                        try:
                            eos_id = tokenizer.convert_tokens_to_ids(potential_eos)
                            if eos_id is not None and eos_id != tokenizer.unk_token_id:
                                tokenizer.eos_token = potential_eos
                                tokenizer.eos_token_id = eos_id
                                tokenizer.pad_token = potential_eos
                                tokenizer.pad_token_id = eos_id
                                logger.info(f"✓ 找到eos_token: {potential_eos} (id: {eos_id})，已设置为pad_token")
                                break
                        except:
                            continue

                if tokenizer.pad_token is None:
                    raise ValueError(
                        "Tokenizer没有pad_token或eos_token，无法设置padding\n"
                        f"检查tokenizer\n"
                        f"tokenizer类型: {type(tokenizer).__name__}"
                    )

        if tokenizer.pad_token_id is None:
            if tokenizer.pad_token is not None:
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
                logger.info(f"确保设置pad_token_id: {tokenizer.pad_token_id}")

        logger.info(f"pad_token: '{tokenizer.pad_token}', eos_token: '{tokenizer.eos_token}'")
        logger.info(f"pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}")
        logger.info("分词器加载完成")
        
        # 问答测试
        if prompts is None:
            prompts = [
                "根据北京市城市道路空间非机动车停车设施设置规范，",
                "非机动车停车设施内宜附加箭头明确停放朝向，箭头设置应符合哪些要求：",
                "非机动车停车设施的边线应包围存车架等附属设施的线宽应为？",
            ]
        
        logger.info(f"\n测试提示数量: {len(prompts)}")
        logger.info("=" * 60 + "\n")
        
        # 文本生成
        def generate_text(model, tokenizer, prompt_text, max_new_tokens=100, temperature=0.7, top_p=0.9):
            """生成文本"""
            try:
                logger.debug(f"开始编码输入: '{prompt_text[:50]}...'")

                tokenized = tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=512,
                    padding=False,
                    return_tensors=None
                )

                logger.debug(f"tokenized类型: {type(tokenized)}")
                logger.debug(f"tokenized keys: {tokenized.keys() if hasattr(tokenized, 'keys') else 'no keys'}")

                if 'input_ids' not in tokenized:
                    raise ValueError(f"tokenizer 返回结果中无 input_ids 字段: {tokenized}")

                input_ids_list = tokenized['input_ids']
                logger.debug(f"input_ids_list类型: {type(input_ids_list)}")
                logger.debug(f"input_ids_list长度: {len(input_ids_list) if hasattr(input_ids_list, '__len__') else 'no len'}")

                from transformers import BatchEncoding
                inputs = BatchEncoding({
                    'input_ids': torch.tensor([input_ids_list], dtype=torch.long),
                })

                if 'attention_mask' in tokenized:
                    attention_mask_list = tokenized['attention_mask']
                    inputs['attention_mask'] = torch.tensor([attention_mask_list], dtype=torch.long)

                logger.debug(f"手动创建inputs成功: {inputs}")
                logger.debug(f"input_ids shape: {inputs.input_ids.shape}")

            except Exception as e:
                logger.error(f"输入编码失败: {e}")
                logger.error(f"tokenizer类型: {type(tokenizer)}")
                import traceback
                logger.error(f"完整错误信息:\n{traceback.format_exc()}")
                raise ValueError(f"tokenizer 编码输入失败: {e}")

           
            try:
                if hasattr(model, 'device'):
                    device = model.device
                else:
                    try:
                        device = next(model.parameters()).device
                    except StopIteration:
                        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            except Exception as e:
                logger.warning(f"设备检测失败: {e}，使用默认设备")
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            if inputs is None or inputs.input_ids is None:
                raise ValueError(f"tokenizer 返回的输入为空: inputs={inputs}")

            input_ids = inputs.input_ids
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
                logger.debug(f"转换输入为张量")

            input_ids = input_ids.to(device)
            if 'attention_mask' in inputs and inputs.attention_mask is not None:
                attention_mask = inputs.attention_mask.to(device)
            else:
                attention_mask = torch.ones_like(input_ids).to(device)

            logger.debug(f"输入形状: input_ids={input_ids.shape}, attention_mask={attention_mask.shape if attention_mask is not None else None}, 设备: {device}")
            logger.debug(f"分词器配置: pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")
            
            with torch.no_grad():
                generate_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": True,
                    "repetition_penalty": 1.1,
                }
                
                if attention_mask is not None:
                    generate_kwargs["attention_mask"] = attention_mask
                
                if tokenizer.pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
                if tokenizer.eos_token_id is not None:
                    generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
                
                outputs = model.generate(**generate_kwargs)

                if outputs is None:
                    raise ValueError("模型生成为空")

                if not hasattr(outputs, 'shape') or outputs.shape[0] == 0:
                    raise ValueError(f"模型无法生成: {outputs}")

           
            try:
                input_length = input_ids.shape[1]
                generated_tokens = outputs[0][input_length:]

                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                if not generated_text.strip():
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
                    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

            except Exception as decode_error:
                logger.warning(f"解码失败: {decode_error}，尝试其他方法")
                try:
                    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text
                except Exception as backup_error:
                    logger.error(f"备用解码失败: {backup_error}")
                    raise ValueError(f"文本解码失败: {decode_error} -> {backup_error}")

            return full_text, generated_text
        
        logger.info("\n" + "=" * 60)
        logger.info("📝 推理测试无监督训练模型")
        logger.info("=" * 60 + "\n")
        
        results = []
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"【推理测试 {i}/{len(prompts)}】")
            logger.info(f"{'='*60}")
            logger.info(f"📌 输入提示: {prompt}")
            
            full_text = ""
            new_text = ""

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info(f"\n{'─'*60}")
                logger.info("🟢 无监督训练模型生成结果:")
                logger.info(f"{'─'*60}")

                full_text, new_text = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9
                )
                
                logger.info(f"\n📝 完整输出（包含提示）:")
                logger.info(f"   {full_text}")
                logger.info(f"\n✨ 新生成的部分:")
                logger.info(f"   {new_text}")
                logger.info(f"\n📊 统计信息:")
                logger.info(f"   - 提示长度: {len(prompt)} 字符")
                logger.info(f"   - 生成长度: {len(new_text)} 字符")
                logger.info(f"   - 总长度: {len(full_text)} 字符")
                
            except Exception as e:
                logger.error(f"❌ 生成过程中出错: {str(e)}")
                if not full_text:
                    full_text = f"[Error: {str(e)}]"
                    new_text = ""
            
            results.append({
                "prompt": prompt,
                "full_response": full_text,
                "generated_text": new_text
            })
            
            logger.info("-" * 60)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 推理测试总结 - 预训练")
        logger.info("=" * 60)
        logger.info(f"✅ 完成 {len(results)} 个推理测试")
        logger.info("=" * 60 + "\n")
        
        return results
        
    except Exception as e:
        logger.error(f"推理失败: {str(e)}", exc_info=True)
        return None


def plot_training_curves(metrics_file: str = "./output_pretraining/training_metrics.json"):
    """绘制训练曲线（从JSON文件读取）"""
    logger.info("=" * 60)
    logger.info("步骤6: 绘制无监督训练模型训练曲线")
    logger.info("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.font_manager
        matplotlib.use('Agg')
        from pathlib import Path
        import numpy as np
        import json
        import platform
        
        # 设置中文字体
        if platform.system() == 'Windows':
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
            available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
            font_found = False
            for font in chinese_fonts:
                if font in available_fonts:
                    plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                    font_found = True
                    logger.info(f"使用中文字体: {font}")
                    break
            if not font_found:
                logger.warning("若系统中文字体则会显示为方块")
        else:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
        
        plt.rcParams['axes.unicode_minus'] = False
        
        metrics_path = Path(metrics_file)
        if not metrics_path.exists():
            logger.warning(f"未找到训练文件: {metrics_file}")
            logger.info("请确定训练已完成并保存了指标文件")
            return None
        
        logger.info(f"从文件加载训练指标: {metrics_file}")
        
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        
        logger.info(f"加载的指标类型: {list(metrics_data.keys())}")
        
        for key in ["loss", "learning_rate", "epoch", "step"]:
            data = metrics_data.get(key, [])
            logger.info(f"  {key}: {len(data)} 条记录")
            if len(data) > 0:
                logger.info(f"    示例: {data[0]}")
        
        has_loss = len(metrics_data.get("loss", [])) > 0
        has_lr = len(metrics_data.get("learning_rate", [])) > 0
        has_epoch = len(metrics_data.get("epoch", [])) > 0
        
        if not has_loss and not has_lr and not has_epoch:
            logger.warning("训练指标文件中没有找到任何数据")
            return None
        
        # 作图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练无监督训练模型曲线', fontsize=16, fontweight='bold')
        
        # 训练损失
        if has_loss:
            loss_data = metrics_data["loss"]
            steps = [item["step"] for item in loss_data]
            values = [item["value"] for item in loss_data]
            
            axes[0, 0].plot(steps, values, 'b-', linewidth=2, label='Training Loss')
            axes[0, 0].set_xlabel('Steps', fontsize=12)
            axes[0, 0].set_ylabel('Loss', fontsize=12)
            axes[0, 0].set_title('训练损失 (Training Loss)', fontsize=14, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            logger.info(f"✓ 绘制了 {len(steps)} 个训练损失数据点")
        else:
            axes[0, 0].text(0.5, 0.5, '无训练损失数据', ha='center', va='center', fontsize=14)
            axes[0, 0].set_title('训练损失 (Training Loss)', fontsize=14, fontweight='bold')
        
        # 学习率
        if has_lr:
            lr_data = metrics_data["learning_rate"]
            steps = [item["step"] for item in lr_data]
            values = [item["value"] for item in lr_data]
            
            axes[0, 1].plot(steps, values, 'r-', linewidth=2, label='Learning Rate')
            axes[0, 1].set_xlabel('Steps', fontsize=12)
            axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
            axes[0, 1].set_title('学习率变化 (Learning Rate)', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            logger.info(f"✓ 绘制了 {len(steps)} 个学习率数据点")
        else:
            axes[0, 1].text(0.5, 0.5, '无学习率数据', ha='center', va='center', fontsize=14)
            axes[0, 1].set_title('学习率变化 (Learning Rate)', fontsize=14, fontweight='bold')
        
        # Epoch进度
        if has_epoch:
            epoch_data = metrics_data["epoch"]
            steps = [item["step"] for item in epoch_data]
            values = [item["value"] for item in epoch_data]
            
            axes[1, 0].plot(steps, values, 'g-', linewidth=2, marker='o', label='Epoch Progress')
            axes[1, 0].set_xlabel('Steps', fontsize=12)
            axes[1, 0].set_ylabel('Epoch', fontsize=12)
            axes[1, 0].set_title('训练进度 (Epoch Progress)', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.2)
            axes[1, 0].legend()
            logger.info(f"✓ 绘制了 {len(steps)} 个epoch数据点")
        else:
            axes[1, 0].text(0.5, 0.5, '无Epoch数据', ha='center', va='center', fontsize=14)
            axes[1, 0].set_title('训练进度 (Epoch Progress)', fontsize=14, fontweight='bold')
        
        # 训练损失分布
        if has_loss:
            loss_data = metrics_data["loss"]
            values = [item["value"] for item in loss_data]
            
            axes[1, 1].hist(values, bins=min(50, len(values)), color='purple', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'平均值: {np.mean(values):.4f}')
            axes[1, 1].set_xlabel('Loss Value', fontsize=12)
            axes[1, 1].set_ylabel('Frequency', fontsize=12)
            axes[1, 1].set_title('损失分布 (Loss Distribution)', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, '无损失分布数据', ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('损失分布 (Loss Distribution)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
     
        output_path = Path("./output_pretraining/training_curves.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        logger.info(f"✅ 训练曲线已保存: {output_path}")
        logger.info("=" * 60 + "\n")
        
        return str(output_path)
        
    except ImportError as e:
        logger.warning(f"缺少依赖库: {e}")
        logger.warning("请安装: pip install matplotlib")
        return None
    except Exception as e:
        logger.error(f"绘制曲线失败: {str(e)}", exc_info=True)
        return None


def print_training_summary(metrics: dict):
    """打印训练总结和指标"""
    logger.info("\n" + "=" * 60)
    logger.info("📈 训练总结 - 预训练 FULL FINE-TUNING SUMMARY")
    logger.info("=" * 60)
    
    if metrics:
        logger.info(f"✅ 训练状态: 成功完成")
        logger.info(f"⏱️  训练时长: {metrics.get('train_runtime', 0):.2f} 秒")
        logger.info(f"📊 训练样本/秒: {metrics.get('train_samples_per_second', 0):.3f}")
        logger.info(f"🔄 训练步数/秒: {metrics.get('train_steps_per_second', 0):.3f}")
        logger.info(f"📉 最终Loss: {metrics.get('train_loss', 0):.4f}")
        logger.info(f"🔢 训练轮数: {metrics.get('epoch', 0):.1f}")
    else:
        logger.info(f"⚠️  训练状态: 未获取到指标")
    
    logger.info("=" * 60 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ExpressMind无监督训练系统")
    
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "process", "prepare", "train", "eval", "inference", "plot", "check_gpu"],
        default="all",
        help="执行的步骤: all(全部), process(处理PDF), prepare(准备数据), train(训练), eval(评估), inference(推理), plot(绘图), check_gpu(检查GPU)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config_pretraining.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型路径"
    )
    
    args = parser.parse_args()
    
    logger.info("\n" + "=" * 60)
    logger.info("ExpressMind无监督训练系统")
    logger.info("支持完整训练流程 + 自动评估 + 推理测试")
    logger.info("ExpressMind无监督训练模式：所有参数都参与训练")
    logger.info("=" * 60 + "\n")
    
    if not Path(args.config).exists():
        logger.error(f"配置文件不存在: {args.config}")
        logger.info("请确保配置文件存在，或使用默认的 config_pretraining.yaml")
        sys.exit(1)
    
    logger.info(f"使用配置文件: {args.config}")
    
    try:
        if args.step == "check_gpu":
            check_gpu()
            
        elif args.step == "process":
            config = load_config(args.config)
            process_pdfs(config)
            
        elif args.step == "prepare":
            config = load_config(args.config)
            prepare_dataset(config)
            
        elif args.step == "train":
            config = load_config(args.config)
            metrics = train_model(config)
            if metrics:
                print_training_summary(metrics)
                
        elif args.step == "eval":
            eval_metrics = evaluate_model(args.config, args.model_path)
            
        elif args.step == "inference":
            inference_results = inference_model(args.config, args.model_path)
            
        elif args.step == "plot":
            plot_training_curves()
            
        elif args.step == "all":
            # 完整流程释明
            logger.info("🎯 执行ExpressMind无监督训练完整流程")
            logger.info("=" * 60)
            logger.info("步骤清单:")
            logger.info("  1. 检查GPU状态")
            logger.info("  2. 处理PDF文件")
            logger.info("  3. 准备训练数据")
            logger.info("  4. 训练模型（预训练）")
            logger.info("  5. 评估模型")
            logger.info("  6. 推理测试")
            logger.info("  7. 绘制训练曲线")
            logger.info("=" * 60 + "\n")
            
            config = load_config(args.config)
            
            check_gpu()
            
            if not torch.cuda.is_available():
                logger.warning("未检测到 GPU，将使用 CPU 训练（速度较慢）")
                response = input("是否继续？(y/n): ")
                if response.lower() != 'y':
                    logger.info("已取消训练")
                    sys.exit(0)
            
            #处理无监督训练数据PDF
            process_pdfs(config)
            prepare_dataset(config)
            metrics = train_model(config)
            if metrics:
                print_training_summary(metrics)
            
            #评估模型
            logger.info("\n" + "🔍 开始训练后评估...\n")
            eval_metrics = evaluate_model(args.config)
            #测试
            logger.info("\n" + "💬 开始推理测试...\n")
            inference_results = inference_model(args.config)
            #绘制曲线
            logger.info("\n" + "📊 生成训练曲线图...\n")
            plot_path = plot_training_curves()
            
            
            logger.info("\n" + "=" * 60)
            logger.info("🎉 ExpressMind无监督训练完成")
            logger.info("=" * 60)
            logger.info("✅ 完成内容:")
            logger.info("  ✓ GPU检查")
            logger.info("  ✓ PDF处理")
            logger.info("  ✓ 数据准备")
            logger.info("  ✓ ExpressMind无监督训练")
            if eval_metrics:
                logger.info(f"  ✓ 模型评估 (Perplexity: {eval_metrics.get('perplexity', 0):.2f})")
            if inference_results:
                logger.info(f"  ✓ 推理测试 ({len(inference_results)} 个样本)")
            if plot_path:
                logger.info(f"  ✓ 训练曲线: {plot_path}")
            logger.info("\n📁 输出位置:")
            logger.info("  - 模型权重: ./output_pretraining/final_model/")
            logger.info("  - 训练曲线: ./output_pretraining/training_curves.png")
            logger.info("  - 训练日志: ./training_pretraining.log")
            logger.info("=" * 60 + "\n")
            
        logger.info("✨ 程序执行完毕！")
        
    except Exception as e:
        logger.error(f"❌ 执行失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

