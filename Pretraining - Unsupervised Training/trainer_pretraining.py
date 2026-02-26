"""
ExpressMind模型预训练
支持多GPU训练，不使用LoRA
"""
import os
import yaml
import torch
import json
from pathlib import Path
from typing import Optional
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
import logging

from gpu_utils import GPUManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import transformers
import sys
from types import ModuleType

class _BeamSearchScorerPlaceholder:

    def __init__(self, *args, **kwargs):
        pass

def fix_transformers_stream_generator_compatibility():
  
    try:
        try:
            from transformers.generation.beam_search import BeamSearchScorer
            logger.info("BeamSearchScorer 导入成功")
            return True
        except ImportError:
            pass
        
        logger.warning("创建兼容性补丁")
        
        if not hasattr(transformers, 'generation'):
            generation_module = ModuleType('generation')
            sys.modules['transformers.generation'] = generation_module
            setattr(transformers, 'generation', generation_module)
        else:
            generation_module = transformers.generation
        
        if not hasattr(generation_module, 'beam_search'):
            beam_search_module = ModuleType('beam_search')
            sys.modules['transformers.generation.beam_search'] = beam_search_module
            setattr(generation_module, 'beam_search', beam_search_module)
        else:
            beam_search_module = generation_module.beam_search
        
        BeamSearchScorer = _BeamSearchScorerPlaceholder
        transformers.__dict__['BeamSearchScorer'] = BeamSearchScorer
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
        logger.info("✓ 已创建 BeamSearchScorer 兼容性补丁")
        return True
    except Exception as e:
        logger.warning(f"无法修复 transformers_stream_generator 兼容性: {e}")
        return False


fix_transformers_stream_generator_compatibility()


class MetricsCallback(TrainerCallback):
    """训练指标存储JSON文件"""
    
    def __init__(self, output_file):
        self.output_file = output_file
        self.metrics_history = {
            "loss": [],
            "learning_rate": [],
            "epoch": [],
            "step": [],
        }
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录日志调用"""
        if logs:
            step = state.global_step
            
            if "loss" in logs:
                self.metrics_history["loss"].append({
                    "step": step,
                    "value": float(logs["loss"])
                })
                logger.info(f"[MetricsCallback] 记录loss: {logs['loss']:.4f} at step {step}")
            
            if "learning_rate" in logs:
                self.metrics_history["learning_rate"].append({
                    "step": step,
                    "value": float(logs["learning_rate"])
                })
            
            if "epoch" in logs:
                self.metrics_history["epoch"].append({
                    "step": step,
                    "value": float(logs["epoch"])
                })
            
            try:
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"保存失败: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """最终结果"""
        logger.info(f"指标输出: {self.output_file}")


class QwenTrainerFullFinetune:
    """ExpressMind模型训练器"""
    
    def __init__(self, config_path: str = "config_full_finetune.yaml"):
        """
        初始化训练器
        
        Args:
            config_path
        """
        self.config = self.load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self._tokenize_debug_count = 0
        
        GPUManager.print_gpu_info()
        
    def load_config(self, config_path: str) -> dict:
        """加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_gpus(self):
        """GPU"""
        gpu_config = self.config.get("gpu", {})
        
        if gpu_config.get("use_all_gpus", True):
            logger.info(f"ALL GPUs: {torch.cuda.device_count()} 个")
        else:
            gpu_ids = gpu_config.get("device_ids", [])
            if gpu_ids:
                GPUManager.set_visible_gpus(gpu_ids)
    
    def load_tokenizer(self):
        """分词器"""
        model_path = self.config["model"]["model_name_or_path"]
        logger.info(f"分词器: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
        )
        
        logger.info(f"Tokenizer:")
        logger.info(f"  pad_token: {self.tokenizer.pad_token}")
        logger.info(f"  eos_token: {self.tokenizer.eos_token}")
        logger.info(f"  bos_token: {getattr(self.tokenizer, 'bos_token', None)}")
        logger.info(f"  unk_token: {getattr(self.tokenizer, 'unk_token', None)}")
        
        eos_token_id = None
        if self.tokenizer.eos_token is not None:
            eos_token_id = self.tokenizer.eos_token_id
        elif hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            eos_token_id = self.tokenizer.eos_token_id
        elif hasattr(self.tokenizer, 'model') and hasattr(self.tokenizer.model, 'eos_token_id'):
            eos_token_id = self.tokenizer.model.eos_token_id
        
        eos_token_str = None
        if self.tokenizer.eos_token is not None:
            eos_token_str = self.tokenizer.eos_token
        elif hasattr(self.tokenizer, 'special_tokens_map') and 'eos_token' in self.tokenizer.special_tokens_map:
            eos_token_str = self.tokenizer.special_tokens_map['eos_token']
        
        if self.tokenizer.pad_token is None:
            if eos_token_str is not None:
                self.tokenizer.pad_token = eos_token_str
                self.tokenizer.pad_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.convert_tokens_to_ids(eos_token_str)
                logger.info(f"已设置 pad_token = eos_token: {self.tokenizer.pad_token} (id: {self.tokenizer.pad_token_id})")
            else:
                if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                    for potential_eos in ['<|endoftext|>', '</s>', '<|im_end|>']:
                        try:
                            eos_id = self.tokenizer.convert_tokens_to_ids(potential_eos)
                            if eos_id is not None and eos_id != self.tokenizer.unk_token_id:
                                self.tokenizer.pad_token = potential_eos
                                self.tokenizer.pad_token_id = eos_id
                                logger.info(f"eos_token: {potential_eos} (id: {eos_id})，pad_token")
                                break
                        except:
                            continue
                
                if self.tokenizer.pad_token is None:
                    raise ValueError(
                        "Tokenizer-without pad_token and eos_token, no-padding\n"
                        f"check tokenizer\n"
                        f"tokenizer type: {type(self.tokenizer).__name__}"
                    )
        
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.pad_token is not None:
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            elif eos_token_id is not None:
                self.tokenizer.pad_token_id = eos_token_id
            else:
                raise ValueError("no pad_token_id")
        
        assert self.tokenizer.pad_token is not None, "pad_token 设置失败"
        assert self.tokenizer.pad_token_id is not None, "pad_token_id 设置失败"
        
        logger.info(f"分词器加载完成 (pad_token={self.tokenizer.pad_token}, pad_token_id={self.tokenizer.pad_token_id})")
    
    def load_model(self):
        """ExpressMind-14b"""
        model_config = self.config["model"]
        model_path = model_config["model_name_or_path"]
        
        logger.info("=" * 60)
        logger.info("ExpressMind")
        logger.info("=" * 60)
        logger.info(f"模型路径: {model_path}")
        
        logger.info("check transformers_stream_generator compatibility...")
        fix_transformers_stream_generator_compatibility()
        
        if 'BeamSearchScorer' in transformers.__dict__:
            logger.info(f"transformers.__dict__['BeamSearchScorer'] = {type(transformers.__dict__['BeamSearchScorer'])}")
        else:
            logger.warning("transformers.__dict__, reset...")
            BeamSearchScorer = _BeamSearchScorerPlaceholder
            transformers.__dict__['BeamSearchScorer'] = BeamSearchScorer
            setattr(transformers, 'BeamSearchScorer', BeamSearchScorer)
            logger.info("The patch has been manually set up.")
        
        try:
            import transformers_stream_generator
            logger.info("✓ transformers_stream_generator 导入成功")
        except ImportError as e:
            logger.warning(f"transformers_stream_generator 导入失败: {e}，继续尝试加载模型...")
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.config["training"]["bf16"] else torch.float16,
            "device_map": "auto",
        }
        
     
        if model_config.get("load_in_8bit"):
            logger.warning("不使用8bit量化，这会影响训练效果")
            model_kwargs["load_in_8bit"] = True
        elif model_config.get("load_in_4bit"):
            logger.warning("不使用4bit量化，这会影响训练效果")
            model_kwargs["load_in_4bit"] = True
        
        logger.info("加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
    
        if self.config["training"].get("gradient_checkpointing", True):
            logger.info("启用梯度检查点...")
            self.model.gradient_checkpointing_enable()
        
      
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info("=" * 60)
        logger.info("Parametric statistics")
        logger.info("=" * 60)
        logger.info(f"总参数量: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        logger.info(f"可训练比例: {100 * trainable_params / total_params:.2f}%")
        logger.info("=" * 60)
        
        if trainable_params == 0:
            raise RuntimeError("错误：没有可训练参数！")
        
        logger.info("模型加载完成")
    
    def load_dataset(self):
        """加载数据集"""
        data_config = self.config["data"]
        dataset_path = Path(data_config["processed_data_dir"]) / "dataset"
        
        logger.info(f"加载数据集: {dataset_path}")
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")
        
        self.dataset = load_from_disk(str(dataset_path))
        logger.info(f"数据集加载完成: {self.dataset}")
    
    def tokenize_function(self, examples):
        """分词函数"""
        max_length = self.config["model"]["max_length"]
        
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        
        if self._tokenize_debug_count == 0:
            logger.info(f"[tokenize_function] 输入样本数: {len(examples['text'])}")
            logger.info(f"[tokenize_function] tokenized keys: {tokenized.keys()}")
            logger.info(f"[tokenize_function] input_ids type: {type(tokenized['input_ids'])}")
            logger.info(f"[tokenize_function] input_ids length: {len(tokenized['input_ids'])}")
            if len(tokenized['input_ids']) > 0:
                logger.info(f"[tokenize_function] first input_ids type: {type(tokenized['input_ids'][0])}")
                logger.info(f"[tokenize_function] first input_ids length: {len(tokenized['input_ids'][0])}")
                logger.info(f"[tokenize_function] first input_ids sample: {tokenized['input_ids'][0][:10]}...")
        
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        
        if self._tokenize_debug_count == 0:
            logger.info(f"[tokenize_function] labels type: {type(tokenized['labels'])}")
            logger.info(f"[tokenize_function] labels length: {len(tokenized['labels'])}")
            if len(tokenized['labels']) > 0:
                logger.info(f"[tokenize_function] first labels type: {type(tokenized['labels'][0])}")
                logger.info(f"[tokenize_function] first labels length: {len(tokenized['labels'][0])}")
                logger.info(f"[tokenize_function] labels == input_ids: {tokenized['labels'][0] == tokenized['input_ids'][0]}")
            self._tokenize_debug_count += 1
        
        return tokenized
    
    def prepare_dataset(self):
        """准备数据集"""
        logger.info("准备训练数据")
        
        tokenized_dataset = self.dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
            desc="分词处理",
        )
        
        columns_to_remove = []
        if 'token_type_ids' in tokenized_dataset['train'].column_names:
            columns_to_remove.append('token_type_ids')
        if 'attention_mask' in tokenized_dataset['train'].column_names:
            columns_to_remove.append('attention_mask')
        
        if columns_to_remove:
            logger.info(f"移除不需要的字段: {columns_to_remove}")
            tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        
        self.dataset = tokenized_dataset
        
        logger.info("数据准备完成")
        logger.info(f"[prepare_dataset] 训练集样本数: {len(self.dataset['train'])}")
        logger.info(f"[prepare_dataset] 验证集样本数: {len(self.dataset['validation'])}")
        logger.info(f"[prepare_dataset] 训练集特征: {self.dataset['train'].features}")
        
        if len(self.dataset['train']) > 0:
            first_sample = self.dataset['train'][0]
            logger.info(f"[prepare_dataset] 第一个训练样本 keys: {first_sample.keys()}")
            logger.info(f"[prepare_dataset] input_ids type: {type(first_sample['input_ids'])}")
            logger.info(f"[prepare_dataset] input_ids length: {len(first_sample['input_ids'])}")
            logger.info(f"[prepare_dataset] input_ids sample: {first_sample['input_ids'][:10]}...")
            logger.info(f"[prepare_dataset] labels type: {type(first_sample['labels'])}")
            logger.info(f"[prepare_dataset] labels length: {len(first_sample['labels'])}")
            logger.info(f"[prepare_dataset] labels sample: {first_sample['labels'][:10]}...")
            
        if len(self.dataset['train']) > 1:
            lengths = [len(self.dataset['train'][i]['input_ids']) for i in range(min(5, len(self.dataset['train'])))]
            logger.info(f"[prepare_dataset] 前5个样本的input_ids长度: {lengths}")
    
    def get_training_arguments(self) -> TrainingArguments:
        """获取训练参数"""
        train_config = self.config["training"]
        
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        report_to = []
        if self.config["logging"].get("use_tensorboard", False):
            report_to.append("tensorboard")
        
        training_args = TrainingArguments(
            output_dir=train_config["output_dir"],
            num_train_epochs=train_config["num_train_epochs"],
            per_device_train_batch_size=train_config["per_device_train_batch_size"],
            per_device_eval_batch_size=train_config["per_device_train_batch_size"],
            gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
            learning_rate=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"],
            lr_scheduler_type=train_config["lr_scheduler_type"],
            warmup_steps=train_config["warmup_steps"],
            logging_steps=train_config["logging_steps"],
            save_strategy=train_config["save_strategy"],
            save_steps=train_config["save_steps"],
            save_total_limit=train_config["save_total_limit"],
            evaluation_strategy=train_config.get("evaluation_strategy") or train_config.get("eval_strategy", "no"),
            eval_steps=train_config["eval_steps"],
            fp16=train_config.get("fp16", False),
            bf16=train_config.get("bf16", True),
            gradient_checkpointing=train_config.get("gradient_checkpointing", True),
            logging_dir=self.config["logging"]["tensorboard_dir"],
            report_to=report_to,
            ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            max_grad_norm=1.0,
        )
        
        return training_args
    
    def train(self):
        """训练"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已清理GPU缓存")
        
        self.setup_gpus()
        self.load_tokenizer()
        self.load_model()
        self.load_dataset()
        self.prepare_dataset()
        
        training_args = self.get_training_arguments()
        
        if self.tokenizer.pad_token is None:
            logger.warning("Tokenizer pad_token 未设置，强制设置为 eos_token")
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer without eos_token, no pad_token")
        
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError("no pad_token_id")
        
        if hasattr(self.tokenizer, 'init_kwargs'):
            self.tokenizer.init_kwargs['pad_token'] = self.tokenizer.pad_token
        
        logger.info(f"Tokenizer pad_token: {self.tokenizer.pad_token}")
        logger.info(f"Tokenizer pad_token_id: {self.tokenizer.pad_token_id}")
        
        if self.tokenizer.pad_token is None:
            raise ValueError("无法设置 tokenizer 的 pad_token, check tokenizer")
        
        class CustomDataCollatorForCausalLM:
            """基于因果语言模型数据整理器"""
            
            def __init__(self, tokenizer, pad_to_multiple_of=None):
                self.tokenizer = tokenizer
                self.pad_to_multiple_of = pad_to_multiple_of
                self.call_count = 0
            
            def __call__(self, features):
                import torch
                
                debug_enabled = self.call_count < 3
                
                if debug_enabled:
                    logger.info(f"[DataCollator 第{self.call_count+1}次调用] 收到 {len(features)} 个样本")
                
                input_ids_list = [f['input_ids'] for f in features]
                labels_list = [f['labels'] for f in features]
                
                max_length = max(len(ids) for ids in input_ids_list)
                
                if self.pad_to_multiple_of is not None:
                    max_length = ((max_length + self.pad_to_multiple_of - 1) 
                                 // self.pad_to_multiple_of * self.pad_to_multiple_of)
                
                if debug_enabled:
                    lengths = [len(ids) for ids in input_ids_list]
                    logger.info(f"[DataCollator] 样本长度: {lengths}")
                    logger.info(f"[DataCollator] padding到最大长度: {max_length}")
                
                padded_input_ids = []
                padded_labels = []
                attention_mask = []
                
                pad_token_id = self.tokenizer.pad_token_id
                
                for input_ids, labels in zip(input_ids_list, labels_list):
                    padding_length = max_length - len(input_ids)
                    padded_input_ids.append(input_ids + [pad_token_id] * padding_length)
                    padded_labels.append(labels + [-100] * padding_length)
                    attention_mask.append([1] * len(input_ids) + [0] * padding_length)
                
                batch = {
                    'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'labels': torch.tensor(padded_labels, dtype=torch.long),
                }
                
                if debug_enabled:
                    logger.info(f"[DataCollator] ✓ padding成功")
                    logger.info(f"[DataCollator] batch keys: {batch.keys()}")
                    logger.info(f"[DataCollator] input_ids shape: {batch['input_ids'].shape}")
                    logger.info(f"[DataCollator] labels shape: {batch['labels'].shape}")
                    logger.info(f"[DataCollator] attention_mask shape: {batch['attention_mask'].shape}")
                
                self.call_count += 1
                return batch
        
        data_collator = CustomDataCollatorForCausalLM(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8,
        )
        
        logger.info("验证模型参数状态...")
        trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"可训练参数: {trainable_params_count:,} / {total_params_count:,} ({100 * trainable_params_count / total_params_count:.2f}%)")
        
        if trainable_params_count == 0:
            logger.error("没有可训练参数！列出所有参数的梯度状态：")
            for name, param in self.model.named_parameters():
                logger.error(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
            raise RuntimeError("没有可训练参数，无法进行训练！")
        
        self.model.train()
        
        metrics_file = self.config["logging"].get("metrics_file", "./output_full_finetune/training_metrics.json")
        logger.info(f"训练指标将保存到: {metrics_file}")
        metrics_callback = MetricsCallback(output_file=metrics_file)
        logger.info("✓ MetricsCallback已创建")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=data_collator,
            callbacks=[metrics_callback],
        )
        
        logger.info(f"✓ Trainer已创建，使用 {len(trainer.callback_handler.callbacks)} 个callbacks")
        
        logger.info("=" * 60)
        logger.info("开始训练 - 全量微调模式")
        logger.info("=" * 60)
        train_result = trainer.train()
        
        logger.info(f"训练完成，检查指标文件...")
        import os
        if os.path.exists(metrics_file):
            file_size = os.path.getsize(metrics_file)
            logger.info(f"✓ 指标文件已保存: {metrics_file} ({file_size} bytes)")
        else:
            logger.warning(f"⚠️  指标文件不存在: {metrics_file}")
        
        metrics = train_result.metrics
        
        logger.info("保存模型...")
        output_dir = Path(training_args.output_dir) / "final_model"
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        logger.info("=" * 60)
        logger.info(f"最终模型输出: {output_dir}")
        logger.info("=" * 60)
        
        return metrics


if __name__ == "__main__":
    trainer = QwenTrainerFullFinetune(config_path="config_full_finetune.yaml")
    trainer.train()

