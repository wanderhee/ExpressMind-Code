# ExpressMind无监督训练使用指南

## 📋 概述

本论文提供了两种训练方式：
1. **LoRA微调**：只训练少量参数，显存占用小，训练速度快
2. **预训练**：训练所有参数，效果更好但显存占用大

## 🆕 预训练新增文件

```
├── config_pretraining.yaml          # 预训练配置文件
├── main_pretraining.py              # 预训练主程序
├── src/
│   └── trainer_pretraining.py       # 预训练训练器
└── README_pretraining.md                 # 本文档
```

## 🚀 预训练开始

### 1. 环境准备

确保已安装所有依赖：

```bash
conda activate pyt39
pip install -r requirements.txt
```

### 2. 配置文件

**关键配置差异**（`config_pretraining.yaml` vs `config.yaml`）：

```yaml

# 预训练配置
training:
  output_dir: "./output_pretraining"  
  num_train_epochs: 3                 
  learning_rate: 5.0e-6                
  
lora:
  enabled: false  
```

### 3. 运行预训练

```bash

python main_pretraining.py --step check_gpu

python main_pretraining.py --step process

python main_pretraining.py --step prepare

python main_pretraining.py --step train

python main_pretraining.py --step eval

python main_pretraining.py --step inference

python main_pretraining.py --step plot
```

### 4. 输出文件

预训练的输出文件独立存放：

```
output_pretraining/
├── final_model/                   
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── ...
├── training_metrics.json           
├── training_curves.png             
└── checkpoint-XXX/                

training_pretraining.log          
```

## ⚙️ 配置调优建议
```

### 使用DeepSpeed

预训练

1. 创建 `ds_config_zero2.json`：

```json
{
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 32,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": 1
}
```

2. 在配置文件中启用：

```yaml
# config_pretraining.yaml
training:
  deepspeed: "./ds_config_zero2.json"
```

3. 使用DeepSpeed启动：

```bash
deepspeed --num_gpus=3 main_pretraining.py --step train
```

### 学习率

```yaml
training:
  learning_rate: 5.0e-6    
  # 1.0e-5                 
  # 2.0e-6                 
  # 1.0e-6                 
```


### 预训练模型加载

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./output_pretraining/final_model")
```



