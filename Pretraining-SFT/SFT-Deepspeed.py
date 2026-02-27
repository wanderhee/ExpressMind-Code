import os
import json
import random
import pandas as pd
import torch
import torch.distributed as dist
import swanlab
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
import sys

# ================= 配置区域 =================

# 1. 初始化分布式环境
try:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_main_process = (global_rank == 0)
    else:
        global_rank = 0
        world_size = 1
        is_main_process = True
except Exception as e:
    print(f"分布式初始化失败或已由Trainer初始化: {e}")
    global_rank = 0
    is_main_process = True

MODEL_PATH = "/home/ubuntu1/wangzihe/Qwen3-14B"
DATA_PATH = "/home/ubuntu1/wangzihe/LLM-Test/SFT_Trans_shuffled_updated.json"
OUTPUT_DIR = "./output/Qwen3-14B-Full-Traffic"
DEEPSPEED_CONFIG = "./ds_config.json"

MAX_LENGTH = 4096
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUMULATION = 8

# ===========================================

if is_main_process:
    print(f"CUDA 可用性: {torch.cuda.is_available()}")
    print(f"可见的GPU数量: {torch.cuda.device_count()}")
    print(f"当前分布式 Rank: {global_rank}, 总进程数: {world_size}")

# 初始化 SwanLab (仅在主进程)
swanlab_logger = None
if is_main_process:
    try:
        swanlab_logger = swanlab.init(
            project="qwen3-sft-traffic-full",
            experiment_name="qwen3-14b-Full-FT-DS-Eval-400k",
            config={
                "model": "Qwen/Qwen3-14b-Full",
                "data_max_length": MAX_LENGTH,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "grad_acc": GRAD_ACCUMULATION,
                "type": "Full Fine-Tuning + DeepSpeed ZeRO-3"
            },
            log_dir="./swanlab_logs"
        )
    except Exception as e:
        print(f"SwanLab 初始化跳过或失败: {e}")


# ================= 关键修复：PredictionCallback =================
class PredictionCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_df, swanlab_logger, num_samples=3):
        self.tokenizer = tokenizer
        self.eval_df = eval_df
        self.swanlab = swanlab_logger
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """
        在每次评估(evaluate)结束后触发生成测试
        修复说明：在 ZeRO-3 模式下，所有 Rank 必须共同参与 model.generate，
        否则会导致 Rank 0 等待其他 Rank 发送参数而死锁。
        """
        if self.eval_df is None or len(self.eval_df) == 0:
            return

        # 1. 即使是 Rank > 0 也要运行，不要在这里 return!

        # 为了保证所有卡采样的数据一致，我们设置相同的随机种子，或者只处理前 N 条
        # 这里简单起见，我们选取固定的前 num_samples 条，或者所有 rank 都做同样的 sample
        # 注意：如果用 sample(random_state)，需要确保种子一致
        sample_rows = self.eval_df.iloc[:self.num_samples]  # 取前几条固定数据，避免多卡随机性不一致

        if is_main_process:
            print(f"\n>>> Step {state.global_step} 评估结束，开始生成测试样本 (全Rank参与)...")

        # 确保模型在 Eval 模式
        was_training = model.training
        model.eval()

        predictions = []

        # 获取当前设备
        # 在 DeepSpeed 中，model 可能是 DeepSpeedEngine，需要小心处理
        try:
            device = model.device
        except:
            device = torch.device("cuda")

        try:
            for index, row in sample_rows.iterrows():
                sys_prompt = row['Prompt']
                user_input = row['Instruction']

                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_input}
                ]

                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = self.tokenizer([text], return_tensors="pt").to(device)

                with torch.no_grad():
                    # 关键点：所有 Rank 都会运行这一行，ZeRO-3 会自动处理参数收集
                    generated_ids = model.generate(
                        inputs.input_ids,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id,
                        synced_gpus=True  # 显式告诉 HF 这是一个多卡环境 (虽然 ZeRO-3 通常会自动处理)
                    )

                # 只有主进程负责解码和打印
                if is_main_process:
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    print(
                        f"\n--- [Sample Step {state.global_step}] ---\nQuestion: {user_input}\nModel: {response}\n----------------")

                    if self.swanlab:
                        import swanlab as sl
                        predictions.append(
                            sl.Text(f"Q: {user_input}\nA: {response}", caption=f"Step {state.global_step}"))

            # 只有主进程负责上传
            if is_main_process and self.swanlab and predictions:
                self.swanlab.log({"Validation_Prediction": predictions}, step=state.global_step)

        except Exception as e:
            # 打印错误但不要崩溃，方便调试
            print(f"[Rank {global_rank}] 预测生成出错: {e}")

        # 恢复训练模式
        if was_training:
            model.train()

        # 增加一个同步点，防止跑得快的进程提前开始下一轮训练
        if dist.is_initialized():
            dist.barrier()


# ================= 数据处理函数 =================

def pretrain_data_processor(origin_path, new_train_path, new_test_path, test_ratio=0.05, random_seed=42):
    print(f"[Rank {global_rank}] 正在处理数据...")
    random.seed(random_seed)
    data_list = []
    with open(origin_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            try:
                data = json.load(f)
                if isinstance(data, list): data_list = data
            except:
                pass
        else:
            for line in f:
                if not line.strip(): continue
                try:
                    data_list.append(json.loads(line))
                except:
                    continue

    valid_data = [item for item in data_list if 'Prompt' in item and 'Instruction' in item and 'Response' in item]
    random.shuffle(valid_data)
    test_size = max(1, int(len(valid_data) * test_ratio))
    test_data = valid_data[:test_size]
    train_data = valid_data[test_size:]

    with open(new_train_path, "w", encoding="utf-8") as f:
        for item in train_data: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(new_test_path, "w", encoding="utf-8") as f:
        for item in test_data: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return train_data, test_data


# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)


def process_func(example):
    system_text = example['Prompt']
    instruction_text = example['Instruction']
    response_text = example['Response']
    instruction = tokenizer(
        f"<|im_start|>system\n{system_text}<|im_end|>\n<|im_start|>user\n{instruction_text}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False)
    response = tokenizer(f"{response_text}<|im_end|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ================= 主流程 =================

train_json_path = "train_full_traffic.json"
test_json_path = "test_full_traffic.json"

if is_main_process:
    if os.path.exists(DATA_PATH):
        pretrain_data_processor(DATA_PATH, train_json_path, test_json_path)
    else:
        print(f"警告：找不到原始数据 {DATA_PATH}")

if dist.is_initialized():
    dist.barrier()

if os.path.exists(train_json_path):
    train_df = pd.read_json(train_json_path, lines=True)
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
    eval_df = pd.read_json(test_json_path, lines=True)
    eval_ds = Dataset.from_pandas(eval_df)
    eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)
else:
    raise FileNotFoundError("没有训练数据")

if is_main_process:
    print("正在加载模型...")

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                             use_cache=False)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    logging_steps=10,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="swanlab",
    run_name="Qwen3-14B-Full-Traffic-DS-400k",
    optim="adamw_torch",
    bf16=True,
    ddp_find_unused_parameters=False,
    deepspeed=DEEPSPEED_CONFIG,
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=3,
    dataloader_num_workers=4
)

pred_callback = PredictionCallback(tokenizer, eval_df, swanlab_logger, num_samples=3)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[pred_callback]
)

if is_main_process:
    print("开始训练...")
trainer.train()

final_save_path = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_save_path)
if is_main_process:
    tokenizer.save_pretrained(final_save_path)