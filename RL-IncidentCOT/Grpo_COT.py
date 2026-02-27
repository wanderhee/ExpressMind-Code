import re
import torch
import swanlab  # [新增] 引入 SwanLab
from swanlab.integration.huggingface import SwanLabCallback  # [新增] 引入回调
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 1. 配置参数
# ==========================================
# [新增] SwanLab 项目配置
SWANLAB_PROJECT = "Traffic-CoT-GRPO"
SWANLAB_RUN_NAME = "Qwen3-14B-Semantic-Reward-Run1"

MODEL_PATH = "/home/ubuntu1/wangzihe/LLM-Test/output/Qwen3-14B-Full-Traffic"
DATA_FILE = "highway_incident_cot_dataset.json"
OUTPUT_DIR = "output/Qwen-Traffic-GRPO-Semantic"
MAX_SEQ_LENGTH = 4096
LORA_RANK = 64

# 显存分配策略
GPU_MEMORY_UTILIZATION = 0.6
VLLM_GPU_UTILIZATION = 0.3

# ==========================================
# 2. 初始化语义嵌入模型 (全局加载)
# ==========================================
# 使用轻量级模型计算相似度。
# 为了防止显存 OOM，如果显存紧张，可以将 device 设为 'cpu' (速度会变慢但安全)
# 如果显存够用，设为 'cuda'
print("正在加载语义嵌入模型...")
SIMILARITY_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')


# ==========================================
# 3. 定义奖励函数
# ==========================================

def reward_xml_structure(completions, **kwargs):
    """
    奖励函数1：结构完整性
    """
    rewards = []
    pattern = r"<root_cause>.*?</root_cause>\s*<handling_strategy>.*?</handling_strategy>\s*<improvement>.*?</improvement>"
    for completion in completions:
        if re.search(pattern, completion, re.DOTALL):
            rewards.append(1.0)
        else:
            score = 0.0
            if "<root_cause>" in completion: score += 0.2
            if "<handling_strategy>" in completion: score += 0.2
            if "<improvement>" in completion: score += 0.2
            rewards.append(score)
    return rewards


def reward_semantic_similarity(completions, standard_answer, **kwargs):
    """
    奖励函数2：语义相似度
    """
    rewards = []

    # 1. 批量计算生成文本的 Embedding
    completion_embeddings = SIMILARITY_MODEL.encode(completions, convert_to_tensor=True)

    # 2. 批量计算标准答案的 Embedding
    truth_embeddings = SIMILARITY_MODEL.encode(standard_answer, convert_to_tensor=True)

    # 3. 计算余弦相似度
    cosine_scores = util.cos_sim(completion_embeddings, truth_embeddings)

    for i in range(len(completions)):
        # 取对角线上的值
        score = cosine_scores[i][i].item()
        # 可选：缩放分数，例如 score = max(0, (score - 0.3) / 0.7)
        rewards.append(score)

    return rewards


# ==========================================
# 4. 数据处理 (构造 Ground Truth)
# ==========================================

def format_data_for_grpo(example):
    prompt_template = """你是一位高速公路应急处置专家。请根据提供的事故场景，严格按照【分析原因】->【处置策略】->【改进措施】的顺序输出思维链报告。

请务必使用以下XML标签包裹你的回答：
<root_cause>
在此处填写根本原因分析...
</root_cause>
<handling_strategy>
在此处填写详细的处置经过...
</handling_strategy>
<improvement>
在此处填写改进建议...
</improvement>

场景描述：{scenario}
"""
    ground_truth_text = f"""
<root_cause>
{example['root_cause']}
</root_cause>
<handling_strategy>
{example['handling_strategy']}
</handling_strategy>
<improvement>
{example['improvement']}
</improvement>
"""
    return {
        "prompt": prompt_template.format(scenario=example['scenario']),
        "standard_answer": ground_truth_text
    }


# 加载并处理数据
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.map(format_data_for_grpo)

# ==========================================
# 5. 模型加载与配置
# ==========================================

print(f"正在加载模型: {MODEL_PATH} ...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ==========================================
# 6. 配置 SwanLab 回调 [新增]
# ==========================================
swanlab_callback = SwanLabCallback(
    project=SWANLAB_PROJECT,
    experiment_name=SWANLAB_RUN_NAME,
    description="使用GRPO算法优化交通处置CoT，结合语义相似度奖励",
    config={
        "model": "Qwen3-14B",
        "dataset": DATA_FILE,
        "lora_rank": LORA_RANK,
        "max_seq_len": MAX_SEQ_LENGTH,
        "reward_funcs": ["xml_structure", "semantic_similarity"]
    }
)

# ==========================================
# 7. 训练参数配置
# ==========================================

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=3e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=1024,
    max_completion_length=2048,

    use_vllm=True,
    vllm_gpu_memory_utilization=VLLM_GPU_UTILIZATION,
    vllm_device="cuda:0",

    report_to="none",  # [修改] 关闭默认 wandb，完全交给 SwanLab
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reward_xml_structure,
        reward_semantic_similarity
    ],
    args=training_args,
    train_dataset=dataset,
    callbacks=[swanlab_callback]  # [新增] 注册 SwanLab 回调
)

# ==========================================
# 8. 开始训练
# ==========================================
print(f"开始 SwanLab 监控下的 GRPO 训练... (项目: {SWANLAB_PROJECT})")
trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")