# ExpressMind: Reinforcement Learning Library for Highway Incident Management

This project focuses on optimizing the Chain-of-Thought (CoT) reasoning capabilities of Large Language Models (LLMs) for highway emergency response scenarios. By utilizing the **GRPO** (Group Relative Policy Optimization) algorithm and semantic rewards, we aim to enhance the professional logic and structural accuracy of model-generated reports.

---

## 📂 Core Scripts Overview

### 1. Data Pipeline: `Incident_dataset.py`
**Function**: Converts raw highway incident reports (PDF/Docx) into structured reasoning datasets.
* **Multimodal Extraction**: Integrates **PaddleOCR** (offline mode) to handle scanned documents and `pdfplumber` for native text extraction.
* **LLM Distillation**: Uses **DeepSeek-R1-Distill-Llama-70B** as a teacher model to extract specific fields: `scenario`, `root_cause`, `handling_strategy`, and `improvement`.
* **Knowledge Grounding**: The prompt enforces compliance with traffic safety laws and highway maintenance specifications (e.g., JTG H30).

### 2. RL Fine-Tuning: `Grpo_COT.py`
**Function**: Optimizes the model's reasoning trajectory using the **GRPO** algorithm.
* **Training Framework**: Powered by **Unsloth** for 4-bit quantization and LoRA acceleration, significantly reducing VRAM usage.
* **Hybrid Reward Mechanism**:
    * **XML Structure Reward**: Validates if the output strictly follows the required tags (`<root_cause>`, etc.).
    * **Semantic Similarity Reward**: Utilizes `SentenceTransformer` to calculate cosine similarity between generated content and expert ground truth.
* **Experiment Tracking**: Fully integrated with **SwanLab** for real-time visualization of loss, reward trends, and text completions.

### 3. Inference & Validation: `COT_deepseek.py`
**Function**: Performs rapid inference testing for base models (e.g., DeepSeek-V3) or fine-tuned checkpoints.
* **Environment Check**: Verifies model pathing, tokenizer loading, and GPU memory allocation.
* **Template Support**: Automatically applies the correct chat templates for consistent prompt formatting.

---

## 🛠️ Technical Stack

* **RL Framework**: [Unsloth](https://github.com/unslothai/unsloth), [TRL](https://github.com/huggingface/trl)
* **Optimization**: GRPO (Group Relative Policy Optimization)
* **Monitoring**: [SwanLab](https://swanlab.cn/)
* **Text Processing**: PaddleOCR, pdfplumber, python-docx
* **Evaluation**: Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)

---

## 🚀 Quick Start

### Step 1: Build the Dataset
Place your incident reports in the data directory and run the extraction script:
```bash
python Incident_dataset.py
```

### Step 2: Start GRPO Training
Ensure your SwanLab project and model paths are configured:
```bash
python Grpo_COT.py
```
