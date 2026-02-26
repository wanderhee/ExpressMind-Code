#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"


BASE_VLM_MODEL="Qwen/Qwen3-VL-8B-Instruct"
LOCAL_VLM_DIR="./models/Qwen3-VL-8B-Instruct"


MODEL_PATH="./models/ExpressMind"

OUTPUT_MODEL_DIR="./models/ExpressMind-VL"

mkdir -p ./models

if [ ! -d "$LOCAL_VLM_DIR" ]; then
    echo "===================================================="
    echo "正在从 $HF_ENDPOINT 下载基础模型 $BASE_VLM_MODEL..."
    echo "===================================================="

    huggingface-cli download --resume-download "$BASE_VLM_MODEL" --local-dir "$LOCAL_VLM_DIR"
else
    echo "基础模型已存在于 $LOCAL_VLM_DIR，跳过下载。"
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "===================================================="
    echo "提示: 未在默认路径 $MODEL_PATH 找到微调模型。"
    read -p "输入您微调权重的实际绝对路径: " USER_INPUT_PATH
    if [ -n "$USER_INPUT_PATH" ] && [ -d "$USER_INPUT_PATH" ]; then
        MODEL_PATH="$USER_INPUT_PATH"
        echo "已切换权重路径至: $MODEL_PATH"
    else
        echo "错误: 路径无效，退出。"
        exit 1
    fi
fi

echo "1) 微调挂载合并"
echo "2) 全量权重替换"
read -p "请输入数字 (1/2): " CHOICE

case $CHOICE in
    1)
        MODE="lora"
        ;;
    2)
        MODE="full"
        ;;
    *)
        echo "输入无效，退出。"
        exit 1
        ;;
esac

echo "===================================================="
echo "正在开始执行权重替换/合并任务..."
echo "模式: $MODE"
echo "设备: 默认为 CPU (保证 Linux 虚机显存不会溢出)"
echo "===================================================="

python3 replace_llm_backbone.py \
    --mode "$MODE" \
    --qwen3vl_path "$LOCAL_VLM_DIR" \
    --path "$MODEL_PATH" \
    --output_path "$OUTPUT_MODEL_DIR" \
    --device "cpu"

if [ $? -eq 0 ]; then
    echo "===================================================="
    echo "模型合并成功。"
    echo "合并后的模型已保存至: $OUTPUT_MODEL_DIR"
    echo "您现在可以通过以下命令运行 Web Demo 进行演示:"
    echo "python3 web_demo_mm.py --checkpoint-path $OUTPUT_MODEL_DIR"
    echo "===================================================="
else
    echo "执行过程中出现错误，请检查 Python 依赖及错误信息。"
fi
