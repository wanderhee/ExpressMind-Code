
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image
import argparse

def load_vpa_model(model_path):
    """加载VPA增强的模型"""
    print(f"加载VPA模型: {model_path}")
    
   
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
   
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, processor

def vpa_inference(image, prompt, model, processor):
    """使用VPA模型进行推理"""
    
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)
    
   
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    return response

def create_demo(model_path):
    """创建Gradio界面"""
    
    model, processor = load_vpa_model(model_path)
    
    
    example_prompts = [
        ["请检测图中的交通异常事件", "traffic_event"],
        ["分析天气状况对交通的影响", "weather_impact"],
        ["描述视频中的车辆行为", "vehicle_behavior"],
        ["生成交通事件报告", "incident_report"]
    ]
    
    
    with gr.Blocks(title="ExpressMind-VL with VPA 演示") as demo:
        gr.Markdown("# 🚀 ExpressMind-VL with VPA 演示")
        gr.Markdown("### 视觉优先对齐的多模态高速公路大模型")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传图像/视频帧")
                prompt_input = gr.Textbox(
                    label="输入提示",
                    value="请分析图中的交通状况",
                    placeholder="请输入您的提示..."
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 分析", variant="primary")
                    clear_btn = gr.Button("🗑️ 清空")
                
                gr.Examples(
                    examples=example_prompts,
                    inputs=[prompt_input, gr.State()],
                    label="示例提示"
                )
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="分析结果",
                    lines=10,
                    placeholder="模型将在这里生成分析报告..."
                )
        
        def process(image, prompt):
            if image is None:
                return "请先上传图像"
            return vpa_inference(image, prompt, model, processor)
        
        def clear_all():
            return None, "", ""
        
        submit_btn.click(
            fn=process,
            inputs=[image_input, prompt_input],
            outputs=output_text
        )
        
        clear_btn.click(
            fn=clear_all,
            outputs=[image_input, prompt_input, output_text]
        )
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    demo = create_demo(args.model_path)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )