import os
import json
import glob
import torch
import pdfplumber
import re
from tqdm import tqdm
from docx import Document
from paddleocr import PaddleOCR
from modelscope import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


# ==========================================
# 1. 工具类：文档读取与 OCR (已修改为离线模式)
# ==========================================

class DocumentProcessor:
    def __init__(self):
        print("Initializing OCR engine (Offline Mode)...")

        # 定义本地模型路径 (请确保这些文件夹存在且包含模型文件)
        model_root = "/home/ubuntu1/wangzihe/LLM-Test/ocr_models"

        det_path = os.path.join(model_root, "ch_PP-OCRv4_det_infer")
        rec_path = os.path.join(model_root, "ch_PP-OCRv4_rec_infer")
        cls_path = os.path.join(model_root, "ch_ppocr_mobile_v2.0_cls_infer")

        # 简单的路径检查
        if not os.path.exists(det_path):
            raise FileNotFoundError(
                f"OCR Detection model not found at: {det_path}\nPlease download and unzip the models first.")

        # 初始化 PaddleOCR
        # 1. 指定 det_model_dir, rec_model_dir, cls_model_dir 实现离线加载
        # 2. 移除了 show_log=False (新版不兼容)
        # 3. use_angle_cls=True 用于纠正文字方向
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            det_model_dir=det_path,
            rec_model_dir=rec_path,
            cls_model_dir=cls_path
        )

    def read_docx(self, file_path):
        """读取 Word 文件内容"""
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())
            return "\n".join(full_text)
        except Exception as e:
            print(f"Error reading docx {file_path}: {e}")
            return ""

    def read_pdf(self, file_path):
        """读取 PDF 内容，优先提取文本，若无文本则使用 OCR"""
        text_content = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # 尝试直接提取文本
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
                    else:
                        pass  # 如果没有提取到文本，后续会触发 OCR 逻辑

            # 如果直接提取的文本太少（可能是扫描件），则调用 OCR
            if len(text_content.strip()) < 50:
                print(f"Text too short, applying OCR for: {os.path.basename(file_path)}")
                result = self.ocr.ocr(file_path, cls=True)
                ocr_text = []
                if result:
                    for idx in range(len(result)):
                        res = result[idx]
                        if res:
                            for line in res:
                                # result 格式: [[[[x1,y1],...], ("text", conf)], ...]
                                ocr_text.append(line[1][0])
                text_content = "\n".join(ocr_text)

            return text_content
        except Exception as e:
            print(f"Error reading pdf {file_path}: {e}")
            return ""


# ==========================================
# 2. 核心类：DeepSeek 分析器 (保持不变)
# ==========================================

class HighwayIncidentAnalyzer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.doc_processor = DocumentProcessor()
        self.load_model()

    def load_model(self):
        print(f"Checking path: {self.model_dir}")
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Path {self.model_dir} does not exist!")

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)

        print("Loading model (bfloat16)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(self.model_dir)
        except:
            pass

    def build_prompt(self, report_content):
        """
        构建防幻觉、强约束的 Prompt
        """

        system_instruction = """你是一个严谨的“高速公路事件报告数据提取与分析专家”。你的任务是读取一份非结构化的事件报告，并将其转换为严格的结构化 COT（思维链）JSON 数据。

### 核心原则（必须严格遵守）：
1. **绝对忠实原文**：所有“事件经过”、“时间地点”、“处置动作”必须严格来源于提供的文本。严禁编造文本中未提及的细节（如天气、具体车牌、人员姓名），如果文本没写，就不要生成相关细节。
2. **事实与法规分离**：
   - 描述事实时：仅使用文本中的信息。
   - 分析原因时：**必须**调用你内部的《中华人民共和国道路交通安全法》、《公路养护技术规范》(JTG H30) 等相关知识，对文本中的事实进行专业定性。
3. **结构化思维链**：必须严格按照 JSON 要求的四个步骤进行思考和输出。

### JSON 输出结构定义：
请输出且仅输出一个 JSON 对象，包含以下四个字段：
1. **scenario** (场景描述): 
   - 详细描述事件发生的时间、地点、涉及车辆/设施、现场状况。
   - 要求：必须是对原文信息的客观概括，不得添加推测性描述。
2. **root_cause** (原因分析): 
   - 结合高速公路管理规范或法律法规，分析事件发生的原因。
   - 格式要求：必须明确引用具体的违规点或物理原因（例如：“根据《道路交通安全法》第XX条，驾驶员未保持安全距离...” 或 “根据《公路养护规范》，路面油污未及时清理...”）。
3. **handling_strategy** (处置策略): 
   - 提取报告中实际执行的处置流程（如：接警 -> 现场防护 -> 清障 -> 恢复通行）。
   - 要求：按时间顺序整理原文中的处置措施，整合成完整的策略段落。
4. **improvement** (不足与改进):
   - 提取报告中提到的经验教训、不足之处或后续改进计划。
   - 注意：如果报告原文完全没有提到不足或改进，请输出 "报告原文未提及后续改进措施"，严禁自行编造改进建议。

### 输出格式：
仅输出标准的 JSON 字符串，不要包含 Markdown 标记（如 ```json），也不要包含任何解释性文字。
"""

        user_content = f"""请分析以下事件报告文件，并严格按照 System Prompt 的要求生成 JSON 数据。

--- 报告原始内容开始 ---
{report_content}
--- 报告原始内容结束 ---

请生成 JSON："""

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    def extract_json_from_response(self, text):
        if "</think>" in text:
            text = text.split("</think>")[-1]
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
        clean_text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean_text)
        except:
            return {"raw_output": text, "error": "JSON parse failed"}

    def analyze_file(self, file_path):
        print(f"Processing: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        content = ""
        if ext == ".docx":
            content = self.doc_processor.read_docx(file_path)
        elif ext == ".pdf":
            content = self.doc_processor.read_pdf(file_path)

        if not content or len(content) < 10:
            return None

        truncated_content = content[:10000]
        messages = self.build_prompt(truncated_content)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=1500,
                do_sample=False,
                temperature=0.1
            )

        generated_text = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        result_json = self.extract_json_from_response(generated_text)

        if "error" not in result_json:
            result_json["source_file"] = os.path.basename(file_path)

        return result_json


# ==========================================
# 3. 主程序逻辑 (保持不变)
# ==========================================

def main():
    MODEL_DIR = "/home/ubuntu1/wangzihe/DeepSeek-R1-Distill-Llama-70B"
    DATA_DIR = "/home/ubuntu1/wangzihe/2025new-strategy"
    OUTPUT_FILE = "highway_incident_cot_dataset.json"

    analyzer = HighwayIncidentAnalyzer(MODEL_DIR)

    all_data = []

    files = []
    for root, dirs, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            if filename.lower().endswith(('.pdf', '.docx')):
                files.append(os.path.join(root, filename))

    print(f"Found {len(files)} documents. Starting processing...")

    for file_path in tqdm(files):
        try:
            result = analyzer.analyze_file(file_path)
            if result and "error" not in result:
                all_data.append(result)
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"CRITICAL ERROR processing {file_path}: {e}")
            continue

    print(f"\nProcessing complete. Saved {len(all_data)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()