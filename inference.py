import argparse
import os
import random
import numpy as np
import torch
from torch.backends import cudnn

from local_config import JAVA_HOME, JAVA_PATH


import dataclasses
import json
import time
from enum import auto, Enum
from typing import List, Any


import gradio as gr
from PIL import Image
from peft import PeftModelForCausalLM
from skimage import io
from torch import nn
from transformers import LlamaTokenizer
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, transforms

from model.lavis import tasks
from model.lavis.common.config import Config
from model.lavis.data.ReportDataset import create_chest_xray_transform_for_inference, ExpandChannels
from model.lavis.models.blip2_models.modeling_llama_imgemb import LlamaForCausalLM

# Activate for deterministic demo, else comment
SEED = 16
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True

# set java path
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_PATH + os.environ["PATH"]
os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.getcwd(), "gradio_tmp")


abnormalities = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    # Used for gradio server
    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += self.sep + " " + role + ": " + message
                else:
                    ret += self.sep + " " + role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def clear(self):
        self.messages = []
        self.offset = 0
        self.skip_next = False

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }

cfg = Config(parse_args())
vis_transforms = create_chest_xray_transform_for_inference(512, center_crop_size=448)
use_img = False
gen_report = True

def init_blip(cfg):
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    # model.cuda()
    model = model.to(torch.device('cpu'))
    return model


def remap_to_uint8(array: np.ndarray, percentiles=None) -> np.ndarray:
    """Remap values in input so the output range is :math:`[0, 255]`.

    Percentiles can be used to specify the range of values to remap.
    This is useful to discard outliers in the input data.

    :param array: Input array.
    :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
        Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
    :returns: Array with ``0`` and ``255`` as minimum and maximum values.
    """
    array = array.astype(float)
    if percentiles is not None:
        len_percentiles = len(percentiles)
        if len_percentiles != 2:
            message = (
                'The value for percentiles should be a sequence of length 2,'
                f' but has length {len_percentiles}'
            )
            raise ValueError(message)
        a, b = percentiles
        if a >= b:
            raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
        if a < 0 or b > 100:
            raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
        cutoff: np.ndarray = np.percentile(array, percentiles)
        array = np.clip(array, *cutoff)
    array -= array.min()
    array /= array.max()
    array *= 255
    return array.astype(np.uint8)


def load_image(path) -> Image.Image:
    """Load an image from disk.

    The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

    :param path: Path to image.
    :returns: Image as ``Pillow`` ``Image``.
    """
    # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
    image = io.imread(path)

    image = remap_to_uint8(image)
    return Image.fromarray(image).convert("L")

def classify_abnormalities(
    logits,
    thresholds: str = None,
    class_map = {"uncertain": 2, "positive": 1, "negative": 0}
    ):
    """
    Classifies logits and summarizes abnormalities into positive, negative, and uncertain categories.

    Parameters:
    - logits (torch.Tensor): Logits tensor of shape (num_abnormalities, num_classes).
    - abnormalities (list of str): Names of the abnormalities in order matching logits rows.
    - thresholds (str or None): Path to threshold JSON file. If None or not found, argmax is used.
    - class_names (list): Class labels (default: ["positive", "negative", "uncertain"]).

    Returns:
    - dict: Dictionary mapping each class label to comma-separated abnormalities.
    """
    # Validate
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    
    assert logits.shape[0] == len(abnormalities), \
        f"Mismatch: logits rows ({logits.shape[0]}) != abnormalities ({len(abnormalities)})"
    
    probabilities = torch.softmax(logits, dim=1).tolist()
    categorized_abnormalities = {cls: [] for cls in class_map.keys()}

    use_thresholds = False
    thresholds_data = {}

    if thresholds and os.path.isfile(thresholds):
        with open(thresholds, 'r') as f:
            thresholds_data = json.load(f)
            use_thresholds = True

    print(f"use_thresholds: {use_thresholds}")
    for i, (abn, probs) in enumerate(zip(abnormalities, probabilities)):
        if abn == "No Finding":
            continue
        if use_thresholds:
            thresholds_abn = thresholds_data.get(abn, {})
            best_cls = None
            best_score = 0
            for cls, cls_idx in class_map.items():
                threshold = thresholds_abn.get(cls, 0.5)
                prob = probs[cls_idx]
                if prob >= threshold and prob > best_score:
                    best_cls = cls
                    best_score = prob
            if best_cls is not None:
                categorized_abnormalities[best_cls].append(abn)
                # categorized_abnormalities["uncertain"].append(abn)
            # else:
            #     # categorized_abnormalities["uncertain"].append(abn)
            #     cls_idx = torch.tensor(probs).argmax().item()
            #     cls = [k for k, v in class_map.items() if v == cls_idx][0]
            #     categorized_abnormalities[cls].append(abn)
        else:
            cls_idx = torch.tensor(probs).argmax().item()
            cls = [k for k, v in class_map.items() if v == cls_idx][0]
            categorized_abnormalities[cls].append(abn)

    # Format output to natural language
    def format_finding_list(findings):
        if not findings:
            return ""
        if len(findings) == 1:
            return findings[0]
        return ", ".join(findings)

    pos_str = format_finding_list(categorized_abnormalities["positive"])
    neg_str = format_finding_list(categorized_abnormalities["negative"])
    unc_str = format_finding_list(categorized_abnormalities["uncertain"])

    return {
        "positive": pos_str,
        "negative": neg_str,
        "uncertain": unc_str
    }

def format_findings_dict(findings_dict):
    segments = []
    pos_str = findings_dict["positive"]
    neg_str = findings_dict["negative"]
    unc_str = findings_dict["uncertain"]

    if pos_str != "":
        segments.append(f"Positive findings: {pos_str}")
    if neg_str != "":
        segments.append(f"Negative findings: {neg_str}")
    if unc_str != "":
        segments.append(f"Uncertain findings: {unc_str}")

    # Join all non-empty segments into one sentence
    findings_string = ". ".join(segments) if segments else "no common findings"

    return findings_string, pos_str, neg_str, unc_str
    
def init_vicuna():
    use_embs = True

    vicuna_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3", use_fast=False, truncation_side="left", padding_side="left")
    lang_model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3", torch_dtype=torch.float16, device_map='auto')
    vicuna_tokenizer.pad_token = vicuna_tokenizer.unk_token

    if use_embs:
        lang_model.base_model.img_proj_layer = nn.Linear(768, lang_model.base_model.config.hidden_size).to(lang_model.base_model.device)
        vicuna_tokenizer.add_special_tokens({"additional_special_tokens": ["<IMG>"]})
    print(f"lora_path: {cfg.config.model.llm.lora_path}")
    lang_model = PeftModelForCausalLM.from_pretrained(lang_model,
                                                      cfg.config.model.llm.lora_path,
                                                      torch_dtype=torch.float16, use_ram_optimized_load=False).half()
    # lang_model = PeftModelForCausalLM.from_pretrained(lang_model, f"checkpoints/vicuna-7b-img-report/checkpoint-11200", torch_dtype=torch.float16, use_ram_optimized_load=False).half()
    return lang_model, vicuna_tokenizer

blip_model = init_blip(cfg)
lang_model, vicuna_tokenizer = init_vicuna()
blip_model.eval()
lang_model.eval()
cp_transforms = Compose([Resize(512), CenterCrop(488), ToTensor(), ExpandChannels()])


def get_response(input_text, dicom):
    global use_img, blip_model, lang_model, vicuna_tokenizer

    if input_text[-1].endswith(".png") or input_text[-1].endswith(".jpg"):
        image = load_image(input_text[-1])
        # cp_image = cp_transforms(image)
        image = vis_transforms(image)
        dicom = input_text[-1].split('/')[-1].split('.')[0]

        blip_model = blip_model.to(torch.device('cuda'))
       
        logits, qformer_embs = blip_model.forward_image(image[None].to(torch.device('cuda')))
        logits = logits.cpu().detach().squeeze(0)
        qformer_embs = qformer_embs.cpu().detach()
        # print(f"cfg.config.model.mhcac.threshold_path: {cfg.config.model.mhcac.threshold_path}")
        classifications = classify_abnormalities(logits, thresholds=cfg.config.model.mhcac.threshold_path)
        findings, pos_str, neg_str, unc_str = format_findings_dict(classifications)

        if gen_report:
            input_text = (
                f"Image information: <IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG><IMG>.\n\nAbnormality information: {findings}\n\nAct as an expert radiologist. Using only the structured abnormality information and the image-derived features above, write the *Findings* section of a chest X-ray report.\n\n- Do not invent findings. Only describe abnormalities explicitly provided in the 'Abnormality information'.\n- Do not repeat the same information using different wording.\n- Use a single, fluent paragraph in formal radiological style.\n- Use cautious and precise language if uncertain abnormalities are present.\n- Avoid enumeration, bullet points, and speculative phrases.\n- The report should reflect the clinical tone and structure of professionally written reports.\n\nReturn only the generated findings text.")
        use_img = True

        blip_model = blip_model.to(torch.device('cpu'))
        # save image embedding with torch
        torch.save(qformer_embs, 'current_chat_img.pt')
        if not gen_report:
            return None

    else:  # free chat
        input_text = input_text
        findings = None

    '''Generate prompt given input prompt'''
    conv.append_message(conv.roles[0], input_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    '''Call vicuna model to generate response'''
    inputs = vicuna_tokenizer(prompt, return_tensors="pt")  # for multiple inputs, use tokenizer.batch_encode_plus with padding=True
    input_ids = inputs["input_ids"].to(lang_model.device)
    # input_ids = inputs["input_ids"].to(torch.device('cpu'))
    # lang_model = lang_model.cuda()
    generation_output = lang_model.generate(
        input_ids=input_ids,
        dicom=[dicom] if dicom is not None else None,
        use_img=use_img,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=300
    )
    # lang_model = lang_model.cpu()

    preds = vicuna_tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    new_pred = preds[0].split("ASSISTANT:")[-1]
    # remove last message in conv
    conv.messages.pop()
    conv.append_message(conv.roles[1], new_pred)
    return new_pred, findings, pos_str, neg_str, unc_str

'''Conversation template for prompt'''
conv = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant."
           "The assistant gives professional, detailed, and polite answers to the user's questions.",
    roles=["USER", "ASSISTANT"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

# Global variable to store the DICOM string
dicom = None

# Function to update the global DICOM string
def set_dicom(value):
    global dicom
    dicom = value


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history


# Function to clear the chat history
def clear_history(button_name):
    global chat_history, use_img, conv
    chat_history = []
    conv.clear()
    use_img = False
    return []  # Return empty history to the Chatbot


def bot(history):
    # You can now access the global `dicom` variable here if needed
    response, findings, pos_str, neg_str, unc_str = get_response(history[-1][0], None)
    print(response)

    # show report generation prompt if first message after image
    if len(history) == 1:
        input_text = f"Act as an expert radiologist. Using only the structured abnormality information and the image-derived features above, write the *Findings* section of a chest X-ray report.\n\n- Do not invent findings. Only describe abnormalities explicitly provided in the 'Abnormality information'.\n- Do not repeat the same information using different wording.\n- Use a single, fluent paragraph in formal radiological style.\n- Use cautious and precise language if uncertain abnormalities are present.\n- Avoid enumeration, bullet points, and speculative phrases.\n- The report should reflect the clinical tone and structure of professionally written reports.\n\nReturn only the generated findings text."

        if findings is not None:
            input_text = f"Image information: (img_tokens). {findings}. {input_text}"
        history.append([input_text, None])

    history[-1][1] = ""
    if response is not None:
        for character in response:
            history[-1][1] += character
            time.sleep(0.01)
            yield history



def build_gradio_interface():
    css = """
        .gradio-container {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif !important;
            font-size: 22px !important;
            line-height: 1.7 !important;
        }

        .gr-markdown-content h1,
        .gr-markdown-content h2 {
            color: #2a2a2a !important;
            font-weight: 700;
            font-size: 22px !important;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 6px;
            margin-bottom: 14px;
        }

        .gr-markdown-content p {
            color: #333 !important;
            font-weight: 500;
            font-size: 18px !important;
            margin-bottom: 12px;
        }

        .gr-button {
            font-size: 20px !important;
            background-color: #0066cc !important;
            color: white !important;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
        }

        .gr-button:hover {
            background-color: #005bb5 !important;
        }

        .gr-textbox textarea {
            background-color: #fdfdfd !important;
            color: #1a1a1a !important;
            font-size: 20px !important;
            font-weight: 500 !important;
            line-height: 1.8 !important;
            border: 1px solid #cccccc !important;
            border-radius: 8px !important;
            padding: 16px !important;
        }

        .gr-box {
            background-color: #f9f9f9 !important;
            border: 1px solid #e2e2e2 !important;
            border-radius: 10px;
            padding: 16px;
            margin-top: 12px;
        }

        .gr-markdown-content h3 {
            font-size: 24px !important;
            color: #2a2a2a !important;
            margin-bottom: 12px !important;
        }

        label {
            color: #222 !important;
            font-weight: 700 !important;
            font-size: 20px !important;
            margin-bottom: 6px;
            display: block;
        }

        #image-upload {
            max-width: 380px !important;
            min-height: 420px !important;
            border: 1px solid #ddd !important;
            background-color: #f9f9f9 !important;
        }

        #report-box textarea {
            font-size: 20px !important;
            font-weight: 500 !important;
            line-height: 1.8 !important;
            min-height: 250px !important;
            padding: 16px !important;
        }

        #findings-box textarea {
            font-size: 20px !important;
            font-weight: 500 !important;
            line-height: 1.7 !important;
            padding: 14px !important;
            background-color: #fafafa !important;
        }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
        # Title and instructions
        gr.Markdown("""
        <h1>🩻 <span style='color:#000000;'>META-CXR</span> Chest X-Ray AI Assistant</h1>
        <p style='color:#000000;'>Upload a chest X-ray and click <strong>'Generate Report'</strong> to extract abnormalities and generate a radiology summary.</p>
        """)

        with gr.Row(equal_height=True):
            image_input = gr.Image(
                label="📁 Upload Chest X-ray",
                type="filepath",
                tool="editor",
                elem_id="image-upload",
                scale=1
            )

            report_box = gr.Textbox(
                label="📝 Radiology Report",
                placeholder="Generated findings report will appear here...",
                lines=12,
                interactive=False,
                show_copy_button=True,
                elem_id="report-box",
                scale=2
            )

        with gr.Box():
            gr.Markdown("### 🧾 Abnormality Findings")

            with gr.Row():
                positive_box = gr.Textbox(
                    label="✅ Positive Findings",
                    lines=4,
                    interactive=False,
                    show_copy_button=True,
                    elem_id="positive-box",
                    scale=1
                )

                negative_box = gr.Textbox(
                    label="❎ Negative Findings",
                    lines=4,
                    interactive=False,
                    show_copy_button=True,
                    elem_id="negative-box",
                    scale=1
                )

                uncertain_box = gr.Textbox(
                    label="❓ Uncertain Findings",
                    lines=4,
                    interactive=False,
                    show_copy_button=True,
                    elem_id="uncertain-box",
                    scale=1
                )

        # findings_box = gr.Textbox(
        #     label="🔍 Detected Abnormalities",
        #     placeholder="Abnormalities will appear here after generation...",
        #     lines=4,
        #     interactive=False,
        #     show_copy_button=True,
        #     elem_id="findings-box"
        # )

        with gr.Row():
            generate_btn = gr.Button("🧠 Generate Report", scale=1)
            clear_btn = gr.Button("🧹 Clear", scale=1)
            download_btn = gr.Button("📄 Download Report", visible=False)

        # --------- Callbacks ----------

        def clear_all():
            conv.clear()
            return None, "", "", "", "", gr.update(visible=False)

        def generate_report(image_path):
            if not image_path:
                return None, "", "", gr.update(visible=False)

            response, findings, pos_str, neg_str, unc_str = get_response((image_path,), None)
            findings_str = findings or "No abnormalities detected."
            return image_path, pos_str, neg_str, unc_str, response, gr.update(visible=True)

        def download_report_text(text):
            path = f"report_{int(time.time())}.txt"
            with open(path, "w") as f:
                f.write(text)
            return path

        # ---------- Wiring ----------

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[image_input, positive_box, negative_box, uncertain_box, report_box, download_btn]
        )

        generate_btn.click(
            fn=generate_report,
            inputs=[image_input],
            outputs=[image_input, positive_box, negative_box, uncertain_box, report_box, download_btn]
        )

        download_btn.click(
            fn=download_report_text,
            inputs=[report_box],
            outputs=[gr.File()]
        )

    return demo


if __name__ == '__main__':
    demo = build_gradio_interface()
    demo.queue()
    demo.launch(share=True)
