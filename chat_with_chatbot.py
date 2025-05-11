# chat.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

# ========== 推理配置 ==========
CONFIG = {
    "base_model": "Langboat/mengzi-t5-base",
    "lora_path": "D:\python all\pythonfile\Ecommerce_chatbot\saved_model",
    "max_input_length": 512,
    "max_new_tokens": 128,
    "temperature": 0.8,  # 增加多样性
    "top_p": 0.95,  # 核采样
    "repetition_penalty": 1.2,  # 抑制重复
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def load_model():
    # 加载基础模型
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["base_model"])
    base_model = T5ForConditionalGeneration.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.float16 if "cuda" in CONFIG["device"] else torch.float32
    )

    # 加载LoRA权重
    model = PeftModel.from_pretrained(
        base_model,
        CONFIG["lora_path"],
        torch_dtype=torch.float16
    )
    return model.to(CONFIG["device"]), tokenizer


def generate_response(model, tokenizer, user_input):
    input_text = f"客户: {user_input} -> 客服: "
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=CONFIG["max_input_length"],
        truncation=True
    ).to(CONFIG["device"])

    outputs = model.generate(
        **inputs,
        max_new_tokens=CONFIG["max_new_tokens"],
        temperature=CONFIG["temperature"],
        top_p=CONFIG["top_p"],
        repetition_penalty=CONFIG["repetition_penalty"],
        do_sample=True,
        num_beams=3,
        early_stopping=True
    )

    # 提取客服回复部分
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_response.split("-> 客服: ")[-1].strip()


def chat_loop():
    model, tokenizer = load_model()
    model.eval()

    print("=== 电商客服对话系统 ===")
    print("输入 '退出' 结束对话")

    while True:
        try:
            user_input = input("用户: ").strip()
            if user_input.lower() in ["退出", "exit"]:
                break

            response = generate_response(model, tokenizer, user_input)
            print(f"客服: {response}\n")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    chat_loop()