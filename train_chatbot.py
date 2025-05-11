import torch
import random
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# 配置参数
CONFIG = {
    "model_name": "Langboat/mengzi-t5-base",
    "data_path": "D:\python all\pythonfile\Ecommerce_chatbot\Data\customer_service_data -NotSplit.xlsx",
    "daily_data": "heyyliu/daily_dialog_chinese",
    "save_path": "D:\python all\pythonfile\Ecommerce_chatbot\saved_model",
    "max_input_length": 512,
    "max_target_length": 256,
    "batch_size": 8,          # 可减小至4以降低显存需求
    "epochs": 5,
    "learning_rate": 2e-4,
    "lora_rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
# ========== 数据增强配置 ==========
REPLACE_DICT = {
    "电脑": ["笔记本电脑", "设备"],
    "手机": ["智能手机", "移动设备"],
    "你好": ["您好", "您好呀"]
}


# ========== 数据集类（保留原始结构） ==========
class EcommerceDataset(Dataset):
    def __init__(self, customer_texts, service_texts, tokenizer):
        self.customer_texts = customer_texts
        self.service_texts = service_texts
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.customer_texts)

    def __getitem__(self, idx):
        # 原始数据处理逻辑
        customer = str(self.customer_texts[idx]).strip()
        service = str(self.service_texts[idx]).strip()

        # 数据增强（30%概率触发）
        if random.random() < 0.3:
            customer = self.augment_text(customer)
            service = self.augment_text(service)

        input_text = f"客户: {customer} -> 客服: "
        target_text = service

        input_enc = self.tokenizer(
            input_text,
            max_length=CONFIG["max_input_length"],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        target_enc = self.tokenizer(
            target_text,
            max_length=CONFIG["max_target_length"],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        labels = target_enc["input_ids"].squeeze().clone()
        labels[labels == self.pad_token_id] = -100

        return {
            "encoder_input_ids": input_enc["input_ids"].squeeze(),
            "encoder_attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels
        }

    def augment_text(self, text):
        for word, replacements in REPLACE_DICT.items():
            if word in text:
                text = text.replace(word, random.choice(replacements))
        return text


# ========== LoRA配置 ==========
def setup_lora(model):
    config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["q", "v"],
        lora_dropout=0.05,
        task_type="SEQ_2_SEQ_LM"
    )
    return get_peft_model(model, config)


# ========== 主训练逻辑 ==========
def main():
    # 加载数据（完全保留原始代码）
    df = pd.read_excel(CONFIG["data_path"])
    valid_data = df.dropna(subset=["【中文】客户对话内容", "【中文】客服对话内容"])
    cus = valid_data["【中文】客户对话内容"].tolist()
    serv = valid_data["【中文】客服对话内容"].tolist()

    # 数据划分（原始逻辑不变）
    train_cus, val_cus, train_serv, val_serv = train_test_split(
        cus, serv, test_size=0.2, random_state=42
    )

    # 初始化模型
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"])
    model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"])
    model = setup_lora(model)  # 添加LoRA适配器

    # 数据集（保持原始结构）
    train_dataset = EcommerceDataset(train_cus, train_serv, tokenizer)
    val_dataset = EcommerceDataset(val_cus, val_serv, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

    # 训练循环
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    model.to(CONFIG["device"])

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            inputs = {
                "input_ids": batch["encoder_input_ids"].to(CONFIG["device"]),
                "attention_mask": batch["encoder_attention_mask"].to(CONFIG["device"]),
                "labels": batch["labels"].to(CONFIG["device"])
            }
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 验证
        val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

    # 保存适配器权重
    model.save_pretrained(CONFIG["save_path"])


def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                "input_ids": batch["encoder_input_ids"].to(CONFIG["device"]),
                "attention_mask": batch["encoder_attention_mask"].to(CONFIG["device"]),
                "labels": batch["labels"].to(CONFIG["device"])
            }
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
    return total_loss / len(val_loader)


if __name__ == "__main__":
    main()
