# Mengzi-T5-LoRA-E-Commerce Customer Service Chatbot
## 项目简介
  本项目基于 LoRA（Low-Rank Adaptation）技术实现了一个轻量级、高效的电商客服对话系统。通过对中文 T5 模型Langboat/mengzi-t5-base进行低秩自适应微调，在大幅减少训练参数的同时保持了良好的对话生成能力，适用于电商场景的客户咨询自动回复。
# 电商客服对话系统：基于LoRA的高效微调实现

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 项目简介  
本项目基于LoRA（Low-Rank Adaptation）技术实现了一个轻量级、高效的电商客服对话系统。通过对中文T5模型`Langboat/mengzi-t5-base`进行低秩自适应微调，在大幅减少训练参数的同时保持了良好的对话生成能力，适用于电商场景的客户咨询自动回复。

## 核心技术亮点  
### 1. **LoRA高效微调技术**  
- **参数高效训练**：仅训练约0.1%的模型参数（LoRA适配器），相比全量微调节省99%以上的显存占用，支持在消费级GPU或Colab环境中训练。  
- **目标模块优化**：针对T5模型的注意力层（查询矩阵`q`和值矩阵`v`）应用LoRA，在保持模型生成能力的同时降低计算成本。  

### 2. **数据增强与格式处理**  
- **关键词替换增强**：通过预设词典（如`"电脑"→"笔记本电脑"`）对客户/客服文本进行随机替换，30%概率触发增强，提升模型泛化能力。  
- **标准化输入格式**：统一输入格式为`"客户: {问题} -> 客服: "`，明确区分对话角色，引导模型学习条件生成模式。  

### 3. **智能生成策略**  
- **多样化生成控制**：结合温度采样（`temperature=0.8`）、核采样（`top_p=0.95`）和波束搜索（`num_beams=3`），平衡回复的多样性与合理性。  
- **重复抑制**：通过`repetition_penalty=1.2`避免生成重复内容，提升回复流畅度。  

### 4. **工程化设计**  
- **模块化架构**：训练（`train.py`）与推理（`chat.py`）代码分离，支持快速迭代和部署。  

## 项目结构  
```
ecommerce-chatbot/
├── train.py
├── chat.py
├── config.py
├── data/
│   └── customer_service_data.xlsx 
└── saved_model/            # 训练后保存的LoRA适配器权重
```

## 技术栈  
| 工具/框架         | 版本          | 
|-------------------|---------------|
| PyTorch           | ≥2.0          |
| Hugging Face Transformers | ≥4.30 |
| PEFT              | ≥0.7.0        |
| pandas            | ≥2.0          |
| tqdm              | ≥4.60         |
| scikit-learn      | ≥1.3          |

## 安装依赖  
```bash
# 安装Python依赖
pip install torch transformers peft pandas tqdm scikit-learn openpyxl
```

## 使用指南  

### 1. **准备数据集**  
- **数据格式**：Excel文件，需包含两列：  
  - `"【中文】客户对话内容"`：客户问题文本  
  - `"【中文】客服对话内容"`：对应客服回复文本  
- **示例数据**：将数据保存至`data/customer_service_data.xlsx`（路径可在`config.py`中修改）。  

### 2. **训练模型**  
```bash
# 本地训练（修改config.py中的数据路径）
python train.py

# Google Colab训练（自动挂载Drive，需在Colab环境中运行）
直接运行train.py，首次会提示授权挂载Google Drive
```  
- 训练参数配置：在`config.py`中调整`batch_size`（建议4-8）、`epochs`（建议3-5）、`lora_rank`等超参数。  
- 训练完成后，LoRA权重保存至`config.py`中的`save_path`（默认：`saved_model/`）。  

### 3. **启动对话系统**  
```bash
python chat.py
```  
- 输入客户问题（如`"我的订单什么时候发货？"`），按回车获取客服回复。  
- 输入`"退出"`或`"exit"`结束对话。  

## 配置参数说明  
| 参数名              | 说明                                                                 |
|---------------------|----------------------------------------------------------------------|
| `model_name`        | 基础模型名称（默认：`Langboat/mengzi-t5-base`，支持其他T5系列模型） |
| `batch_size`        | 训练批次大小（显存不足时可减小至4）                                 |
| `learning_rate`     | 优化器学习率（建议范围：1e-4 ~ 5e-4）                              |
| `lora_rank`         | LoRA低秩矩阵秩（默认8，越小参数越少，性能可能下降）                 |
| `max_input_length`  | 输入文本最大长度（建议512，根据模型限制调整）                       |
| `max_new_tokens`    | 生成回复最大长度（默认128，避免过长输出）                           |

## 贡献与反馈  
欢迎提交Issue或Pull Request改进项目：  
- **优化方向**：多轮对话支持、自定义数据增强策略、性能指标（BLEU/ROUGE）评估、模型量化部署等。  
- **反馈方式**：在GitHub Issues中描述问题或建议，或发送邮件至`your-email@example.com`。  

## 许可证  
本项目采用MIT许可证，详细见[LICENSE](LICENSE)。  

## 联系我
- 邮箱：kcposke915@163.com

## 致谢  
- 基础模型：[Langboat/mengzi-t5-base](https://huggingface.co/Langboat/mengzi-t5-base)  
- LoRA技术：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)   
```
