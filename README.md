# 🦙 TinyLlama Q&A Assistant – Fine-Tuned Customer Support LLM

This project demonstrates how to fine-tune the TinyLlama 1B model on a custom customer support Q&A dataset using parameter-efficient techniques like LoRA. The result is a lightweight, domain-specific language model capable of answering frequently asked questions with high accuracy.

## 🚀 Project Overview

- Fine-tuned TinyLlama 1B on a structured Q&A dataset (in JSON format)
- Used LoRA (Low-Rank Adaptation) for efficient training on low-resource hardware
- Applied 4-bit quantization using `bitsandbytes` to optimize memory usage
- Trained using HuggingFace's `transformers`, `datasets`, and `PEFT` libraries
- Model can be deployed as a fast, local assistant for customer support scenarios

## 📂 Dataset

- Format: JSON
- Structure: `{ "question": "...", "answer": "..." }`
- Sample data: [train_expanded.json](./train_expanded.json)

## 🧠 Model & Training

- **Base Model:** [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Tokenizer:** HuggingFace tokenizer matching TinyLlama
- **Fine-Tuning Method:** LoRA using PEFT
- **Quantization:** 4-bit (`bnb_4bit`) via bitsandbytes
- **Loss Function:** Causal Language Modeling (CLM)

## 🛠️ Tech Stack

- Python
- HuggingFace Transformers
- HuggingFace Datasets
- PEFT (Parameter-Efficient Fine-Tuning)
- bitsandbytes
- LoRA
- JSON

## 🧪 Example Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained("path/to/finetuned-model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = qa_pipeline("How can I track my order?", max_new_tokens=100)
print(response[0]["generated_text"])
