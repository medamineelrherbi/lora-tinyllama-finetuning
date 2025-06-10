# 🦙 TinyLLaMA Fine-Tuning — Travel Chatbot Project

This project was built as part of my learning journey into **Large Language Models (LLMs)**, where I explored **fine-tuning TinyLLaMA** using **LoRA (Low-Rank Adaptation)** and the Hugging Face `trl` library. The goal was to create a **travel chatbot** capable of understanding user intent and responding accurately, using a real-world dataset.

By working on this, I aimed to gain **practical, hands-on experience** with model training, dataset preprocessing, LoRA parameter tuning, and evaluation — all critical components in modern NLP pipelines.

---

## 🛠️ Project Highlights

- 🔍 **Model:** [TinyLLaMA-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.1) — small, open, and chat-ready LLM.
- 🧾 **Dataset:** [Bitext Travel Chatbot Dataset](https://huggingface.co/datasets/bitext/Bitext-travel-llm-chatbot-training-dataset).
- 🧪 **Technique:** Parameter-efficient fine-tuning using [LoRA](https://arxiv.org/abs/2106.09685).
- 🧰 **Libraries Used:** `transformers`, `trl`, `peft`, `datasets`, `evaluate`.

---

## 🎯 Objective

Fine-tune a pre-trained LLaMA-based model to respond to travel-related queries while learning how to:

- Preprocess and balance real-world intent-based datasets.
- Apply LoRA for lightweight, cost-efficient fine-tuning.
- Generate, decode, and evaluate model predictions using ROUGE.

---

## ⚙️ LoRA & Training Configuration

| Parameter                        | Value         | Reason                                                                 |
|----------------------------------|---------------|------------------------------------------------------------------------|
| `r` (LoRA rank)                  | `16`          | Balanced trade-off between performance and memory usage                |
| `lora_alpha`                    | `32`          | Scales the LoRA updates effectively                                     |
| `lora_dropout`                 | `0.05`        | Helps prevent overfitting                                                |
| `target_modules`               | `['q_proj', 'v_proj']` | Targeting attention projection layers for efficiency            |
| `num_train_epochs`             | `3`           | Enough for convergence on this small dataset                          |
| `per_device_train_batch_size`  | `4`           | Small batch size due to memory constraints                            |
| `gradient_accumulation_steps`  | `2`           | Simulates a batch size of 8 to stabilize training                     |
| `learning_rate`                | `2e-4`        | Learning rate tuned for stable LoRA fine-tuning                       |
| `max_grad_norm`                | `0.3`         | Prevents exploding gradients                                          |
| `save_steps`                   | `500`         | Periodic saving for monitoring or resuming                            |

> ✅ **Why these values?**  
They were chosen based on best practices from LoRA literature, balancing memory, training time, and expected convergence speed — ideal for low-resource environments like Google Colab.

---

## 📊 Evaluation with ROUGE

The model’s performance was evaluated using the **ROUGE** metric with stemming ("running" → "run", "evaluated" → "evaluat") enabled. This allows comparing generated responses to ground truth even if wording differs slightly.  
**ROUGE** A set of metrics used to evaluate the quality of machine-generated text (e.g., summaries, responses) by comparing it to human-written reference ("ground truth") text. 

---

## 🧪 Sample Inference

```python
inputs = tokenizer.encode("Query: where can I find my baggage", return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=25)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```
## ✅ What I Learned
Practical implementation of LoRA fine-tuning on real datasets.  
Model configuration and training lifecycle with Hugging Face tools.  
Evaluation using ROUGE and interpretation of generative outputs.  
Handling small memory environments while still fine-tuning LLMs.  


