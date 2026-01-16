import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, PeftConfig
from trl import SFTTrainer

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype = "float16",
    bnb_4bit_quant_type = "nf4",

)


dataset = load_dataset("csv", data_files="jokes.csv", split="train")
#model_name = "mistralai/Mistral-7B-Instruct-v0.1"
#model_name = "Qwen/Qwen3-4B-Instruct-2507"
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)

print("model loaded")

def preprocess_data(example):
    return tokenizer(
        example["selftext"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

def apply_chat_template(example):
    messages = [
        {"role": "user", "content": f"You are a helpful joke generator that creates funny jokes. Please generate a joke that contains the following words: \"{example['word1']}\" and \"{example['word2']}\""},
        {"role": "assistant", "content": example['selftext']}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt

#new_dataset = dataset.map(apply_chat_template)
#print(new_dataset[0])

tokenized_dataset = dataset.map(preprocess_data, batched=True)

#tokenized_dataset = tokenized_dataset.remove_columns(['id', 'word1', 'word2', 'selftext', ])
print(tokenized_dataset[0])

training_args = TrainingArguments(
    output_dir="./qwen0.5b",
    save_strategy="steps",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   
    num_train_epochs=3, 
    logging_steps=100,
    save_steps=500,
    fp16=True,
    push_to_hub=False,
    save_total_limit=1
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
)


print("starting training")
trainer.train()
print("training complete")