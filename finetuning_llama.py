import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")


model = "/leonardo/home/userexternal/mkhan002/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("LLaMA 3 loaded successfully!")

dataset = load_dataset("json", data_files="train.jsonl")


def preprocess_function(examples):
    inputs = [f"Instruction: {instr}\nResponse: {out}" for instr, out in zip(examples["instruction"], examples["output"])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = model_inputs.input_ids.copy()
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Setup LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # typical for LLaMA
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_llama3",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the LoRA weights
model.save_pretrained("./lora_llama3")