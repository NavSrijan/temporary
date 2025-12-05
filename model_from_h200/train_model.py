import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments, LineByLineTextDataset
from transformers import GPT2TokenizerFast

# 1. Load the Tokenizer we just made
print("Loading Tokenizer...")
tokenizer = GPT2TokenizerFast(
    vocab_file="./waf_tokenizer/vocab.json",
    merges_file="./waf_tokenizer/merges.txt",
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>"
)
# GPT-2 needs a pad token
tokenizer.pad_token = "<pad>"

# 2. Define the "Tiny" GPT-2 Architecture
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,      # Max sequence length (HTTP requests usually fit here)
    n_ctx=512,
    n_embd=256,           # Small embedding dimension (Standard is 768)
    n_layer=4,            # Only 4 layers (Standard is 12)
    n_head=4,             # 4 Attention heads
    activation_function="gelu_new",
)

model = GPT2LMHeadModel(config)
print(f"Model Parameters: {model.num_parameters() / 1_000_000:.2f} Million")

# 3. Prepare the Dataset
# We treat every line in your file as a separate training example
print("Loading Dataset...")
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="train_vx.txt", # TRAIN ONLY ON BENIGN DATA if you want Anomaly Detection!
    block_size=128,                 # Truncate long lines
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, # mlm=False is critical for GPT (Causal LM)
)

# 4. Training Settings
training_args = TrainingArguments(
    output_dir="./waf_model_checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=5,             # Train longer since model is small
    per_device_train_batch_size=32, # Adjust based on GPU VRAM
    save_steps=5000,
    save_total_limit=1,
    learning_rate=3e-4,             # Higher LR for smaller models
    logging_steps=100,
    prediction_loss_only=True,
    remove_unused_columns=False,
)

# 5. Train
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print("Starting Training...")
trainer.train()

# 6. Save Final Model
trainer.save_model("./tiny_gpt2_waf")
tokenizer.save_pretrained("./tiny_gpt2_waf")
print("Model saved to ./tiny_gpt2_waf")
