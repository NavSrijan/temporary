import argparse
import os
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./roberta_waf")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--from_scratch", action="store_true")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)

    print("Loading dataset...")
    dataset = load_dataset(
        "text",
        data_files={"train": args.train_file}
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_len
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )

    print("Setting up config...")
    config = RobertaConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.max_len,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )


    if args.from_scratch:
        print("Initializing RoBERTa from scratch...")
        model = RobertaForMaskedLM(config)
    else:
        print("Loading pretrained RoBERTa and resizing embeddings...")
        model = RobertaForMaskedLM.from_pretrained("roberta-base")
        model.resize_token_embeddings(len(tokenizer))

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    print("Setting training args...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=collator
    )

    print("Training...")
    trainer.train()

    print("Saving model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Done. Model saved at:", args.output_dir)


if __name__ == "__main__":
    main()

