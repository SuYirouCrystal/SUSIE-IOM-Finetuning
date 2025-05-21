import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# def main(config_path: str):
def main(config_path: str, resume_checkpoint: str = None): #Resume
    with open(config_path) as f:
        config = json.load(f)

    model_name = config.get("model_name", "t5-base")
    dataset_path = config.get("dataset_path", "data/processed/train.jsonl")
    output_dir = config.get("output_dir", "models/iom_model")
    num_epochs = config.get("num_train_epochs", 3)
    batch_size = config.get("batch_size", 4)
    lr = config.get("learning_rate", 5e-5)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset = load_dataset('json', data_files=dataset_path, split='train')
    def preprocess(examples):
        return tokenizer(
            examples['source'],
            text_target=examples['target'],
            max_length=512,
            truncation=True
        )

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=100,
        predict_with_generate=False,
        remove_unused_columns=False,
        fp16=False,
        dataloader_pin_memory=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # trainer.train()
    trainer.train(resume_from_checkpoint=resume_checkpoint) #Resume
    trainer.save_model(output_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune T5/BART on IOM QA data')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to finetune_config.json')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    # main(args.config)
    main(args.config, args.resume) #Resume