import sys
from pprint import pprint

import hydra
from omegaconf import DictConfig
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from sft.code.prepare_data import AlpacaEvalDataset
from sft.code.utils import load_model_and_tokenizer

pprint(sys.argv)
sys.argv.pop()


@hydra.main(
    version_base=None, config_path="../config", config_name="qwen_alpaca_trainer"
)
def main(config: DictConfig):
    print(f"Training {config.model_name_or_path} with {config.dataset} dataset")
    model, tokenizer = load_model_and_tokenizer(config)
    dataset = AlpacaEvalDataset(config, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.max_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        learning_rate=config.learning_rate,
        report_to="wandb",  # logs to wandb
        run_name=config.run_name,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        deepspeed=config.deepspeed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
