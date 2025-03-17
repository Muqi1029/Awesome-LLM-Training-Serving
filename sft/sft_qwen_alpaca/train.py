import hydra
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from sft.code.prepare_data import get_dataloader
from sft.code.utils import load_model_and_tokenizer


@hydra.main(config_path="../config", config_name="qwen_alpaca.yaml")
def main(config: DictConfig):
    model, tokenizer = load_model_and_tokenizer(config)
    dataloader = get_dataloader(config, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.max_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_dir=config.logging_dir,
        report_to="wandb",
        run_name=config.run_name,
        learning_rate=config.learning_rate,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataloader,
    )
    trainer.train()


if __name__ == "__main__":
    main()
