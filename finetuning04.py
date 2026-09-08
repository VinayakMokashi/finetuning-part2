from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset, load_dataset
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


class FineTuneInstructionModel:
    def __init__(self):
        self.seed = 42
        # Every weight here is pretrained, so unlike finetuning03.py there is no
        # random head to pin down. Seeding up front still makes dropout and batch
        # order reproducible from the first line rather than from the point where
        # Trainer seeds them.
        set_seed(self.seed)
        self.model_name = "t5-small"
        self.dataset_name = "tatsu-lab/alpaca"
        self.train_size = 2000
        self.eval_size = 200
        self.max_input_length = 256
        self.max_target_length = 64
        self.tokenizer = self.setup_tokenizer()
        self.model = self.setup_model()
        self.data_collator = self.setup_data_collator()
        self.dataset = self.load_dataset()
        self.train_dataset = self.create_train_subset()
        self.eval_dataset = self.create_eval_subset()
        self.tokenized_train = self.tokenize_dataset(self.train_dataset)
        self.tokenized_eval = self.tokenize_dataset(self.eval_dataset)
        self.training_args = self.setup_training_args()
        self.trainer = self.setup_trainer()

    def setup_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def setup_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def setup_data_collator(self):
        # Pads each batch to its own longest sequence, and pads the labels with
        # -100 so those positions are excluded from the loss.
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
        )

    def load_dataset(self):
        # 52k instruction / input / output rows. Shuffled once with a fixed seed
        # so the train and eval slices below are disjoint and reproducible.
        return load_dataset(self.dataset_name)["train"].shuffle(seed=42)

    def load_dataset_csv(self):
        # The twelve hand-written rows this script started from. Far too small to
        # teach instruction following, but useful as a fast smoke test: point
        # load_dataset() here to exercise the whole pipeline in seconds.
        csv_path = Path(__file__).parent / "finetune_instruction_data.csv"
        return Dataset.from_pandas(pd.read_csv(csv_path))

    def create_train_subset(self):
        return self.dataset.select(range(self.train_size))

    def create_eval_subset(self):
        # Held out: these rows are never trained on.
        return self.dataset.select(range(self.train_size, self.train_size + self.eval_size))

    def build_prompt(self, instruction, input_text):
        # Roughly half of Alpaca's rows have no input at all, so the Input line is
        # dropped entirely rather than left dangling and empty.
        if input_text and input_text.strip():
            return f"Instruction: {instruction}\nInput: {input_text}"
        return f"Instruction: {instruction}"

    def preprocess_function(self, examples):
        inputs = [
            self.build_prompt(inst, inp)
            for inst, inp in zip(examples["instruction"], examples["input"])
        ]
        model_inputs = self.tokenizer(
            inputs, truncation=True, max_length=self.max_input_length
        )
        model_inputs["labels"] = self.tokenizer(
            examples["output"], truncation=True, max_length=self.max_target_length
        )["input_ids"]
        return model_inputs

    def tokenize_dataset(self, dataset):
        return dataset.map(
            self.preprocess_function, batched=True, remove_columns=dataset.column_names
        )

    def setup_training_args(self):
        return TrainingArguments(
            output_dir="./instruction_result",
            eval_strategy="epoch",
            learning_rate=3e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_strategy="no",
        )

    def setup_trainer(self):
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_eval,
            data_collator=self.data_collator,
        )

    def evaluate_loss(self):
        model = self.trainer.model
        model.eval()

        def compute_loss(logits, labels):
            shift_logits = logits.view(-1, logits.size(-1)).cpu().numpy()
            shift_labels = labels.view(-1).cpu().numpy()
            num_classes = shift_logits.shape[1]
            batch_size = shift_logits.shape[0]

            labels_one_hot = np.zeros((batch_size, num_classes))
            valid_indices = shift_labels != -100
            labels_one_hot[np.arange(batch_size)[valid_indices], shift_labels[valid_indices]] = 1

            shifted_logits = shift_logits - np.max(shift_logits, axis=1, keepdims=True)
            exp_logits = np.exp(shifted_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            log_probs = np.log(probs + 1e-12)

            loss = -np.sum(labels_one_hot * log_probs) / np.sum(valid_indices)
            return loss

        eval_dataloader = DataLoader(
            self.tokenized_eval,
            batch_size=4,
            shuffle=False,
            collate_fn=self.data_collator,
        )

        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)

                decoder_input_ids = model._shift_right(labels)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids
                )
                total_loss += compute_loss(outputs.logits, labels)
                num_batches += 1

        return total_loss / num_batches

    def generate(self, prompts, max_new_tokens=48):
        model = self.trainer.model
        model.eval()
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def sample_prompts(self, count=4):
        rows = self.eval_dataset.select(range(count))
        prompts = [self.build_prompt(r["instruction"], r["input"]) for r in rows]
        references = [r["output"] for r in rows]
        return prompts, references

    def train_model(self):
        self.trainer.train()


def show_generations(title, prompts, outputs, references):
    print(f"\n=== {title} ===")
    for prompt, output, reference in zip(prompts, outputs, references):
        print(f"\nprompt    : {prompt[:160]}")
        print(f"generated : {output[:160]}")
        print(f"reference : {reference[:160]}")


if __name__ == "__main__":
    model_trainer = FineTuneInstructionModel()
    prompts, references = model_trainer.sample_prompts()

    before = model_trainer.generate(prompts)
    eval_loss_before = model_trainer.evaluate_loss()

    model_trainer.train_model()

    after = model_trainer.generate(prompts)
    eval_loss_after = model_trainer.evaluate_loss()

    show_generations("BEFORE fine-tuning", prompts, before, references)
    show_generations("AFTER fine-tuning", prompts, after, references)

    print(f"\neval loss before fine-tuning: {eval_loss_before:.4f}")
    print(f"eval loss after  fine-tuning: {eval_loss_after:.4f}")
    print(f"improvement: {eval_loss_before - eval_loss_after:.4f}")
