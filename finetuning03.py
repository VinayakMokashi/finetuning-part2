from peft import get_peft_model, LoraConfig, PeftType
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch 
import numpy as np


class PEFTSentimentClassifier:
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.tokenizer = self.setup_tokenizer()
        self.base_model = self.setup_base_model()
        self.peft_config = self.setup_peft_config()
        self.model = self.setup_peft_model()
        self.dataset = self.load_dataset()
        self.train_subset = self.create_train_subset()
        self.test_subset = self.create_test_subset()
        self.tokenized_train = self.tokenize_dataset(self.train_subset)
        self.tokenized_test = self.tokenize_dataset(self.test_subset)
        self.training_args = self.setup_training_args()
        self.trainer = self.setup_trainer()

    def setup_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)
    
    def setup_base_model(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
    
    def setup_peft_config(self):
        return LoraConfig(
            peft_type=PeftType.LORA,
            task_type="SEQ_CLS",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
    
    def setup_peft_model(self):
        return get_peft_model(self.base_model, self.peft_config)
    
    def load_dataset(self):
        return load_dataset("stanfordnlp/imdb")
    
    def create_train_subset(self):
        return self.dataset["train"].shuffle(seed=42).select(range(500))
    
    def create_test_subset(self):
        return self.dataset["test"].shuffle(seed=42).select(range(100))
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)
    
    def tokenize_dataset(self, dataset):
        return dataset.map(self.tokenize_function, batched=True)
    
    def setup_training_args(self):
        return TrainingArguments(
            output_dir="./peft_results",
            eval_strategy="epoch",
            learning_rate=1e-4,
            per_device_train_batch_size=8,
            num_train_epochs=1,
        )
    
    def setup_trainer(self):
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_test,
        )
    
    def evaluate_model(self):
        return self.trainer.evaluate()
    
    def train_model(self):
        self.trainer.train()

    def evaluate_loss(self):
        model = self.trainer.model
        model.eval()
        eval_dataloader = self.trainer.get_eval_dataloader()
        total_loss = 0
        num_examples = 0
        with torch.no_grad():
            for inputs in eval_dataloader:
                labels = inputs.pop("labels")
                batch_size = labels.shape[0]

                outputs = model(**inputs)
                logits = outputs.logits

                logits_np = logits.view(-1, model.config.num_labels).cpu().numpy()
                labels_np = labels.view(-1).cpu().numpy()

                loss = self.cross_entropy_loss(logits_np, labels_np)
                total_loss += loss * batch_size

                num_examples += batch_size
        
        return total_loss / num_examples

    def cross_entropy_loss(self, logits, labels):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        if len(labels.shape) == 1:
            labels_one_hot = np.zeros((batch_size, num_classes))
            labels_one_hot[np.arange(batch_size), labels] = 1
        else:
            labels_one_hot = labels
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        eps = 1e-12  
        log_probs = np.log(probs + eps)
        loss = -np.sum(labels_one_hot * log_probs) / batch_size
        return loss


if __name__ == "__main__":
    peft_classifier = PEFTSentimentClassifier()
    peft_classifier.model.print_trainable_parameters()

    loss_before = peft_classifier.evaluate_loss()
    peft_classifier.train_model()
    loss_after = peft_classifier.evaluate_loss()

    print(f"eval loss before fine-tuning: {loss_before:.4f}")
    print(f"eval loss after  fine-tuning: {loss_after:.4f}")
    print(f"improvement: {loss_before - loss_after:.4f}")
