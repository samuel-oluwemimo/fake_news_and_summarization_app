import pandas as pd
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from torch.nn import CrossEntropyLoss

# --- 1. Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'FacebookAI/roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# --- 2. Load Data using Pandas ---
# Login using e.g. `huggingface-cli login` to access this dataset
try:
    fakenews_train_df = pd.read_parquet("hf://datasets/hruthik07/FakeNewsDataset/data/train-00000-of-00001.parquet")
    fakenews_validation_df = pd.read_parquet("hf://datasets/hruthik07/FakeNewsDataset/data/test-00000-of-00001.parquet")
except Exception as e:
    print(f"Could not load data from Hugging Face Hub. Make sure you are logged in. Error: {e}")
    exit()


# --- 3. Preprocessing and Utility Functions ---
# Define label mappings
id2label = {0: 'fake', 1: 'real'}
label2id = {'fake': 0, 'real': 1}

# This function now correctly maps string labels to integer IDs (0 or 1)
def map_labels(example):
    # Map 'fake' to 0 and 'real' to 1
    label = example['label']
    if label in ['real', 'true', 'half-true', 'mostly-true']:
        return {'labels': 1}
    else:
        return {'labels': 0}

# This function tokenizes a BATCH of examples, which is much faster
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=128)

# This function computes metrics from evaluation predictions
def compute_classification_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 4. Prepare Datasets using `datasets` library ---
# Convert pandas DataFrames to Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(fakenews_train_df)
validation_dataset = Dataset.from_pandas(fakenews_validation_df)

# Apply the label mapping
print("Mapping labels to integers...")
train_dataset = train_dataset.map(map_labels)
validation_dataset = validation_dataset.map(map_labels)

# Apply tokenization in batches
print("Mapping tokenization function to datasets...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True)

# Remove original text columns and set format to PyTorch tensors
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text', 'label']).with_format("torch")
tokenized_validation_dataset = tokenized_validation_dataset.remove_columns(['text', 'label',]).with_format("torch")


# --- 5. Model and Trainer Configuration ---
model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
).to(device)

# Calculate class weights for imbalanced data
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_dataset['labels']),
    y=train_dataset['labels']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Custom Trainer with weighted loss
class CustomTrainer(Trainer):
    # Corrected signature for compute_loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch", # Use 'eval_strategy' instead of deprecated 'evaluation_strategy'
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_classification_metrics,
)

# --- 6. Train and Evaluate ---
if __name__ == "__main__":
    print("Starting classification training... 🚀")
    trainer.train()

    print("\nEvaluating classification results after training: 📊")
    trainer.evaluate()

    # Save the final model
    print("Saving classification model and tokenizer... 💾")
    save_path = r'models\fake_news_classifier'
    trainer.save_model(f'{save_path}\\classification_model')
    tokenizer.save_pretrained(f'{save_path}\\classification_tokenizer')
    print(f"Model and tokenizer saved to {save_path}\\classification_model and {save_path}\\classification_tokenizer")