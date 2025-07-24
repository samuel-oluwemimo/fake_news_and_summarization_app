import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = 'google/pegasus-cnn_dailymail'
tokenizer = AutoTokenizer.from_pretrained(model)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)
dataset_bbc = load_from_disk('../../data/bbc_news_summary/gopalkalpande_bbc_news_summary_raw/train')


def convert_examples_to_features(examples):
    input_encodings = tokenizer(examples['Articles'], truncation=True, padding="max_length", max_length=128)
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(examples['Summaries'], truncation=True, padding="max_length", max_length=128)
    return{
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
    }

dataset_bbc_pro = dataset_bbc.map(convert_examples_to_features, batched=True)
seq2seqcollator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

trainer_args = TrainingArguments(
            output_dir='pegasus-bbc',
            num_train_epochs=2,
            warmup_steps=500,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy='no',
            eval_steps=500,
            save_steps=1e6, # Saves only at the end
            gradient_accumulation_steps=16
        )

trainer = Trainer(
    model=model_pegasus,
    args=trainer_args,
    train_dataset=dataset_bbc_pro,
    data_collator=seq2seqcollator
)

if __name__ == "__main__":
    print("👾 ...Starting summarization training... 🚀")
    trainer.train()

    save_path = 'models/text_summarizer'

    print("Saving summarization model and tokenizer... 💾")
    model_pegasus.save_pretrained(f'{save_path}/model_bbc_pegasus')
    tokenizer.save_pretrained(f'{save_path}/tokenizer')



