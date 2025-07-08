import json
from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
import evaluate

# Load and combine dataset splits
def load_split(file):
    with open(file, "r") as f:
        return [json.loads(line) for line in f]

train_data = load_split("dataset_train.json")
val_data = load_split("dataset_val.json")
test_data = load_split("dataset_test.json")

# Convert to HuggingFace Dataset format
raw_dataset = {
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data),
}

# Encode labels
le = LabelEncoder()
le.fit([d["label"] for d in train_data])
for split in raw_dataset:
    raw_dataset[split] = raw_dataset[split].map(lambda x: {"label": le.transform([x["label"]])[0]})

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["prompt"], truncation=True)
raw_dataset = {k: v.map(tokenize, batched=True) for k, v in raw_dataset.items()}

# Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(le.classes_))

# Training config
args = TrainingArguments(
    output_dir="./selector_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs"
)

# Metrics
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save
model.save_pretrained("./selector_model")
tokenizer.save_pretrained("./selector_model")
with open("label_encoder.json", "w") as f:
    json.dump({i: l for i, l in enumerate(le.classes_)}, f)

