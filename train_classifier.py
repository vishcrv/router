import json
from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import evaluate
import numpy as np
import torch

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

# Calculate class weights for imbalanced datasets
train_labels = [d["label"] for d in train_data]
class_weights = compute_class_weight('balanced', classes=np.unique(le.transform(train_labels)), 
                                   y=le.transform(train_labels))
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Tokenizer with optimized settings
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    # Use longer max_length for better context capture
    return tokenizer(batch["prompt"], truncation=True, padding=True, max_length=512)

raw_dataset = {k: v.map(tokenize, batched=True) for k, v in raw_dataset.items()}

# Custom model class to handle class weights
class WeightedDistilBertForSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.class_weights = class_weights
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Filter out any unexpected kwargs that might be passed by newer transformers versions
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['num_items_in_batch']}
        
        # Call parent forward method
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, 
                                 labels=labels, **filtered_kwargs)
        
        # Override loss calculation with class weights if provided
        if labels is not None and self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
            loss = loss_fct(outputs.logits, labels)
            outputs.loss = loss
            
        return outputs

# Model with class weights
model = WeightedDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=len(le.classes_),
    class_weights=class_weights
)

# Enhanced training config optimized for A100
args = TrainingArguments(
    output_dir="./selector_model",
    per_device_train_batch_size=64,  # Increased for A100
    per_device_eval_batch_size=64,   # Increased for A100
    gradient_accumulation_steps=2,    # Effective batch size = 128
    num_train_epochs=5,               # More epochs for better convergence
    learning_rate=3e-5,               # Slightly higher LR
    weight_decay=0.01,                # L2 regularization
    warmup_ratio=0.1,                 # Warmup for stable training
    lr_scheduler_type="cosine",       # Cosine decay schedule
    logging_dir="./logs",
    logging_steps=50,
    eval_steps=100,
    save_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    seed=42,                          # Reproducibility
    fp16=True,                        # Mixed precision for A100
    dataloader_num_workers=0,         # Set to 0 to avoid fork issues
    remove_unused_columns=True,       # Remove unused columns to avoid conflicts
    push_to_hub=False,
    report_to=[],                     # Disable wandb/tensorboard completely
    disable_tqdm=False,
    gradient_checkpointing=True,      # Memory optimization
)

# Enhanced metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    
    # Calculate multiple metrics
    acc = accuracy.compute(predictions=preds, references=labels)
    prec = precision.compute(predictions=preds, references=labels, average='weighted')
    rec = recall.compute(predictions=preds, references=labels, average='weighted')
    f1_score = f1.compute(predictions=preds, references=labels, average='weighted')
    
    return {
        'accuracy': acc['accuracy'],
        'precision': prec['precision'],
        'recall': rec['recall'],
        'f1': f1_score['f1']
    }

# Custom data collator for dynamic padding (simplified)
data_collator = DataCollatorWithPadding(tokenizer)

# Trainer with optimizations
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train with error handling and monitoring
print("Starting training...")
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed: {e}")
    raise

# Evaluate on test set
print("Evaluating on test set...")
test_results = trainer.evaluate(eval_dataset=raw_dataset["test"])
print(f"Test Results: {test_results}")

# Save model and tokenizer
print("Saving model...")
model.save_pretrained("./selector_model")
tokenizer.save_pretrained("./selector_model")

# Save enhanced label encoder info
label_info = {
    "id2label": {i: l for i, l in enumerate(le.classes_)},
    "label2id": {l: i for i, l in enumerate(le.classes_)},
    "num_labels": len(le.classes_),
    "class_weights": class_weights.tolist()
}

with open("label_encoder.json", "w") as f:
    json.dump(label_info, f, indent=2)

print("Training and evaluation completed!")
print(f"Model saved to: ./selector_model")
print(f"Label encoder saved to: label_encoder.json")