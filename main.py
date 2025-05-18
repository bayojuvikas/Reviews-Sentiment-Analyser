import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Combine Review_Title and Review_Text
train_df['text'] = train_df['Review_Title'].fillna('') + " " + train_df['Review_Text'].fillna('')
test_df['text'] = test_df['Review_Title'].fillna('') + " " + test_df['Review_Text'].fillna('')

# Map sentiment to binary labels
train_df['label'] = train_df['sentiment'].map({'Positive': 1, 'Negative': 0})

# Train-validation split
train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=42)

# Convert to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_data[['text', 'label']])
val_dataset = Dataset.from_pandas(val_data[['text', 'label']])

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=256)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2,
    id2label={0: "Negative", 1: "Positive"},
    label2id={"Negative": 0, "Positive": 1}
).to(device)

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds)
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=12,
    logging_dir='./logs',
    logging_steps=50
)

# Initialize Trainer (⚠️ `tokenizer` is deprecated — safe to skip)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Prepare test dataset
test_dataset = Dataset.from_pandas(test_df[['text']])
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])

# Predict
preds = trainer.predict(test_dataset)
labels = preds.predictions.argmax(-1)

# Map predictions back to sentiment
test_df['sentiment'] = ['Positive' if label == 1 else 'Negative' for label in labels]

# Output to CSV
output_df = test_df[['ID', 'sentiment']]
output_df.to_csv("output.csv", index=False)
print("✅ Predictions saved to output.csv.")
