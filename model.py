import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from processing import create_data
import os

config = {
    "raw_path": "dataset/Cleaned_dataset",
    "save_path": "dataset/Processed_dataset",
    "batch_size": 16,
}
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
label_map = {
        'EVENT': 0, 'IP': 1, 'O': 2, 'PERSON': 3, 'LOCATION': 4,
        'URL': 5, 'ORGANIZATION': 6, 'SKILL': 7, 'MISCELLANEOUS': 8,
        'EMAIL': 9, 'DATETIME': 10, 'PRODUCT': 11, 'PHONENUMBER': 12,
        'QUANTITY': 13, 'ADDRESS': 14, 'PERSONTYPE': 15
    }
# Khởi tạo mô hình
model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(label_map))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_data, dev_data, test_data = create_data(config)
# Định nghĩa optimizer và scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
total_steps = len(train_data) * 3  # 3 là số epoch, bạn có thể thay đổi

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def flat_accuracy(preds, labels):
    pred_flat = np.concatenate([np.argmax(p, axis=1) for p in preds])
    labels_flat = np.concatenate(labels)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def compute_metrics(preds, labels, label_map):
    preds_flat = [p for pred in preds for p in pred]
    labels_flat = [l for label in labels for l in label]

    preds_flat = [p for p, l in zip(preds_flat, labels_flat) if l != -100]
    labels_flat = [l for l in labels_flat if l != -100]

    accuracy = accuracy_score(labels_flat, preds_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='macro', labels=list(label_map.values()))

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)

epochs = 20
patience = 2
early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.01)
model_path = 'best_model.pt'

for epoch in trange(epochs, desc="Epoch"):
    print(f'\nEpoch {epoch + 1}/{epochs}')

    # Training
    model.train()
    total_train_loss = 0

    for step, batch in enumerate(tqdm(train_data, desc="Training", leave=False)):
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].to(device)

        model.zero_grad()
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            print(f"  Batch {step}/{len(train_data)} - Train Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_data)
    print(f'Average training loss: {avg_train_loss:.2f}')

    # Validation
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dev_data, desc="Validation", leave=False):
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
            loss = outputs.loss
            logits = outputs.logits

        total_val_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()

        all_preds.extend([list(p) for p in logits])
        all_labels.extend(label_ids)

    avg_val_loss = total_val_loss / len(dev_data)
    val_accuracy = flat_accuracy(all_preds, all_labels)
    precision, recall, f1, _ = compute_metrics(all_preds, all_labels)

    print(f'Validation loss: {avg_val_loss:.2f}')
    print(f'Validation Accuracy: {val_accuracy:.2f}')
    print(f'Validation Precision: {precision:.2f}')
    print(f'Validation Recall: {recall:.2f}')
    print(f'Validation F1-score: {f1:.2f}')

    early_stopping(avg_val_loss, model, model_path)

    if early_stopping.early_stop:
        print("Early stopping")
        break


# Load the best model
model.load_state_dict(torch.load(model_path))

# Testing
model.eval()
all_preds = []
all_labels = []

for batch in tqdm(test_data, desc="Testing"):
    batch_input_ids = batch['input_ids'].to(device)
    batch_attention_mask = batch['attention_mask'].to(device)
    batch_labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        logits = outputs.logits

    logits = logits.detach().cpu().numpy()
    label_ids = batch_labels.to('cpu').numpy()

    all_preds.extend([list(p) for p in logits])
    all_labels.extend(label_ids)

test_accuracy = flat_accuracy(all_preds, all_labels)
precision, recall, f1, _ = compute_metrics(all_preds, all_labels)

print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test Precision: {precision:.2f}')
print(f'Test Recall: {recall:.2f}')
print(f'Test F1-score: {f1:.2f}')

model.save_pretrained('ner_model')
tokenizer.save_pretrained('ner_model')
