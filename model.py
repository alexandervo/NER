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
import matplotlib.pyplot as plt

label_map = {
    'EVENT': 0, 'IP': 1, 'O': 2, 'PERSON': 3, 'LOCATION': 4,
    'URL': 5, 'ORGANIZATION': 6, 'SKILL': 7, 'MISCELLANEOUS': 8,
    'EMAIL': 9, 'DATETIME': 10, 'PRODUCT': 11, 'PHONENUMBER': 12,
    'QUANTITY': 13, 'ADDRESS': 14, 'PERSONTYPE': 15
}

# Initialize the model with higher dropout
model = BertForTokenClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=len(label_map),
    hidden_dropout_prob=0.1,  # Increase dropout rate
    attention_probs_dropout_prob=0.1,  # Increase attention dropout rate
)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
config = {
    "raw_path": "dataset/Cleaned_dataset/",
    "save_path": "dataset/Processed_dataset",
    "batch_size": 32,  # Increase batch size
}



train_data, dev_data, test_data = create_data(config)

# Define optimizer and scheduler with lower learning rate
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-12)
total_steps = len(train_data) * 5 # Number of epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
epochs = 5  # Increase number of epochs
patience = 1
best_valid_loss = float('inf')
early_stop_counter = 0

train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

def calculate_accuracy(preds, labels, mask):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    mask_flat = mask.flatten()
    return np.sum((pred_flat == labels_flat) * mask_flat) / np.sum(mask_flat)

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')

    model.train()
    total_train_loss = 0
    total_train_accuracy = 0

    for step, batch in enumerate(train_data):
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].to(device)

        model.zero_grad()
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        logits = outputs.logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()
        mask_ids = batch_attention_mask.to('cpu').numpy()
        total_train_accuracy += calculate_accuracy(logits, label_ids, mask_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_data)
    avg_train_accuracy = total_train_accuracy / len(train_data)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    print(f'Average training loss: {avg_train_loss:.2f}')
    print(f'Average training accuracy: {avg_train_accuracy:.2f}')

    model.eval()
    total_valid_loss = 0
    total_valid_accuracy = 0

    with torch.no_grad():
        for batch in dev_data:
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
            loss = outputs.loss
            total_valid_loss += loss.item()

            logits = outputs.logits.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()
            mask_ids = batch_attention_mask.to('cpu').numpy()
            total_valid_accuracy += calculate_accuracy(logits, label_ids, mask_ids)

    avg_valid_loss = total_valid_loss / len(dev_data)
    avg_valid_accuracy = total_valid_accuracy / len(dev_data)
    valid_losses.append(avg_valid_loss)
    valid_accuracies.append(avg_valid_accuracy)

    print(f'Average validation loss: {avg_valid_loss:.2f}')
    print(f'Average validation accuracy: {avg_valid_accuracy:.2f}')

    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print("Early stopping triggered")
        break

model.load_state_dict(torch.load('best_model.pt'))

model.save_pretrained('ner_model')
tokenizer.save_pretrained('ner_model')

# Plotting the results
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.show()
