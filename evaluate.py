from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from processing import create_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

config = {
    "raw_path": "dataset/Cleaned_dataset",
    "save_path": "dataset/Processed_dataset",
    "batch_size": 16,
}
_, _, test_data = create_data(config)
label_map = {
        'EVENT': 0, 'IP': 1, 'O': 2, 'PERSON': 3, 'LOCATION': 4,
        'URL': 5, 'ORGANIZATION': 6, 'SKILL': 7, 'MISCELLANEOUS': 8,
        'EMAIL': 9, 'DATETIME': 10, 'PRODUCT': 11, 'PHONENUMBER': 12,
        'QUANTITY': 13, 'ADDRESS': 14, 'PERSONTYPE': 15
    }
#tokenizer = BertTokenizerFast.from_pretrained('ner_model')
model = BertForTokenClassification.from_pretrained('ner_model', num_labels=len(label_map))
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

# Hàm đánh giá mô hình
def evaluate_model(model, data_loader, label_map):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=2)

            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            for i in range(len(labels)):
                all_preds.append(preds[i])
                all_labels.append(labels[i])

    return compute_metrics(all_preds, all_labels, label_map)

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def get_flat_labels_and_predictions(preds, labels):
    y_true = [l for label in labels for l in label if l != -100]
    y_pred = [p for pred, label in zip(preds, labels) for p, l in zip(pred, label) if l != -100]
    return y_true, y_pred

def evaluate_model_with_cm(model, data_loader, label_map):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=2)

            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            for i in range(len(labels)):
                all_preds.append(preds[i])
                all_labels.append(labels[i])

    y_true, y_pred = get_flat_labels_and_predictions(all_preds, all_labels)
    metrics = compute_metrics(all_preds, all_labels, label_map)

    # Lấy nhãn từ label_map
    label_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    return metrics, y_true, y_pred, label_names


# Đặt mô hình và dữ liệu trên thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Đánh giá mô hình và hiển thị ma trận nhầm lẫn
results, y_true, y_pred, label_names = evaluate_model_with_cm(model, test_data, label_map)

# Hiển thị các chỉ số đánh giá
for metric, value in results.items():
    print(f'{metric}: {value:.4f}')

# Hiển thị ma trận nhầm lẫn
plot_confusion_matrix(y_true, y_pred, list(range(len(label_names))))