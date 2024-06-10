import os
import pandas as pd
import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.nn.utils.rnn import pad_sequence

# Hàm đọc dữ liệu từ file và chuyển đổi thành dataframe
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = eval(file.read())
    sentences = [[word for word, label in sentence] for sentence in data]
    labels = [[label for word, label in sentence] for sentence in data]
    df = pd.DataFrame({'sentence': sentences, 'label': labels})
    return df

# Khởi tạo tokenizer
tokenizer = BertTokenizerFast.from_pretrained('ner_model')

# Tạo label map
label_map = {
        'EVENT': 0, 'IP': 1, 'O': 2, 'PERSON': 3, 'LOCATION': 4,
        'URL': 5, 'ORGANIZATION': 6, 'SKILL': 7, 'MISCELLANEOUS': 8,
        'EMAIL': 9, 'DATETIME': 10, 'PRODUCT': 11, 'PHONENUMBER': 12,
        'QUANTITY': 13, 'ADDRESS': 14, 'PERSONTYPE': 15
    }

# Hàm tokenize và ánh xạ nhãn
def tokenize_and_align_labels(sentence, label):
    tokenized_inputs = tokenizer(sentence, truncation=True, is_split_into_words=True)
    labels = []

    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(label_map[label[word_idx]])
        else:
            labels.append(label_map[label[word_idx]] if label[word_idx] == 'O' else label_map[label[word_idx]])
        previous_word_idx = word_idx

    return tokenized_inputs, labels

# Chuyển đổi dữ liệu
def data_for_bert(df):
    tokenized_texts_and_labels = [tokenize_and_align_labels(eval(s), eval(l)) for s, l in zip(df['sentence'], df['label'])]

    input_ids = [x[0]['input_ids'] for x in tokenized_texts_and_labels]
    attention_masks = [x[0]['attention_mask'] for x in tokenized_texts_and_labels]
    labels = [x[1] for x in tokenized_texts_and_labels]
    return input_ids, attention_masks, labels

class NERDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad sequences to the same length
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is typically used for ignoring labels during loss calculation

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def create_data(config):
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])

    if not os.path.exists(os.path.join(config["save_path"], "train_data.csv")):
        print("Đang khởi tạo dữ liệu...")
        train_data = load_data(os.path.join(config["raw_path"], "train.txt"))
        dev_data = load_data(os.path.join(config["raw_path"], "dev.txt"))
        test_data = load_data(os.path.join(config["raw_path"], "test.txt"))
        train_data.to_csv(os.path.join(config["save_path"], "train_data.csv"), index=False)
        dev_data.to_csv(os.path.join(config["save_path"], "dev_data.csv"), index=False)
        test_data.to_csv(os.path.join(config["save_path"], "test_data.csv"), index=False)
        print("Khởi tạo dữ liệu hoàn tất!")

    print("Đang tải dữ liệu...")
    train_data = pd.read_csv(os.path.join(config["save_path"], "train_data.csv"))
    dev_data = pd.read_csv(os.path.join(config["save_path"], "dev_data.csv"))
    test_data = pd.read_csv(os.path.join(config["save_path"], "test_data.csv"))
    print("Tải dữ liệu hoàn tất!")

    train = read_train_data(train_data, config)
    dev = read_train_data(dev_data, config)
    test = read_train_data(test_data, config)
    return train, dev, test

def read_train_data(df, config):
    input_ids, attention_masks, labels = data_for_bert(df)
    dataset = NERDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        batch_size=config["batch_size"],
        collate_fn=collate_fn  # Use the custom collate function
    )
    return dataloader
