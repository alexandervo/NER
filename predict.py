'''import re
import torch

def get_key_by_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None

def predict_entities(model, tokenizer, sentence):
    model.eval()  # Set model to evaluation mode for prediction
    label_map = {
        'EVENT': 0, 'IP': 1, 'O': 2, 'PERSON': 3, 'LOCATION': 4,
        'URL': 5, 'ORGANIZATION': 6, 'SKILL': 7, 'MISCELLANEOUS': 8,
        'EMAIL': 9, 'DATETIME': 10, 'PRODUCT': 11, 'PHONENUMBER': 12,
        'QUANTITY': 13, 'ADDRESS': 14, 'PERSONTYPE': 15
    }

    with torch.no_grad():  # Disable gradient calculation for prediction
        # Preprocess the sentence
        encoded_sentence = tokenizer(sentence, return_tensors="pt")
        input_ids = encoded_sentence["input_ids"].to(model.device)
        attention_mask = encoded_sentence["attention_mask"].to(model.device)

        # Make prediction
        outputs = model(input_ids, attention_mask=attention_mask)
        predicted_labels = torch.argmax(outputs.logits, dim=-1)

        # Decode predicted labels and tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        predicted_entities = []

        # Iterate over each item in predicted_labels
        for token_index, label_tensor in enumerate(predicted_labels[0]):
            label = label_tensor.item()
            entity_type = get_key_by_value(label_map, label)
            token = tokens[token_index]

            if token not in ['[CLS]', '[SEP]']:
                predicted_entities.append({"token": token, "entity_type": entity_type})

        s = ""
        current_entity_type = None

        for entity in predicted_entities:
            token = entity.get("token")
            entity_type = entity.get("entity_type")

            # Check for entity start or continuation
            if current_entity_type is None or entity_type != current_entity_type:
                if current_entity_type and current_entity_type != 'O':
                    s += f" ({current_entity_type})"
                if token:
                    if token not in [',', '.', '!', '?', ':', ';']:
                        s += " " + token
                    else:
                        s += token
                current_entity_type = entity_type
            else:
                if token:
                    if token not in [',', '.', '!', '?', ':', ';']:
                        s += " " + token
                    else:
                        s += token

        # Append the last entity type if not 'O'
        if current_entity_type and current_entity_type != 'O':
            s += f" ({current_entity_type})"

        # Remove subword markers
        text = re.sub(r" ##", "", s)
        return text.strip()'''

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
