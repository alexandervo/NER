from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForTokenClassification
import re

app = Flask(__name__)

# Load the model and tokenizer
model = BertForTokenClassification.from_pretrained('ner_model')
tokenizer = BertTokenizer.from_pretrained('ner_model')
model.eval()

def get_key_by_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None

def predict_entities(sentence):
    label_map = {
        'EVENT': 0, 'IP': 1, 'O': 2, 'PERSON': 3, 'LOCATION': 4,
        'URL': 5, 'ORGANIZATION': 6, 'SKILL': 7, 'MISCELLANEOUS': 8,
        'EMAIL': 9, 'DATETIME': 10, 'PRODUCT': 11, 'PHONENUMBER': 12,
        'QUANTITY': 13, 'ADDRESS': 14, 'PERSONTYPE': 15
    }

    # label_map = {'O': 0, 'IP': 1, 'EVENT': 2, 'PERSON': 3, 'LOCATION': 4, 'URL': 5, 'ORGANIZATION': 6, 'SKILL': 7, 'MISCELLANEOUS': 8, 'EMAIL': 9, 'DATETIME': 10, 'PRODUCT': 11, 'PHONENUMBER': 12, 'QUANTITY': 13, 'ADDRESS': 14, 'PERSONTYPE': 15}
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

        if current_entity_type and current_entity_type != 'O':
            s += f" ({current_entity_type})"

        # Remove subword markers
        text = re.sub(r" ##", "", s)
        return text.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form['sentence']
        result = predict_entities(sentence)
        return render_template('index.html', result=result, sentence=sentence)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)