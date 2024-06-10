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
        0: 'EVENT', 1: 'IP', 2: 'O', 3: 'PERSON', 4: 'LOCATION',
        5: 'URL', 6: 'ORGANIZATION', 7: 'SKILL', 8: 'MISCELLANEOUS',
        9: 'EMAIL', 10: 'DATETIME', 11: 'PRODUCT', 12: 'PHONENUMBER',
        13: 'QUANTITY', 14: 'ADDRESS', 15: 'PERSONTYPE'
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

        current_entity = None
        current_tokens = []

        for token_index, label_tensor in enumerate(predicted_labels[0]):
            label = label_tensor.item()
            entity_type = label_map[label]
            token = tokens[token_index]

            if token not in ['[CLS]', '[SEP]']:
                if current_entity == entity_type:
                    current_tokens.append(token.replace('##', ''))
                else:
                    if current_entity:
                        predicted_entities.append({
                            "entity": current_entity,
                            "tokens": " ".join(current_tokens)
                        })
                    current_entity = entity_type
                    current_tokens = [token.replace('##', '')]

        if current_entity:
            predicted_entities.append({
                "entity": current_entity,
                "tokens": " ".join(current_tokens)
            })

        return predicted_entities
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form['sentence']
        entities = predict_entities(sentence)
        return render_template('index.html', entities=entities, sentence=sentence)
    return render_template('index.html', entities=[])

if __name__ == '__main__':
    app.run(debug=True)