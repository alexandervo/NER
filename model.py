import torch
from transformers import BertForTokenClassification, BertTokenizer

# Load the model
model = BertForTokenClassification.from_pretrained('ner_model')

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('ner_model')

# Set the model to evaluation mode
model.eval()