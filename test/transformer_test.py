__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import math

import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn


model_type = 'bert-base-uncased'


tokenizer = BertTokenizer.from_pretrained(model_type)
bert_model = BertModel.from_pretrained(model_type)

sentence = "Hello, my dog is cute"

print(tokenizer.encode(sentence, max_length=10, padding=True, truncation=True))

inputs = tokenizer("Hello, my dog is cute",
                   padding='max_length',
                   truncation=True,
                   max_length=7,
                   return_tensors="pt")

print(inputs['input_ids'].shape)

print(inputs)

outputs = bert_model(**inputs)


"""
last hidden states.
shape of last hidden state is [bs x seq_len x hidden_dim]
"""
print(outputs[0].shape)

# last_hidden_state = torch.einsum('ijk->ikj', outputs[0])
last_hidden_state = outputs[0]

start_logits, end_logits = logits.split(1, dim=-1)

# bi_gram = nn.Conv1d(768, 768, 2)(last_hidden_state)
# tri_gram = nn.Conv1d(768, 768, 3)(last_hidden_state)
#
# bi_gram = bi_gram.transpose(1, 2)
# tri_gram = tri_gram.transpose(1, 2)
#
# print(bi_gram.shape, tri_gram.shape)
# print(torch.cat([bi_gram, tri_gram], dim=1).shape)


