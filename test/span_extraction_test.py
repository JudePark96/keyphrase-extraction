__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import math

from transformers import BertTokenizer, BertForQuestionAnswering, BertModel
import torch
import torch.nn as nn


class SpanAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SpanAttention, self).__init__()

        self.hidden_size = hidden_size
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states,):
        '''hidden_states and active_mask for word_level'''

        query = self.query_layer(hidden_states)
        key = self.key_layer(hidden_states)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        return attention_probs

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

outputs = model(**inputs)
last_hidden_states = outputs[0]

print(SpanAttention(768)(last_hidden_states).shape)
