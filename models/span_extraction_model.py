__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import torch.nn.functional as F
import torch.nn as nn
import torch


from transformers import BertModel, BertTokenizer


class SpanClassifier(nn.Module):
    def __init__(self, bert_model: BertModel, model_type: str) -> None:
        super(SpanClassifier, self).__init__()
        self.bert = bert_model
        self.model_type = model_type
        self.init_config(model_type)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        :param batch: dict_keys(['doc', 'title', 'start_pos', 'end_pos'])
        :return:
        """
        bs = batch['doc']['input_ids'].size(0)
        doc_last_hidden_states, doc_pooled_output = self.bert_model(**batch['doc'])
        title_last_hidden_states, title_pooled_output = self.bert_model(**batch['title'])

        if self.model_type == 'baseline':
            return self.baseline(doc_last_hidden_states=doc_last_hidden_states)
        elif self.model_type == 'span_rank':
            pass
        elif self.model_type == 'span_rank_title_orh':
            pass
        pass

    def init_config(self, model_type: str):
        if model_type == 'baseline':
            self.s_classifier = nn.Linear(self.bert.config.hidden_size, 1, bias=True)
            self.e_classifier = nn.Linear(self.bert.config.hidden_size, 1, bias=True)
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        elif model_type == 'span_rank':
            # loss = L_rank + L_cross{start_pos} + L_cross{end_pos
            self.s_classifier = nn.Linear(self.bert.config.hidden_size, 1, bias=True)
            self.e_classifier = nn.Linear(self.bert.config.hidden_size, 1, bias=True)
            self.rank_loss = nn.MarginRankingLoss(margin=1.0, reduction='mean')
        elif model_type == 'span_rank_title_orh':
            self.rank_loss = nn.MarginRankingLoss(margin=1.0, reduction='mean')

    def baseline(self, doc_last_hidden_states, start_pos, end_pos):
        """
        """
        s_logits = F.sigmoid(self.s_classifier(doc_last_hidden_states).squeeze(dim=-1))
        e_logits = F.sigmoid(self.e_classifier(doc_last_hidden_states).squeeze(dim=-1))

        s_loss = self.bce_loss(s_logits, start_pos)
        e_loss = self.bce_loss(e_logits, end_pos)

        total_loss = s_loss + e_loss

        return total_loss

    def span_rank(self, doc_last_hidden_states, start_pos, end_pos):
        pass



if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    batch_sentences = ["Hello, my dog is cute", "Hello, my cat is cute"]
    inputs = tokenizer(batch_sentences, return_tensors="pt")
    print(inputs)

