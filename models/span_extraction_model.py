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
        doc_last_hidden_states, doc_pooled_output = self.bert(**batch['doc'])
        title_last_hidden_states, title_pooled_output = self.bert(**batch['title'])

        if self.model_type == 'baseline':
            return self.baseline(doc_last_hidden_states=doc_last_hidden_states,
                                 attention_mask=batch['doc']['attention_mask'],
                                 start_pos=batch['start_pos'],
                                 end_pos=batch['end_pos'])
        elif self.model_type == 'span_rank':
            pass
        elif self.model_type == 'span_rank_title_orh':
            pass
        pass

    def init_config(self, model_type: str) -> None:
        if model_type == 'baseline':
            self.s_classifier = nn.Linear(self.bert.config.hidden_size, 2, bias=True)
            self.e_classifier = nn.Linear(self.bert.config.hidden_size, 2, bias=True)
            self.nll_loss = nn.NLLLoss(reduction='mean')
        elif model_type == 'span_rank':
            # loss = L_rank + L_cross{start_pos} + L_cross{end_pos
            self.s_classifier = nn.Linear(self.bert.config.hidden_size, 2, bias=True)
            self.e_classifier = nn.Linear(self.bert.config.hidden_size, 2, bias=True)
            self.rank_loss = nn.MarginRankingLoss(margin=1.0, reduction='mean')
        elif model_type == 'span_rank_title_orh':
            self.rank_loss = nn.MarginRankingLoss(margin=1.0, reduction='mean')

    def baseline(self,
                 doc_last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor,
                 start_pos: torch.Tensor,
                 end_pos: torch.Tensor) -> torch.Tensor:

        # [bs x seq_len] => [(bs * seq_len)]
        attention_mask = attention_mask.view(-1) == 1

        # [bs x seq_len x num_label]
        s_logits = F.log_softmax(self.s_classifier(doc_last_hidden_states).squeeze(dim=-1), dim=-1)
        e_logits = F.log_softmax(self.e_classifier(doc_last_hidden_states).squeeze(dim=-1), dim=-1)

        # [(bs * seq_len) x num_label]
        s_logits = s_logits.view(-1, 2)[attention_mask]
        e_logits = e_logits.view(-1, 2)[attention_mask]

        start_pos = start_pos.view(-1)[attention_mask]
        end_pos = end_pos.view(-1)[attention_mask]

        # for calculating loss function, the shape has to be equal.
        assert s_logits.shape[0] == start_pos.shape[0]
        assert e_logits.shape[0] == end_pos.shape[0]

        s_loss = self.nll_loss(s_logits, start_pos)
        e_loss = self.nll_loss(e_logits, end_pos)

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

