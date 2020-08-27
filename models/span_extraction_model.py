__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from transformers import BertModel, BertTokenizer
from utils.metric import f1_score
from typing import Tuple, Any


import torch.nn.functional as F
import torch.nn as nn
import torch


class SpanClassifier(nn.Module):
    def __init__(self, bert_model: BertModel, model_type: str) -> None:
        super(SpanClassifier, self).__init__()
        self.bert = bert_model
        self.model_type = model_type
        self.init_config(model_type)

    def forward(self, batch: dict, is_eval: bool=False) -> Any:
        """
        :param batch: dict_keys(['doc', 'title', 'start_pos', 'end_pos'])
        :return:
        """
        doc_last_hidden_states, doc_pooled_output = self.bert(**batch['doc'])
        title_last_hidden_states, title_pooled_output = self.bert(**batch['title'])

        if is_eval:
            if self.model_type == 'baseline':
                return self.baseline_evaluate(doc_last_hidden_states=doc_last_hidden_states,
                                              attention_mask=batch['doc']['attention_mask'],
                                              start_pos=batch['start_pos'],
                                              end_pos=batch['end_pos'])
        else:
            if self.model_type == 'baseline':
                return self.baseline(doc_last_hidden_states=doc_last_hidden_states,
                                     attention_mask=batch['doc']['attention_mask'],
                                     start_pos=batch['start_pos'],
                                     end_pos=batch['end_pos'])
            elif self.model_type == 'span_rank':
                return self.span_rank(doc_last_hidden_states=doc_last_hidden_states,
                                      attention_mask=batch['doc']['attention_mask'],
                                      start_pos=batch['start_pos'],
                                      end_pos=batch['end_pos'])
                pass
            elif self.model_type == 'span_rank_title_orh':
                pass

    def init_config(self, model_type: str) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_type == 'baseline':
            self.s_classifier = nn.Linear(self.bert.config.hidden_size, 2, bias=True)
            self.e_classifier = nn.Linear(self.bert.config.hidden_size, 2, bias=True)
            self.nll_loss = nn.NLLLoss(reduction='mean').to(self.device)
        elif model_type == 'span_rank':
            # loss = L_rank + L_cross{start_pos} + L_cross{end_pos}
            self.s_classifier = nn.Linear(self.bert.config.hidden_size, 2, bias=True)
            self.e_classifier = nn.Linear(self.bert.config.hidden_size, 2, bias=True)
            self.rank_loss = nn.MarginRankingLoss(margin=1.0, reduction='mean').to(self.device)
            self.nll_loss = nn.NLLLoss(reduction='mean').to(self.device)
        elif model_type == 'span_rank_title_orh':
            self.rank_loss = nn.MarginRankingLoss(margin=1.0, reduction='mean')

    def baseline(self,
                 doc_last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor,
                 start_pos: torch.Tensor,
                 end_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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

        return total_loss, s_loss, e_loss

    def span_rank(self,
                  doc_last_hidden_states: torch.Tensor,
                  attention_mask: torch.Tensor,
                  start_pos: torch.Tensor,
                  end_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

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

        s_score = s_logits.max(dim=-1)[0]
        e_score = e_logits.max(dim=-1)[0]

        s_pos_score = s_score[start_pos == 1]
        s_neg_score = s_score[start_pos == 0]

        e_pos_score = e_score[end_pos == 1]
        e_neg_score = e_score[end_pos == 0]

        assert s_pos_score.shape[0] == e_pos_score.shape[0]
        assert s_neg_score.shape[0] == e_neg_score.shape[0]

        flag = torch.FloatTensor([1]).to(self.device)
        s_rank_loss = self.rank_loss(s_pos_score.unsqueeze(-1), s_neg_score.unsqueeze(0), flag)
        e_rank_loss = self.rank_loss(e_pos_score.unsqueeze(-1), e_neg_score.unsqueeze(0), flag)

        total_loss = s_loss + e_loss + s_rank_loss + e_rank_loss

        return total_loss, s_loss, e_loss, s_rank_loss, e_rank_loss

    def baseline_evaluate(self,
                          doc_last_hidden_states: torch.Tensor,
                          attention_mask: torch.Tensor,
                          start_pos: torch.Tensor,
                          end_pos: torch.Tensor):
        # [bs x seq_len] => [(bs * seq_len)]
        attention_mask = attention_mask.view(-1) == 1

        # [bs x seq_len x num_label]
        s_logits = F.log_softmax(self.s_classifier(doc_last_hidden_states).squeeze(dim=-1), dim=-1)
        e_logits = F.log_softmax(self.e_classifier(doc_last_hidden_states).squeeze(dim=-1), dim=-1)

        start_pos = start_pos.view(-1)[attention_mask]
        end_pos = end_pos.view(-1)[attention_mask]

        # [bs x seq_len]
        s_score = s_logits.max(dim=-1)[0]
        e_score = e_logits.max(dim=-1)[0]

        # s_f1 = f1_score(y_true=start_pos, y_pred=s_score)
        # e_f1 = f1_score(y_true=end_pos, y_pred=e_score)

        return {
            's_score': s_score,
            'e_score': e_score,
            's_gt': start_pos,
            'e_gt': end_pos
        }


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    batch_sentences = ["Hello, my dog is cute", "Hello, my cat is cute"]
    inputs = tokenizer(batch_sentences, return_tensors="pt")
    print(inputs)
