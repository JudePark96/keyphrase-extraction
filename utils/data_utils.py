__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import codecs
import json
import logging
import pickle
import torch


from typing import Any
from tqdm import tqdm
from transformers import BertTokenizer
from config import bert_model_config
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger()


class KP20KDataset(Dataset):
    def __init__(self, features: list) -> None:
        super(KP20KDataset, self).__init__()
        self.features = features

    def __getitem__(self, index: int) -> None:
        return self.features[index]

    def __len__(self) -> int:
        return len(self.features)


def save_pickle_data(filename: str, data: Any) -> None:
    with codecs.open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def get_and_save_dataset(
        json_path: str,
        output_path: str,
        tokenizer: BertTokenizer,
        max_doc_seq_len: int,
        max_title_seq_len: int):

    features = []

    with codecs.open(json_path, 'r', 'utf-8') as f:
        for idx, line in tqdm(enumerate(f)):
            # {url, doc_word, title, keyphrase}
            json_object = json.loads(line)

            if idx % 500 == 0:
                print(f'Dataset {idx} -> {str(json_object)}')

            """
            Dictionary consists of 
            
            input_ids
            token_input_ids
            attention_mask
            """
            encoded_title = tokenizer(' '.join(json_object['title'][0]),
                                      padding='max_length',
                                      truncation=True,
                                      max_length=max_title_seq_len,
                                      return_tensors='pt')

            encoded_doc_words = tokenizer(' '.join(json_object['doc_words']),
                                          padding='max_length',
                                          truncation=True,
                                          max_length=max_doc_seq_len,
                                          return_tensors='pt')


            # if use n-gram feature, should be changed the way to manipulate.
            start_end_pos = json_object['start_end_pos']
            start_pos = []
            end_pos = []

            for position in start_end_pos:
                # if the keyphrase used several times in the document.
                if len(position) > 1:
                    for nested_position in position:
                        start_pos.append(nested_position[0])
                        end_pos.append(nested_position[1])

                # this keyphrase used only one time in the document.
                start_pos.append(position[0][0])
                end_pos.append(position[0][1])

            # length should be equal.
            assert len(start_pos) == len(end_pos)

            start_pos = sorted(start_pos)
            end_pos = sorted(end_pos)

            start_pos_tensor = torch.zeros(1, max_doc_seq_len)
            end_pos_tensor = torch.zeros(1, max_doc_seq_len)

            for s, e in zip(start_pos, end_pos):
                # [CLS] ... [SEP]
                if ((s + 1) >= max_doc_seq_len) or ((e + 1) >= max_doc_seq_len):
                    continue
                else:
                    start_pos_tensor[0][s + 1] = 1
                    end_pos_tensor[0][e + 1] = 1

            features.append({
                'url': json_object['url'],
                'title': encoded_title,
                'doc_words': encoded_doc_words,
                'start_position': start_pos_tensor,
                'end_position': end_pos_tensor
            })

        f.close()

    save_pickle_data(output_path, features)

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(bert_model_config)
    get_and_save_dataset('../rsc/prepro_dataset/kp20k.train.json', '../rsc/kp20k.train.feature.pkl', tokenizer, 128, 30)

