__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import codecs
import json
import logging
import h5py

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

            encoded_title = tokenizer(' '.join(json_object['title'][0]),
                                      padding=True,
                                      truncation=True,
                                      max_length=max_title_seq_len)

            encoded_doc_words = tokenizer(json_object['doc_words'],
                                          padding=True,
                                          truncation=True,
                                          max_length=max_doc_seq_len)

            start_end_pos = json_object['start_end_pos']

            features.append({
                'title': encoded_title,
                'doc_words': encoded_doc_words,
                'start_end_pos': start_end_pos,
            })

        f.close()

    output_file = h5py.File(output_path, 'w')
    output_file['features'] = features
    output_file.close()

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(bert_model_config)
    get_and_save_dataset('../rsc/prepro_dataset/kp20k.dev.json', '../rsc/', tokenizer, 128, 30)

