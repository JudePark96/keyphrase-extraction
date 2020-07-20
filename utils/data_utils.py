__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import argparse
import codecs
import json
import logging
import pickle


from typing import Any

import h5py as h5py
import torch
from tqdm import tqdm
from transformers import BertTokenizer


logger = logging.getLogger()


def save_pickle_data(filename: str, data: Any) -> None:
    with codecs.open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def load_pickle_data(filename: str) -> Any:
    with codecs.open(filename, 'rb') as f:
        data = pickle.load(f)
        f.close()
    print(data)


def filter_absent_keyphrase(t: torch.Tensor) -> bool:
    return False if str(t.size()) == 'torch.Size([0, 1])' else True


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
            keyphrases = json_object['keyphrases']

            start_pos = []
            end_pos = []

            squeezed_input_ids = encoded_doc_words['input_ids'].squeeze(dim=0)

            start_pos_tensor = torch.zeros((1, max_doc_seq_len)).long()
            end_pos_tensor = torch.zeros((1, max_doc_seq_len)).long()

            for keyphrase in keyphrases:
                # the case of consisting of several words.
                if len(keyphrase) > 1:
                    start_keyphrase = keyphrase[0]
                    end_keyphrase = keyphrase[-1]

                    # present keyphrase extraction
                    if start_keyphrase in json_object['doc_words'] and end_keyphrase in json_object['doc_words']:
                        decoded_start_keyphrase = tokenizer.encode(start_keyphrase)
                        decoded_end_keyphrase = tokenizer.encode(end_keyphrase)

                        start_token = decoded_start_keyphrase[1] # except [CLS] token.
                        end_token = decoded_end_keyphrase[-2] # except [SEP] token.

                        s_p, e_p = (squeezed_input_ids == start_token).nonzero(), \
                                   (squeezed_input_ids == end_token).nonzero()

                        if filter_absent_keyphrase(s_p) and filter_absent_keyphrase(e_p):
                            if max_doc_seq_len > s_p[0].item() and max_doc_seq_len > e_p[0].item():
                                start_pos_tensor[0][s_p[0].item()] = 1
                                end_pos_tensor[0][s_p[0].item()] = 1

                else:
                    # present keyphrase extraction
                    if keyphrase[0] in json_object['doc_words']:
                        # ex: telepresence -> '[CLS]', 'tel', '##ep', '##res', '##ence', '[SEP]'
                        decoded_keyphrase = tokenizer.encode(keyphrase[0])

                        # without special tokens such as CLS, SEP
                        if len(decoded_keyphrase) == 3:
                            keyphrase_wo_special_tokens = decoded_keyphrase[1]
                            s_p, e_p = (squeezed_input_ids == keyphrase_wo_special_tokens).nonzero(), \
                                       (squeezed_input_ids == keyphrase_wo_special_tokens).nonzero()

                            if filter_absent_keyphrase(s_p) and filter_absent_keyphrase(e_p):
                                if max_doc_seq_len > s_p[0].item() and max_doc_seq_len > e_p[0].item():
                                    start_pos_tensor[0][s_p[0].item()] = 1
                                    end_pos_tensor[0][s_p[0].item()] = 1
                        else:
                            keyphrase_wo_special_tokens = decoded_keyphrase[1:len(decoded_keyphrase)-1]
                            s_p = (squeezed_input_ids == keyphrase_wo_special_tokens[0]).nonzero()
                            e_p = (squeezed_input_ids == keyphrase_wo_special_tokens[-1]).nonzero()

                            if filter_absent_keyphrase(s_p) and filter_absent_keyphrase(e_p):
                                if max_doc_seq_len > s_p[0].item() and max_doc_seq_len > e_p[0].item():
                                    start_pos_tensor[0][s_p[0].item()] = 1
                                    end_pos_tensor[0][s_p[0].item()] = 1

            features.append({
                'url': json_object['url'],
                'title': encoded_title,
                'doc_words': encoded_doc_words,
                'start_pos_tensor': start_pos_tensor,
                'end_pos_tensor': end_pos_tensor
            })

            if idx == 100:
                break

        f.close()

    """
    url => [1, 2, 3, ..., n]
    title => [input_ids, token_input_ids, attention_mask]
    doc_words => [input_ids, token_input_ids, attention_mask]
    start_pos_tensor
    end_pos_tensor
    """

    # label
    start_pos_tensors = torch.cat([feature['start_pos_tensor'] for feature in features], dim=0)
    end_pos_tensors = torch.cat([feature['end_pos_tensor'] for feature in features], dim=0) # [n x max_len], n is size of dataset.

    # encoded_title
    title_input_ids = torch.cat([feature['title']['input_ids'] for feature in features], dim=0)
    title_token_type_ids = torch.cat([feature['title']['token_type_ids'] for feature in features], dim=0)
    title_attention_mask = torch.cat([feature['title']['attention_mask'] for feature in features], dim=0)

    # encoded_doc_words
    doc_input_ids = torch.cat([feature['doc_words']['input_ids'] for feature in features], dim=0)
    doc_token_type_ids = torch.cat([feature['doc_words']['token_type_ids'] for feature in features], dim=0)
    doc_attention_mask = torch.cat([feature['doc_words']['attention_mask'] for feature in features], dim=0)

    feature_output = h5py.File(output_path, 'w')

    title_group = feature_output.create_group('title')
    title_group.create_dataset('input_ids', data=title_input_ids)
    title_group.create_dataset('token_type_ids', data=title_token_type_ids)
    title_group.create_dataset('attention_mask', data=title_attention_mask)

    doc_group = feature_output.create_group('doc')
    doc_group.create_dataset('input_ids', data=doc_input_ids)
    doc_group.create_dataset('token_type_ids', data=doc_token_type_ids)
    doc_group.create_dataset('attention_mask', data=doc_attention_mask)

    label_group = feature_output.create_group('label')
    label_group.create_dataset('start_pos', data=start_pos_tensors)
    label_group.create_dataset('end_pos', data=end_pos_tensors)

    feature_output.close()

    print('feature preprocessing is done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_type', type=str, required=True,
                        help="watch out transformers library. ex => bert-base-uncased")
    parser.add_argument('--source_file', type=str, required=True)
    parser.add_argument('--dest_file', type=str, required=True)
    parser.add_argument('--max_doc_seq_len', type=int, required=True)
    parser.add_argument('--max_title_seq_len', type=int, required=True)

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_type)
    # get_and_save_dataset('../rsc/prepro_dataset/kp20k.train.json', '../rsc/kp20k.train.feature.pkl', tokenizer, 128, 30)
    get_and_save_dataset(args.source_file,
                         args.dest_file,
                         tokenizer,
                         args.max_doc_seq_len,
                         args.max_title_seq_len)

