__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import codecs
import json
import logging
import re
import argparse
import unicodedata

from typing import Tuple, List, Dict, Union
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()
logger = logging.getLogger()


def load_data(output_path: str,
              src_fields: List[str]=['title', 'abstract'],
              trg_fields: List[str]=['keyword']) -> List[Tuple[str, str, List[str]]]:

    features = []

    with codecs.open(output_path, 'r', 'utf-8') as f:
        for idx, line in enumerate(tqdm(f)):
            json_object = json.loads(line)

            title = ' '.join([json_object[src_fields[0]]])
            abstract = ' '.join([json_object[src_fields[1]]])
            keyword = [(re.split(';', json_object[f])) for f in trg_fields][0]

            features.append((title, abstract, keyword))

    return features


def tokenize(features: List[Tuple[str, str, List[str]]],
             max_src_seq_len: int,
             min_src_seq_len: int,
             max_trg_seq_len: int,
             min_trg_seq_len: int,
             valid_check: bool=True) -> List[Dict[str, Union[List[str], list]]]:

    def tokenize_fn(text: str) -> List[str]:
        digit = '<digit>'

        # remove line breakers
        text = re.sub(r'[\r\n\t]', ' ', text)

        # pad spaces to the left and right of special punctuations
        text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)

        # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\'%]', text))

        # replace digit to special token.
        tokens = [w if not re.match('^\d+$', w) else digit for w in tokens]

        return tokens

    preprocessed_features = []

    for idx, (title, src, trgs) in tqdm(enumerate(features)):
        src_filter_flag = False
        src_tokens = tokenize_fn(src)

        # min_seq_len 과 max_seq_len 을 충족하지 못하면 넘긴다.
        if len(src_tokens) > max_src_seq_len:
            src_filter_flag = True
        if len(src_tokens) < min_src_seq_len:
            src_filter_flag = True

        if valid_check and src_filter_flag:
            continue

        # 글 제목도 토큰화함.
        title_tokens = tokenize_fn(title)

        trgs_tokens = []

        for trg in trgs:
            trg_filter_flag = False
            trg = trg.lower()

            # FILTER 1: remove all the abbreviations/acronyms in parentheses in keyphrases
            trg = re.sub(r'\(.*?\)', '', trg)
            trg = re.sub(r'\[.*?\]', '', trg)
            trg = re.sub(r'\{.*?\}', '', trg)

            # FILTER 2: ingore all the phrases that contains strange punctuations, very DIRTY data!
            puncts = re.findall(r'[,_\"<>\(\){}\[\]\?~`!@$%\^=]', trg)

            trg_tokens = tokenize_fn(trg)

            if len(puncts) > 0:
                continue

            if len(trg_tokens) > max_trg_seq_len:
                trg_filter_flag = True
            if len(trg_tokens) < min_trg_seq_len:
                trg_filter_flag = True

            if valid_check and trg_filter_flag:
                continue

            if valid_check and (len(trg_tokens) > 0 and re.match(r'\d\d[a-zA-Z\-]\d\d',
                                                                 trg_tokens[0].strip())) or (
                    len(trg_tokens) > 1 and re.match(r'\d\d\w\d\d', trg_tokens[1].strip())):
                continue

            trgs_tokens.append(trg_tokens)

        if valid_check and len(trgs_tokens) == 0:
            continue

        preprocessed_features.append({'title': title_tokens,
                                      'doc_words': src_tokens,
                                      'keyphrases': trgs_tokens})
    return preprocessed_features


def filter_absent_keyphrases(examples: List[Dict[str, Union[List[str], list]]]):
    logger.info('strat filter absent keyphrases for KP20k...')

    def find_stem_answer(word_list, ans_list):
        norm_doc_char, stem_doc_char = norm_doc_to_char(word_list)
        norm_stem_phrase_list = norm_phrase_to_char(ans_list)

        tot_ans_str = []
        tot_start_end_pos = []

        for norm_ans_char, stem_ans_char in norm_stem_phrase_list:

            norm_stem_doc_char = " ".join([norm_doc_char, stem_doc_char])

            if norm_ans_char not in norm_stem_doc_char and stem_ans_char not in norm_stem_doc_char:
                continue
            else:
                norm_doc_words = norm_doc_char.split(" ")
                stem_doc_words = stem_doc_char.split(" ")

                norm_ans_words = norm_ans_char.split(" ")
                stem_ans_words = stem_ans_char.split(" ")

                assert len(norm_doc_words) == len(stem_doc_words)
                assert len(norm_ans_words) == len(stem_ans_words)

                # find postions
                tot_pos = []

                for i in range(0, len(stem_doc_words) - len(stem_ans_words) + 1):

                    Flag = False

                    if norm_ans_words == norm_doc_words[i:i + len(norm_ans_words)]:
                        Flag = True

                    elif stem_ans_words == norm_doc_words[i:i + len(stem_ans_words)]:
                        Flag = True

                    elif norm_ans_words == stem_doc_words[i:i + len(norm_ans_words)]:
                        Flag = True

                    elif stem_ans_words == stem_doc_words[i:i + len(stem_ans_words)]:
                        Flag = True

                    if Flag:
                        tot_pos.append([i, i + len(norm_ans_words) - 1])
                        assert (i + len(stem_ans_words) - 1) >= i

                if len(tot_pos) > 0:
                    tot_start_end_pos.append(tot_pos)
                    tot_ans_str.append(norm_ans_char.split())

        assert len(tot_ans_str) == len(tot_start_end_pos)
        assert len(word_list) == len(norm_doc_char.split(" "))

        if len(tot_ans_str) == 0:
            return None
        return {'keyphrases': tot_ans_str, 'start_end_pos': tot_start_end_pos}

    def norm_phrase_to_char(phrase_list):
        norm_phrases = set()
        for phrase in phrase_list:
            p = " ".join([w.strip() for w in phrase if len(w.strip()) > 0])
            if len(p) < 1: continue
            norm_phrases.add(unicodedata.normalize('NFD', p))

        norm_stem_phrases = []
        for norm_chars in norm_phrases:
            stem_chars = " ".join([stemmer.stem(w) for w in norm_chars.split(" ")])
            norm_stem_phrases.append((norm_chars, stem_chars))

        return norm_stem_phrases

    def norm_doc_to_char(word_list):
        norm_char = unicodedata.normalize('NFD', " ".join(word_list))
        stem_char = " ".join([stemmer.stem(w.strip()) for w in norm_char.split(" ")])

        return norm_char, stem_char
    data_list = []

    null_ids, absent_ids = 0, 0

    url = 0

    for idx, ex in enumerate(tqdm(examples)):
        lower_words = [t.lower() for t in ex['doc_words']]
        present_phrases = find_stem_answer(word_list=lower_words, ans_list=ex['keyphrases'])
        if present_phrases is None:
            null_ids += 1
            continue
        if len(present_phrases['keyphrases']) != len(ex['keyphrases']):
            absent_ids += 1

        data = {}
        data['url'] = url
        data['doc_words'] = ex['doc_words']
        data['title'] = ex['title'],
        data['keyphrases'] = present_phrases['keyphrases']
        data['start_end_pos'] = present_phrases['start_end_pos']

        data_list.append(data)
        url += 1

    logger.info('Null : number = {} '.format(null_ids))
    logger.info('Absent : number = {} '.format(absent_ids))

    return data_list


def save_ground_truths(examples, filename, kp_key):
    with open(filename, 'w', encoding='utf-8') as f_pred:
        for ex in tqdm(examples):
            data = {}
            data['url'] = ex['url']
            data['KeyPhrases'] = ex[kp_key]
            f_pred.write("{}\n".format(json.dumps(data)))
        f_pred.close()
    logger.info('Success save reference to %s' % filename)


def save_preprocess_data(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as fo:
        for data in tqdm(data_list):
            fo.write("{}\n".format(json.dumps(data)))
        fo.close()
    logger.info("Success save file to %s \n" % filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_dataset', type=str, required=True,
                        help="The path to the source dataset (raw json).")
    parser.add_argument('--ground_truth_path', type=str, required=True,
                        help="The path to save preprocess data")
    parser.add_argument('--preprocess_path', type=str, required=True,
                        help="The path to save preprocess data")
    parser.add_argument('-max_src_seq_len', type=int, default=300,
                        help="Maximum document sequence length")
    parser.add_argument('-min_src_seq_len', type=int, default=20,
                        help="Minimum document sequence length")
    parser.add_argument('-max_trg_seq_len', type=int, default=6,
                        help="Maximum keyphrases sequence length to keep.")
    parser.add_argument('-min_trg_seq_len', type=int, default=0,
                        help="Minimun keyphrases sequence length to keep.")

    args = parser.parse_args()

    features = load_data(args.source_dataset)
    filtered_features = filter_absent_keyphrases(
        tokenize(features,
                 args.max_src_seq_len,
                 args.min_src_seq_len,
                 args.max_trg_seq_len,
                 args.min_trg_seq_len)
    )

    if 'train' not in args.ground_truth_path:
        save_ground_truths(filtered_features, args.ground_truth_path, 'keyphrases')

    save_preprocess_data(filtered_features, args.preprocess_path)
