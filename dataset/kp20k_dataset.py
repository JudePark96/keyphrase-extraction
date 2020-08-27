__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import h5py


from models.span_extraction_model import SpanClassifier
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader
from typing import Dict


class KP20KTrainingDataset(Dataset):
    def __init__(self, source_file: str) -> None:
        super(KP20KTrainingDataset, self).__init__()
        self.source_file = source_file

        with h5py.File(self.source_file, 'r') as features_hdf:
            self.feature_keys = list(features_hdf.keys())
            self.num_instances = features_hdf.get('doc/input_ids').shape[0]

            features_hdf.close()

        print('total examples : %d' % self.num_instances)

    def __getitem__(self, index: int) -> Dict:
        return self.read_hdf_features(index)

    def __len__(self) -> int:
        return self.num_instances

    def read_hdf_features(self, index: int) -> Dict:
        feature = {}

        with h5py.File(self.source_file, 'r') as features_hdf:
            feature['doc'] = {
                'input_ids': features_hdf['doc/input_ids'][index],
                'token_type_ids': features_hdf['doc/token_type_ids'][index],
                'attention_mask': features_hdf['doc/attention_mask'][index]
            }

            feature['title'] = {
                'input_ids': features_hdf['title/input_ids'][index],
                'token_type_ids': features_hdf['title/token_type_ids'][index],
                'attention_mask': features_hdf['title/attention_mask'][index]
            }

            feature['start_pos'] = features_hdf['label/start_pos'][index]
            feature['end_pos'] = features_hdf['label/end_pos'][index]

            features_hdf.close()

        return feature


if __name__ == '__main__':
    dataset = KP20KTrainingDataset('../rsc/features/kp20k.feature.train.256.32.hdf5')
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=6)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = SpanClassifier(bert_model, 'baseline')

    for batch in loader:
        print(model(batch))
        break