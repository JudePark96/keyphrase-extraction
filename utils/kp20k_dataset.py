__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from torch.utils.data import Dataset


class KP20KDataset(Dataset):
    def __init__(self, features: list) -> None:
        super(KP20KDataset, self).__init__()
        self.features = features

    def __getitem__(self, index: int) -> None:
        return self.features[index]

    def __len__(self) -> int:
        return len(self.features)