__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import h5py


file_path = '../rsc/features/kp20k.feature.train.256.32.hdf5'


with h5py.File(file_path, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    print(f['doc/input_ids'])
    print(f['title/input_ids'])
    print(f.get('doc/input_ids').shape)
    print(f['label/start_pos'])

    """
    Keys: <KeysViewHDF5 ['doc', 'label', 'title']>
    <HDF5 dataset "input_ids": shape (465744, 256), type "<i8">
    <HDF5 dataset "input_ids": shape (465744, 32), type "<i8">
    """
