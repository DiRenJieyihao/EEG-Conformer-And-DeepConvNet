from braindecode.datasets import MOABBDataset
import numpy as np
from braindecode.preprocessing import (exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.preprocessing import create_windows_from_events
import argparse


def data_preprocess(dataset):
    low_cut_hz = 0.  # low cut frequency for filtering
    high_cut_hz = 38.  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                     factor_new=factor_new, init_block_size=init_block_size)
    ]

    # Transform the data
    preprocess(dataset, preprocessors)

    trial_start_offset_seconds = 0.
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
    )

    return windows_dataset


def save_data(windows_dataset, subject_id):
    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']
    valid_set = splitted['session_E']

    # label_dict = train_set.datasets[0].windows.event_id.items()
    # labels = list(dict(sorted(list(label_dict), key=lambda kv: kv[1])).keys())
    #
    # print("label_dict:{}".format(label_dict))
    # print("labels:{}".format(labels))

    train_x_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []
    for tl in train_set:
        train_x_list.append(tl[0])
        train_y_list.append(tl[1])

    for tl in valid_set:
        test_x_list.append(tl[0])
        test_y_list.append(tl[1])

    X_train = np.array(train_x_list)
    Y_train = np.array(train_y_list)
    X_test = np.array(test_x_list)
    Y_test = np.array(test_y_list)

    print("X_train.shape:{}".format(X_train.shape))
    print("Y_train.shape:{}".format(Y_train.shape))
    print("X_test.shape:{}".format(X_test.shape))
    print("Y_test.shape:{}".format(Y_test.shape))

    np.save("./data/BCI/X{}_train.npy".format(subject_id), X_train, allow_pickle=True)
    np.save("./data/BCI/Y{}_train.npy".format(subject_id), Y_train, allow_pickle=True)
    np.save("./data/BCI/X{}_test.npy".format(subject_id), X_test, allow_pickle=True)
    np.save("./data/BCI/Y{}_test.npy".format(subject_id), Y_test, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bci data extract')
    parser.add_argument('--subject_id', type=int)
    args = parser.parse_args()

    subject_id = args.subject_id
    print("subject_id:{}".format(subject_id))
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

    windows_dataset = data_preprocess(dataset)
    save_data(windows_dataset, subject_id=subject_id)


