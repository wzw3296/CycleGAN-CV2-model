import os
import argparse
import numpy as np
import pickle
import preprocess

def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)

def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)

def data_preprocess(src_folder, target_folder, cache_folder):
    num_mcep = 36
    sampling_rate = 16000
    frame_period = 5.0
    n_frames = 128

    src_audio = preprocess.load_wavs(wav_dir=src_folder, sr=sampling_rate)
    target_audio = preprocess.load_wavs(wav_dir=target_folder, sr=sampling_rate)

    src_f0s, _, _, _, src_coded = preprocess.world_encode_data(
        wave=src_audio, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    target_f0s, _, _, _, target_coded = preprocess.world_encode_data(
        wave=target_audio, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)

    src_logf0_mean, src_logf0_std = preprocess.logf0_statistics(f0s=src_f0s)
    target_logf0_mean, target_logf0_std = preprocess.logf0_statistics(f0s=target_f0s)

    src_coded_transposed = preprocess.transpose_in_list(lst=src_coded)
    target_coded_transposed = preprocess.transpose_in_list(lst=target_coded)

    src_coded_norm, src_coded_mean, src_coded_std = preprocess.coded_sps_normalization_fit_transform(
        coded_sps=src_coded_transposed)
    target_coded_norm, target_coded_mean, target_coded_std = preprocess.coded_sps_normalization_fit_transform(
        coded_sps=target_coded_transposed)

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    np.savez(os.path.join(cache_folder, 'logf0s_normalization.npz'),
             mean_A=src_logf0_mean,
             std_A=src_logf0_std,
             mean_B=target_logf0_mean,
             std_B=target_logf0_std)

    np.savez(os.path.join(cache_folder, 'mcep_normalization.npz'),
             mean_A=src_coded_mean,
             std_A=src_coded_std,
             mean_B=target_coded_mean,
             std_B=target_coded_std)

    save_pickle(variable=src_coded_norm,
                fileName=os.path.join(cache_folder, "src_coded_norm.pickle"))
    save_pickle(variable=target_coded_norm,
                fileName=os.path.join(cache_folder, "target_coded_norm.pickle"))

if __name__ == '__main__':
    src_folder = './data/sam/'
    target_folder = './data/bea/'
    cache_folder = './cache/'
    data_preprocess(src_folder, target_folder, cache_folder)
