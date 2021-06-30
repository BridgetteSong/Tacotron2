import os

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import hparams as hp
import numpy as np

from utils import load_wav, mel_spectrogram_torch
from text import splittoken2index, alltoken2index

token2index = splittoken2index if hp.split_tone else alltoken2index

def get_tts_dataset(force_gta):
    phone_files = {}
    with open(os.path.join(hp.phones_path), "r") as f:
        for line in f:
            name_context = line.split()
            name = name_context[0]
            token_index = [token2index["<sos/eos>"]]
            tones = [0]
            for token in name_context[1:]:
                if hp.split_tone and "sp" not in token and token[-1] in {"1", "2", "3", "4", "5"}:
                    # split tones
                    token_index.append(token2index[token[:-1]])
                    tones.append(int(token[-1]))
                else:
                    token_index.append(token2index[token])
                    tones.append(0)
            token_index.append(token2index["<sos/eos>"])
            tones.append(0)
            phone_files[name[4:] + ".npy"] = np.stack((np.array(token_index), np.array(tones)), axis=0) if hp.split_tone else np.array(token_index)


    mel_files = {}
    if hp.pre_mel:
        print("loading mels from disk")
        for features in os.listdir(hp.mels_path):
            if features[-10:-4] == "-feats":
                # from parallel wavegan
                mel_files[features[:-10] + ".npy"] = features
            else:
                mel_files[features] = features

    wav_files = set([wav[:-4] + ".npy" for wav in os.listdir(hp.wav_path) if wav[-4:] == ".wav"])
    dataset_ids = []
    for ids in phone_files.keys():
        if hp.pre_mel and ids in mel_files:
            dataset_ids.append(ids)
        elif not hp.pre_mel and ids in wav_files:
            dataset_ids.append(ids)

    train_dataset = TTSDataset(dataset_ids, phone_files, mel_files)

    sampler = DistributedSampler(train_dataset) if not force_gta and hp.distributed_run else None
    shuffle = False if not force_gta and hp.distributed_run else True

    train_set = DataLoader(train_dataset,
                           collate_fn=lambda batch: collate_fn(batch),
                           batch_size=hp.batch_size,
                           sampler=sampler,
                           shuffle=shuffle,
                           num_workers=1,
                           pin_memory=True)

    return train_set


class TTSDataset(Dataset):
    def __init__(self, dataset_ids, phone_files, mel_files):
        self.phone_files = phone_files
        self.mel_files = mel_files
        self.metadata = dataset_ids
        self.spk_ids = hp.spk_ids
        print("len(dataset) {}".format(len(dataset_ids)))
        self.phone_path = hp.phones_path
        self.mel_path = hp.mels_path

    def __getitem__(self, index):
        idx = self.metadata[index]
        phones = np.load(os.path.join(self.phone_path, idx)) if len(self.phone_files) == 0 else self.phone_files[idx]
        if hp.pre_mel and len(self.mel_files) > 0:
            mel = np.load(os.path.join(self.mel_path, self.mel_files[idx]))
            if hp.n_mel_channels == mel.shape[1] and mel.shape[0] != hp.n_mel_channels: mel = mel.T
        else:
            assert hp.n_mel_channels == 80
            audio, sampling_rate = load_wav(os.path.join(hp.wav_path, idx[:-4] + ".wav"))
            if sampling_rate != hp.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, hp.sampling_rate))
            audio = audio / hp.max_wav_value
            audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            mel = mel_spectrogram_torch(audio, hp.filter_length, hp.n_mel_channels, hp.sampling_rate,
                                        hp.hop_length, hp.win_length, hp.mel_fmin, hp.mel_fmax, compression=hp.compression)
            mel = mel.squeeze(0).numpy()

        speaker_id = self.spk_ids[idx[:2]] if idx[:2] in self.spk_ids else len(self.spk_ids) - 1
        assert mel.shape[0] == hp.n_mel_channels

        return idx, phones, mel, speaker_id

    def __len__(self):
        return len(self.metadata)


def collate_fn(batch):
    input_lengths, sort_idx = torch.sort(
        torch.tensor([x[1].shape[-1] for x in batch], dtype=torch.long),
        dim=0, descending=True)
    max_x_len = input_lengths[0]
    spec_lens = [x[2].shape[-1] for x in batch]

    max_spec_len = max(spec_lens)
    if max_spec_len % hp.n_frames_per_step != 0:
        max_spec_len += hp.n_frames_per_step - max_spec_len % hp.n_frames_per_step

    # ----------------------- padding ----------------------------
    phone_pad = torch.zeros([len(batch), 2, max_x_len], dtype=torch.long) \
        if hp.split_tone else torch.zeros([len(batch), max_x_len], dtype=torch.long)
    mels_pad = torch.zeros([len(batch), batch[0][2].shape[0], max_spec_len])
    gate_padded = torch.zeros([len(batch), max_spec_len])

    ids = []
    mel_lengths = []
    speaker_ids = []

    for i in range(sort_idx.size(0)):
        ids.append(batch[sort_idx[i]][0])
        phones = batch[sort_idx[i]][1]
        if hp.split_tone:
            phone_pad[i, :, :phones.shape[-1]] = torch.from_numpy(phones)
        else:
            phone_pad[i, :phones.shape[-1]] = torch.from_numpy(phones)
        mel = batch[sort_idx[i]][2]
        mels_pad[i, :, :mel.shape[1]] = torch.from_numpy(mel)
        mel_lengths.append(mel.shape[1])
        gate_padded[i, mel.shape[1] - 1:] = 1.0
        speaker_ids.append(batch[sort_idx[i]][3])

    mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)

    return ids, phone_pad, mels_pad, gate_padded, speaker_ids, input_lengths, mel_lengths
