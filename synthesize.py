
import torch
import hparams as hp
from dataset import token2index
from model import Tacotron2
import argparse
import os
import numpy as np
import time
import soundfile as sf

from utils import reconstruct_waveform


def synthesis_text(model, input_text, spkid):
    token_index = [token2index["<sos/eos>"]]
    tones = [0]
    for token in input_text:
        if hp.split_tone and "sp" not in token and token[-1] in {"1", "2", "3", "4", "5"}:
            # split tones
            token_index.append(token2index[token[:-1]])
            tones.append(int(token[-1]))
        else:
            token_index.append(token2index[token])
            tones.append(0)
    token_index.append(token2index["<sos/eos>"])
    tones.append(0)
    inputs = np.stack((np.array(token_index), np.array(tones)), axis=0) if hp.split_tone else np.array(token_index)
    inputs_index = torch.tensor([inputs], dtype=torch.long).to(device)
    _, mel_outputs_postnet = model.inference(inputs_index, spkid)
    print("mel_outputs_postnet.size(): ", mel_outputs_postnet.size())

    return mel_outputs_postnet


def gen_mel(mel_output, output_dir, filename, vocoder):
    if vocoder == "gl":
        wav = reconstruct_waveform(mel_output.contiguous().view(hp.n_mel_channels, mel_output.size(-1)), compression=hp.compression)
        sf.write(os.path.join(output_dir, filename + "_gl.wav"), wav, hp.sampling_rate, "PCM_16")
    else:
        np.save(os.path.join(output_dir, filename + ".npy"), mel_output.data.cpu().numpy())


if __name__ == "__main__" :
    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', default=None, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--weights_path', '-w', required=True, help='[string/path] Load in different Tacotron Weights')
    parser.add_argument('--name', '-n', default="test", help='saved audio name to be synthesized')
    parser.add_argument('--file', '-f', default=None, help='Speaker name to be synthesized')
    parser.add_argument("--output", "-o", default="file_outputs", type=str, help="output dir")
    parser.add_argument("--device", "-d", default="gpu", choices=["gpu", "cpu"], type=str, help="device type")
    parser.add_argument("--vocoder", default="gl", choices=["gl", "lpcnet", "hifigan", "pwg"], type=str, help="vocoder type")
    parser.add_argument("--spk", default=0, type=int, choices=[0,1,2,3], help="speaker id")
    args = parser.parse_args()


    global device
    device = torch.device('cuda') if torch.cuda.is_available() and args.device == "gpu" else torch.device('cpu')

    model = Tacotron2(hp).to(device)
    checkpoint_dict = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.eval()

    if args.file is not None:
        os.makedirs(args.output, exist_ok=True)
        with open(args.file) as f:
            for ii, line in enumerate(f):
                name_input_text = line.strip().split()
                name = name_input_text[0]
                input_text = name_input_text[1:]
                # if name + ".wav" in set(os.listdir(output)): continue
                tstime = time.time()
                mel_output = synthesis_text(model, input_text, spkid=0)
                tetime = time.time()
                print("tacotron rtf: ", np.round((tetime - tstime) * hp.sampling_rate / (hp.hop_length * mel_output.shape[-1]), 4))
                gen_mel(mel_output=mel_output, output_dir=args.output, filename=name, vocoder=args.vocoder)

    else:
        tstime = time.time()
        mel_output = synthesis_text(model, args.input_text, spkid=0)
        tetime = time.time()    
        print("tacotron rtf: ", np.round((tetime - tstime) * hp.sampling_rate / (hp.hop_length * mel_output.shape[-1]), 4))
        gen_mel(mel_output=mel_output, output_dir="", filename=args.name, vocoder=args.vocoder)
    print('Done.\n')
