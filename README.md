# Expressive Tacotron (implementation with Pytorch)

## Introduction

The expressive Tacotron framework includes various deep learning architectures such as **Global Style Token (GST)**, **Variational Autoencoder (VAE)**, and **Gaussian Mixture Variational Autoencoder (GMVAE)**, and **X-vectors** for building prosody encoder.

## Available recipes
### Expressive Mode
- [x] [Global Style Token (GST)](https://arxiv.org/abs/1803.09017)
- [x] [Variational Autoencoder (VAE)](https://arxiv.org/abs/1812.04342)
- [x] [Gaussian Mixture VAE (GMVAE)](https://arxiv.org/abs/1810.07217)
- [x] X-vectors

### Attention Mode
- [x] [Guided Attention](https://arxiv.org/abs/1710.08969)
- [x] [Tacotron2](https://arxiv.org/pdf/1712.05884.pdf)
- [x] [Forward Attention](https://arxiv.org/abs/1807.06736)
- [x] [GMMv2 Attention](https://arxiv.org/pdf/1910.10288.pdf)
- [ ] [Dynamic Convolution Attention](https://arxiv.org/pdf/1910.10288.pdf) (Todo)

## Training
Single Tacotron2 with Forward Attention by defalut. If you want to train with expressive mode, you can reference [Expressive Tacotron](https://github.com/BridgetteSong/ExpressiveTacotron/blob/master/model_attention.py).
1. transfer texts to phones, and save as **"phones_path"** in **hparams.py** and change phone dictionary in **text.py**
2. `python train.py` for single GPU
3. `python -m multiproc train.py` for multi GPUs

## Inference Demo
1. `python synthesis.py -w checkpoints/checkpoint_200k_steps.pyt -i "hello word" --vocoder gl`

Default **Griffin_Lim** Vocoder. For other command line options, please refer to the `synthesis.py` section.

## Acknowledgements
This implementation uses code from the following repos: [NVIDIA](https://github.com/NVIDIA/tacotron2), [MozillaTTS](https://github.com/mozilla/TTS), [ESPNet](https://github.com/espnet/espnet), [ERISHA](https://github.com/ajinkyakulkarni14/ERISHA), [ForwardAttention](https://github.com/jxzhanggg/nonparaSeq2seqVC_code/blob/master/pre-train/model/basic_layers.py)

