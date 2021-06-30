
from text import splittoken2index, alltoken2index
import os

# CONFIG -----------------------------------------------------------------------------------------------------------#

# File Path
wav_path = "/aistore/asr/exec/yins/data/DB-4/all_16k"
text_dict_path = "/ssd3/exec/yins/data/english_data/text_dicts_path/ljspeech_dict_text.pkl"

# Settings for all models
mel_type = ["lpc", "mel"][1]
pre_mel = False

# Setting Saving path
phones_path = "/ssd3/exec/chenhanying/espnet/egs2/db4/tts6/dump/lpcnet_feat/org/tr_no_dev/text"
mels_path = "/ssd4/exec/yins/work/vocoders/ParallelWaveGAN/egs/db4/dump/train_nodev/raw/dump.1" \
    if mel_type == "mel" else "/ssd3/exec/yins/db4/biaobei_progressive/lpc_features"

checkpoint_path = "checkpoints_db4_fa_mel_cbhg_splittoken_r2_small"
gta_path = os.path.join(checkpoint_path, "gta")


################################
# Audio Parameters             #
################################
max_wav_value=32768.0
sampling_rate=16000
filter_length=1024
hop_length=256
win_length=1024
mel_fmin=80.0
mel_fmax=7600.0
compression = ["log", "log10"][0]  # must be one of them

################################
# Encoder Network parameters   #
################################
split_tone = True
num_chars = len(splittoken2index) if split_tone else len(alltoken2index)
encoder_kernel_size=5
encoder_n_convolutions=3
encoder_embedding_dim=256
tone_embedding = encoder_embedding_dim//8

#############################################
# Reference Encoder Network Hyperparameters #
#############################################
speaker_encoder_type = ["GST", "VAE", "GMVAE"][1]
expressive_encoder_type = ["GST", "VAE", "GMVAE"][1]

spk_ids = {"us": 0,
           "**": 1}
speaker_classes = len(spk_ids)

emotioned = False
emotion_classes = speaker_classes

cat_lambda = 0.0
cat_incr = 0.01
cat_step = 1000
cat_step_after = 20
cat_max_step = 300000

kl_lambda = 0.00001
kl_incr = 0.000001
kl_step = 1000
kl_step_after = 500
kl_max_step = 300000

# reference_encoder
ref_enc_filters=[32, 32, 64, 64, 128, 128]
ref_enc_size=[3, 3]
ref_enc_strides=[2, 2]
ref_enc_pad=[1, 1]
ref_enc_gru_size=128

# Style Token Layer
token_num=10
num_heads=8

# embedding size
token_embedding_size=256
speaker_embedding_size=64
vae_size=32

################################
# Decoder Network parameters   #
################################
feed_back_last=True
n_mel_channels={"lpc": 20, "mel": 80}[mel_type]
n_frames_per_step=2
decoder_rnn_dim=384
prenet_dims=[256, 128]
gate_threshold=0.5
max_decoder_steps=1000
p_attention_dropout=0.1
p_decoder_dropout=0.1

#################################
# Attention Network parameters  #
#################################
attention_mode=["GMM", "FAV2"][1]
attention_rnn_dim=384
attention_dim=128

# Location Layer parameters
attention_location_n_filters=32
attention_location_kernel_size=17

# GMM parameters
delta_bias=1.0
sigma_bias=10.0
gmm_kernel=5

################################
# Auxiliary Loss parameters    #
################################
guided_sigma=0.4                # weight for guided attention loss. default 0.4
pos_weight=10.0                 # BCEWithLogitsLoss pos_weight, default 10.0

################################
# Mel-post Network parameters  #
################################
postnet_embedding_dims=[256, 256, 256, 256]
postnet_kernel_sizes=[3, 3, 5, 5]
p_postnet_dropout=0.5

postnet_k=5
postnet_num_highways=4
post_projections=[encoder_embedding_dim, encoder_embedding_dim]

################################
# Training parameters          #
################################
distributed_run=True
dist_backend="nccl"
dist_url="tcp://localhost:43021"
seed=1234
dynamic_loss_scaling=True
batch_size=32
learning_rate=1e-3
weight_decay=1e-6
lr_decay=0.999
training_steps=200_000
epochs=500
grad_clip_thresh=1.0
save_checkpoint_every_n_step=1_000 if distributed_run else 10_000
# ------------------------------------------------------------------------------------------------------------------#
