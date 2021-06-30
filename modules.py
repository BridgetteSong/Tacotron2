# modules.py includes various encoders, GST, VAE, GMVAE, X-vectors
# adapted from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
# MIT License
#
# Copyright (c) 2018 MagicGirl Sakura
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import math

from gmvae import GMVAENet


class SpeakerEncoderNetwork(nn.Module):
    def __init__(self, hp):
        super().__init__()
        if hp.speaker_encoder_type.lower() == 'gst':
            self.encoder = GST(hp, hp.speaker_classes)
        elif hp.speaker_encoder_type.lower() == 'vae':
            self.encoder = VAE(hp, hp.speaker_classes)
        elif hp.speaker_encoder_type.lower() == 'gst_vae':
            self.encoder = GST_VAE(hp, hp.speaker_classes)
        elif hp.speaker_encoder_type.lower() == 'gmvae':
            self.encoder = GMVAE(hp, hp.speaker_classes)
        elif hp.speaker_encoder_type.lower() == 'x-vector':
            self.encoder = X_vector(hp, hp.speaker_classes)
        elif hp.speaker_encoder_type.lower() == 'vqvae':
            raise ValueError("Error: unsupported type of 'vqvae'")
        else:
            raise ValueError("Erroe: unsupported type of 'speaker encoder'")

    def forward(self, inputs, input_lengths=None):

        embedding, cat_prob = self.encoder(inputs, input_lengths)

        return (embedding, cat_prob)


class ExpressiveEncoderNetwork(nn.Module):
    def __init__(self, hp):
        super().__init__()
        if hp.expressive_encoder_type.lower() == 'gst':
            self.encoder = GST(hp, hp.emotion_classes)
        elif hp.expressive_encoder_type.lower() == 'vae':
            self.encoder = VAE(hp, hp.emotion_classes)
        elif hp.expressive_encoder_type.lower() == 'gst_vae':
            self.encoder = GST_VAE(hp, hp.emotion_classes)
        elif hp.expressive_encoder_type.lower() == 'gmvae':
            self.encoder = GMVAE(hp, hp.emotion_classes)
        elif hp.expressive_encoder_type.lower() == 'x-vector':
            self.encoder = X_vector(hp, hp.emotion_classes)
        elif hp.expressive_encoder_type.lower() == 'vqvae':
            raise ValueError("Error: unsupported type of 'vqvae'")
        else:
            raise ValueError("Erroe: unsupported type of 'speaker encoder'")

    def forward(self, inputs, input_lengths=None):

        embedding, cat_prob = self.encoder(inputs, input_lengths)

        return (embedding, cat_prob)


class LanguageEncoderNetwork(nn.Module):
    def __init__(self, hp):
        super().__init__()
        if hp.language_encoder_type == 'gst':
            self.encoder = GST(hp, hp.language_classes)
        elif hp.language_encoder_type == 'vae':
            self.encoder = VAE(hp, hp.language_classes)
        elif hp.language_encoder_type == 'gst_vae':
            self.encoder = GST_VAE(hp, hp.language_classes)
        elif hp.language_encoder_type == 'gmvae':
            self.encoder = GMVAE(hp, hp.language_classes)
        elif hp.language_encoder_type == 'x-vector':
            self.encoder = X_vector(hp, hp.language_classes)
        elif hp.language_encoder_type == 'vqvae':
            pass  # self.encoder =

    def forward(self, inputs, input_lengths=None):

        embedding, cat_prob = self.encoder(inputs, input_lengths)

        return (embedding, cat_prob)


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, hp):

        super().__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hp.ref_enc_filters[i])
             for i in range(K)])

        out_channels = self.calculate_channels(hp.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.ref_enc_gru_size,
                          batch_first=True)
        self.n_mel_channels = hp.n_mel_channels
        self.ref_enc_gru_size = hp.ref_enc_gru_size

    def forward(self, inputs, input_lengths=None):
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        if input_lengths is not None:
            # print(input_lengths.cpu().numpy(), 2, len(self.convs))
            input_lengths = (input_lengths.cpu().numpy() / 2 ** len(self.convs))
            input_lengths = max(input_lengths.round().astype(int), [1])
            # print(input_lengths, 'input lengths')
            out = nn.utils.rnn.pack_padded_sequence(
                out, input_lengths, batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    """
    inputs --- [N, token_embedding_size//2]
    """

    def __init__(self, hp):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num, hp.token_embedding_size // hp.num_heads))
        d_q = hp.token_embedding_size // 2
        d_k = hp.token_embedding_size // hp.num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=hp.token_embedding_size,
            num_heads=hp.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1) # [N, token_num, token_embedding_size//num_heads]
        # print(query.shape, keys.shape)
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    """
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class GST(nn.Module):
    def __init__(self, hp, classes_):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)
        self.stl = STL(hp)

        self.categorical_layer = nn.Linear(hp.token_embedding_size, classes_)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        # print(enc_out.shape)
        style_embed = self.stl(enc_out)

        cat_prob = F.softmax(self.categorical_layer(style_embed.squeeze(0)), dim=-1)
        # print(style_embed.shape, cat_prob.shape)
        return (style_embed.squeeze(0), cat_prob)


class GST_VAE(nn.Module):
    def __init__(self, hp, classes_):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)
        self.stl = STL(hp)

        self.mean_linear = nn.Linear(hp.token_embedding_size, hp.token_embedding_size)
        self.logvar_linear = nn.Linear(hp.token_embedding_size, hp.token_embedding_size)
        self.categorical_layer = nn.Linear(hp.token_embedding_size, classes_)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        style_embed = self.stl(enc_out)

        latent_mean = self.mean_linear(style_embed)
        latent_logvar = self.logvar_linear(style_embed)
        std = torch.exp(0.5 * latent_logvar)
        eps = torch.randn_like(std)
        # ze = eps.mul(std).add_(latent_mean)
        ze = eps * std + latent_mean

        cat_prob = F.softmax(self.categorical_layer(ze), dim=-1)
        # print(ze.unsqueeze(0).shape, cat_prob.shape)
        return (ze, (latent_mean, latent_logvar, cat_prob))


class VAE(nn.Module):
    def __init__(self, hp, classes_):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)

        self.mean_linear = nn.Linear(hp.ref_enc_gru_size, hp.token_embedding_size)
        self.logvar_linear = nn.Linear(hp.ref_enc_gru_size, hp.token_embedding_size)
        self.categorical_layer = nn.Linear(hp.token_embedding_size, classes_)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)

        latent_mean = self.mean_linear(enc_out)
        latent_logvar = self.logvar_linear(enc_out)
        std = torch.exp(0.5 * latent_logvar)
        eps = torch.randn_like(std)
        # z = eps.mul(std).add_(latent_mean)
        z = eps * std + latent_mean
        cat_prob = F.softmax(self.categorical_layer(z), dim=-1)

        return (z, (latent_mean, latent_logvar, cat_prob))


class GMVAE(nn.Module):
    def __init__(self, hp, classes_):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)
        self.gmvae = GMVAENet(hp.ref_enc_gru_size, hp.token_embedding_size, classes_)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)

        (z, (z, mu, var, y_mu, y_var, prob, logits)) = self.gmvae(enc_out)
        # print(out['prob_cat'].shape, out['logits'].shape)

        return (z, (z, mu, var, y_mu, y_var, prob, logits))


class X_vector(nn.Module):
    def __init__(self, hp, num_classes):
        super(X_vector, self).__init__()

        self.input_dim = hp.input_dim
        self.output_dim = hp.output_dim
        self.num_classes = num_classes
        self.layer1 = TDNN_cpu([-2, 2], self.input_dim, self.output_dim, full_context=True)
        self.layer2 = TDNN_cpu([-2, 1, 2], self.output_dim, self.output_dim, full_context=True)
        self.layer3 = TDNN_cpu([-3, 1, 3], self.output_dim, self.output_dim, full_context=True)
        self.layer4 = TDNN_cpu([1], self.output_dim, self.output_dim, full_context=True)
        self.layer5 = TDNN_cpu([1], self.output_dim, 1500, full_context=True)
        self.statpool_layer = StatsPooling()
        self.FF = FullyConnected(self.output_dim)
        self.last_layer = nn.Linear(self.output_dim, self.num_classes)

    def forward(self, x, input_lengths=None):
        x = x.permute(0, 2, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.statpool_layer(x)
        embedding = self.FF(x)
        prob_ = self.last_layer(embedding)

        return embedding, prob_


class StatsPooling(nn.Module):
    def __init__(self):
        super(StatsPooling, self).__init__()

    def forward(self, varient_length_tensor):
        mean = varient_length_tensor.mean(dim=1)
        std = varient_length_tensor.std(dim=1)
        return torch.cat((mean, std), dim=1)


class FullyConnected(nn.Module):
    def __init__(self, out_dim):
        super(FullyConnected, self).__init__()
        self.hidden1 = nn.Linear(3000, 512)
        self.hidden2 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return x


"""Time Delay Neural Network as mentioned in the 1989 paper by Waibel et al. (Hinton) and the 2015 paper by Peddinti et al. (Povey)"""


class TDNN(nn.Module):
    def __init__(self, context, input_dim, output_dim, full_context=True, device='cpu'):
        """
        Definition of context is the same as the way it's defined in the Peddinti paper. It's a list of integers, eg: [-2,2]
        By deault, full context is chosen, which means: [-2,2] will be expanded to [-2,-1,0,1,2] i.e. range(-2,3)
        """
        super(TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.check_valid_context(context)
        self.kernel_width, context = self.get_kernel_width(context, full_context)
        self.register_buffer('context', torch.LongTensor(context))
        self.full_context = full_context
        stdv = 1. / math.sqrt(input_dim)
        self.kernel = nn.Parameter(torch.Tensor(output_dim, input_dim, self.kernel_width).normal_(0, stdv)).cuda()
        self.bias = nn.Parameter(torch.Tensor(output_dim).normal_(0, stdv)).cuda()
        # self.cuda_flag = False

    def forward(self, x):
        """
        x is one batch of data
        x.shape: [batch_size, sequence_length, input_dim]
        sequence length is the length of the input spectral data (number of frames) or if already passed through the convolutional network, it's the number of learned features

        output size: [batch_size, output_dim, len(valid_steps)]
        """
        # Check if parameters are cuda type and change context
        # if type(self.bias.data) == torch.cuda.FloatTensor and self.cuda_flag == False:
        #     self.context = self.context.cuda()
        #     self.cuda_flag = True
        conv_out = self.special_convolution(x, self.kernel, self.context, self.bias)
        activation = F.relu(conv_out).transpose(1, 2).contiguous()
        # print ('output shape: {}'.format(activation.shape))
        return activation

    def special_convolution(self, x, kernel, context, bias):
        """
        This function performs the weight multiplication given an arbitrary context. Cannot directly use convolution because in case of only particular frames of context,
        one needs to select only those frames and perform a convolution across all batch items and all output dimensions of the kernel.
        """
        input_size = x.shape
        assert len(input_size) == 3, 'Input tensor dimensionality is incorrect. Should be a 3D tensor'
        [batch_size, input_sequence_length, input_dim] = input_size
        # print ('mel size: {}'.format(input_dim))
        # print ('sequence length: {}'.format(input_sequence_length))

        x = x.transpose(1, 2).contiguous()

        # Allocate memory for output
        valid_steps = self.get_valid_steps(self.context, input_sequence_length)
        xs = Variable(self.bias.data.new(batch_size, kernel.shape[0], len(valid_steps)))

        # Perform the convolution with relevant input frames
        for c, i in enumerate(valid_steps):
            features = torch.index_select(x, 2, context + i)
            # print ('features taken:{}'.format(features))
            xs[:, :, c] = F.conv1d(features, kernel, bias=bias)[:, :, 0]
        return xs

    @staticmethod
    def check_valid_context(context):
        # here context is still a list
        assert context[0] <= context[-1], 'Input tensor dimensionality is incorrect. Should be a 3D tensor'

    @staticmethod
    def get_kernel_width(context, full_context):
        if full_context:
            context = range(context[0], context[-1] + 1)
        return len(context), context

    @staticmethod
    def get_valid_steps(context, input_sequence_length):
        start = 0 if context[0] >= 0 else -1 * context[0]
        end = input_sequence_length if context[-1] <= 0 else input_sequence_length - context[-1]
        return range(start, end)


"""Time Delay Neural Network as mentioned in the 1989 paper by Waibel et al. (Hinton) and the 2015 paper by Peddinti et al. (Povey)"""


class TDNN_cpu(nn.Module):
    def __init__(self, context, input_dim, output_dim, full_context=True, device='cpu'):
        """
        Definition of context is the same as the way it's defined in the Peddinti paper. It's a list of integers, eg: [-2,2]
        By deault, full context is chosen, which means: [-2,2] will be expanded to [-2,-1,0,1,2] i.e. range(-2,3)
        """
        super(TDNN_cpu, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.check_valid_context(context)
        self.kernel_width, context = self.get_kernel_width(context, full_context)
        self.register_buffer('context', torch.LongTensor(context))
        self.full_context = full_context
        stdv = 1. / math.sqrt(input_dim)
        self.kernel = nn.Parameter(torch.Tensor(output_dim, input_dim, self.kernel_width).normal_(0, stdv))
        self.bias = nn.Parameter(torch.Tensor(output_dim).normal_(0, stdv))
        # self.cuda_flag = False

    def forward(self, x):
        """
        x is one batch of data
        x.shape: [batch_size, sequence_length, input_dim]
        sequence length is the length of the input spectral data (number of frames) or if already passed through the convolutional network, it's the number of learned features

        output size: [batch_size, output_dim, len(valid_steps)]
        """
        # Check if parameters are cuda type and change context
        # if type(self.bias.data) == torch.cuda.FloatTensor and self.cuda_flag == False:
        #     self.context = self.context.cuda()
        #     self.cuda_flag = True
        conv_out = self.special_convolution(x, self.kernel, self.context, self.bias)
        activation = F.relu(conv_out).transpose(1, 2).contiguous()
        # print ('output shape: {}'.format(activation.shape))
        return activation

    def special_convolution(self, x, kernel, context, bias):
        """
        This function performs the weight multiplication given an arbitrary context. Cannot directly use convolution because in case of only particular frames of context,
        one needs to select only those frames and perform a convolution across all batch items and all output dimensions of the kernel.
        """
        input_size = x.shape
        assert len(input_size) == 3, 'Input tensor dimensionality is incorrect. Should be a 3D tensor'
        [batch_size, input_sequence_length, input_dim] = input_size
        # print ('mel size: {}'.format(input_dim))
        # print ('sequence length: {}'.format(input_sequence_length))

        x = x.transpose(1, 2).contiguous()

        # Allocate memory for output
        valid_steps = self.get_valid_steps(self.context, input_sequence_length)
        xs = Variable(self.bias.data.new(batch_size, kernel.shape[0], len(valid_steps)))

        # Perform the convolution with relevant input frames
        for c, i in enumerate(valid_steps):
            features = torch.index_select(x, 2, context + i)
            # print ('features taken:{}'.format(features))
            xs[:, :, c] = F.conv1d(features, kernel, bias=bias)[:, :, 0]
        return xs

    @staticmethod
    def check_valid_context(context):
        # here context is still a list
        assert context[0] <= context[-1], 'Input tensor dimensionality is incorrect. Should be a 3D tensor'

    @staticmethod
    def get_kernel_width(context, full_context):
        if full_context:
            context = range(context[0], context[-1] + 1)
        return len(context), context

    @staticmethod
    def get_valid_steps(context, input_sequence_length):
        start = 0 if context[0] >= 0 else -1 * context[0]
        end = input_sequence_length if context[-1] <= 0 else input_sequence_length - context[-1]
        return range(start, end)