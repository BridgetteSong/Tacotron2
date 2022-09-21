import torch
from torch import nn
from torch.nn import functional as F


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(max_len, dtype=torch.long).cuda()
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class ZoneOutCell(torch.nn.Module):
    """ZoneOut Cell module.
    This is a module of zoneout described in
    `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`_.
    This code is modified from `eladhoffer/seq2seq.pytorch`_.
    Examples:
        >> lstm = torch.nn.LSTMCell(16, 32)
        >> lstm = ZoneOutCell(lstm, 0.5)
    .. _`Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`:
        https://arxiv.org/abs/1606.01305
    .. _`eladhoffer/seq2seq.pytorch`:
        https://github.com/eladhoffer/seq2seq.pytorch
    """

    def __init__(self, cell, zoneout_rate=0.1):
        """Initialize zone out cell module.
        Args:
            cell (torch.nn.Module): Pytorch recurrent cell module
                e.g. `torch.nn.Module.LSTMCell`.
            zoneout_rate (float, optional): Probability of zoneout from 0.0 to 1.0.
        """
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_rate = zoneout_rate
        if zoneout_rate > 1.0 or zoneout_rate < 0.0:
            raise ValueError(
                "zoneout probability must be in the range from 0.0 to 1.0."
            )

    def forward(self, inputs, hidden):
        """Calculate forward propagation.
        Args:
            inputs (Tensor): Batch of input tensor (B, input_size).
            hidden (tuple):
                - Tensor: Batch of initial hidden states (B, hidden_size).
                - Tensor: Batch of initial cell states (B, hidden_size).
        Returns:
            tuple:
                - Tensor: Batch of next hidden states (B, hidden_size).
                - Tensor: Batch of next cell states (B, hidden_size).
        """
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout_rate)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple(
                [self._zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)]
            )

        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)
        self.W2.bias.data.fill_(-1.0)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1.0 - g) * x
        return y


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=True, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=True, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        """
        :param attention_rnn_dim: prenet(query) dims
        :param embedding_dim: encoder_seq dims
        :param attention_dim: attention dims
        :param attention_location_n_filters: conv number filters for previous alignmnents
        :param attention_location_kernel_size: conv kernel size
        """
        super(LocationSensitiveAttention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=True, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=True,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=True)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class GMMAttention(nn.Module):
    def __init__(self, query_dim, attention_dim, kernel, delta_bias, sigma_bias):
        super(GMMAttention, self).__init__()
        self.query_layer = LinearNorm(query_dim, attention_dim, bias=True, w_init_gain='relu')
        self.v = LinearNorm(attention_dim, 3*kernel, bias=True)
        torch.nn.init.constant_(self.v.linear_layer.bias[(1 * kernel):(2 * kernel)], delta_bias)  # bias mean
        torch.nn.init.constant_(self.v.linear_layer.bias[(2 * kernel):(3 * kernel)], sigma_bias)  # bias std

    def forward(self, query, memory, prev_mu, memory_time, mask=None):
        processed_query = self.v(F.relu(self.query_layer(query)))  # [B, 3*K]
        w_hat, delta_hat, sigma_hat = torch.chunk(processed_query, 3, dim=1)
        w = torch.softmax(w_hat, dim=1).unsqueeze(2)  # [B, k, 1]
        delta = F.softplus(delta_hat).unsqueeze(2) # [B, k, 1]
        sigma = F.softplus(sigma_hat).unsqueeze(2) # [B, k, 1]
        current_mu = prev_mu + delta
        z = math.sqrt(2*math.pi) * sigma  # [B, k, 1]
	log_energies = -torch.log(z) - 0.5 * (memory_time - current_mu)**2 / sigma**2  # [B, K, N]
        if mask is not None:
            log_energies.masked_fill_(mask.unsqueeze(1), -float(1e10))
        energies = w * F.softmax(log_energies, dim=-1)  # [B, K, N]
        alignments = torch.sum(energies, dim=1, keepdim=True)  # [B, 1, N]
        attention_context = torch.bmm(alignments.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return attention_context, alignments, current_mu


class ForwardAttentionV2(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(ForwardAttentionV2, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=True, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim,
                                       bias=True, w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=True)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float(1e4)

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat:  prev. and cumulative att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, log_alpha):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        log_energy = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            log_energy.data.masked_fill_(mask, self.score_mask_value)

        fwd_shifted_alpha = F.pad(log_alpha[:, :-1], [1, 0], 'constant', self.score_mask_value)
        biased = torch.logsumexp(torch.cat([log_alpha.unsqueeze(2), fwd_shifted_alpha.unsqueeze(2)], 2), 2)

        log_alpha_new = biased + log_energy
        attention_weights = F.softmax(log_alpha_new, dim=1)

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights, log_alpha_new


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=True)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.elu(linear(x)), p=0.5, training=True)
        return x


class ConvPostnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(ConvPostnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.post_dropout = hparams.p_postnet_dropout
        in_channels = [hparams.n_mel_channels] + hparams.postnet_embedding_dims[:-1]
        for i in range(len(in_channels)):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(in_channels[i], hparams.postnet_embedding_dims[i],
                             kernel_size=hparams.postnet_kernel_sizes[i], stride=1,
                             padding=int((hparams.postnet_kernel_sizes[i] - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dims[i]))
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dims[-1], hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_sizes[-1], stride=1,
                         padding=int((hparams.postnet_kernel_sizes[-1] - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), self.post_dropout, self.training)
        x = F.dropout(self.convolutions[-1](x), self.post_dropout, self.training)
        return x


class CBHG(nn.Module):
    """adapted from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/tacotron2/cbhg.py
    """
    def __init__(self, hparams):
        super().__init__()
        self.postnet_k = hparams.postnet_k
        self.in_channel = hparams.encoder_embedding_dim
        self.postnet_num_highways = hparams.postnet_num_highways
        self.post_projections = hparams.post_projections
        self.split_tone = hparams.split_tone
        if self.split_tone:
            self.embedding_tone = nn.Embedding(6, hparams.tone_embedding)
            self.embedding_phone = nn.Embedding(hparams.num_chars,
                                                hparams.encoder_embedding_dim - hparams.tone_embedding)
        else:
            self.embedding_phone = nn.Embedding(hparams.num_chars, hparams.encoder_embedding_dim)

        self.bank_kernels = [i for i in range(1, self.postnet_k + 1)]
        self.conv_bank = nn.ModuleList()
        for k in self.bank_kernels:
            if k % 2 != 0:
                padding = (k - 1) // 2
            else:
                padding = ((k - 1) // 2, (k - 1) // 2 + 1)
            self.conv_bank += [
                torch.nn.Sequential(
                    torch.nn.ConstantPad1d(padding, 0.0),
                    ConvNorm(self.in_channel, self.in_channel, kernel_size=k, stride=1, padding=0, dilation=1, w_init_gain='relu'),
                    nn.BatchNorm1d(self.in_channel),
                    nn.ELU(),
                )
            ]

        self.maxpool = torch.nn.Sequential(
            torch.nn.ConstantPad1d((0, 1), 0.0),
            torch.nn.MaxPool1d(2, stride=1)
        )
        self.projections = torch.nn.Sequential(
            ConvNorm(len(self.bank_kernels) * self.in_channel, self.post_projections[0], kernel_size=3, stride=1, padding=1, dilation=1, w_init_gain='relu'),
            nn.BatchNorm1d(self.post_projections[0]),
            nn.ELU(),
            ConvNorm(self.post_projections[0], self.post_projections[1], kernel_size=3, stride=1, padding=1, dilation=1, w_init_gain='linear'),
            nn.BatchNorm1d(self.post_projections[1]),
        )

        self.highways = nn.ModuleList()
        for i in range(self.postnet_num_highways):
            hn = HighwayNetwork(self.post_projections[-1])
            self.highways.append(hn)

        self.lstm = nn.LSTM(self.post_projections[-1], self.post_projections[-1]//2, batch_first=True, bidirectional=True)

    def forward(self, x_phone, input_lengths):
        if self.split_tone:
            x_phone, x_tone = x_phone[:, 0, :], x_phone[:, 1, :]
            x = self.embedding_phone(x_phone)
            x_tone = self.embedding_tone(x_tone)
            x = torch.cat([x, x_tone], dim=-1)
        else:
            x = self.embedding_phone(x_phone)

        x = x.transpose(1, 2)
        residual = x
        convs = []
        for k in range(len(self.bank_kernels)):
            convs += [self.conv_bank[k](x)]
        convs = torch.cat(convs, dim=1)
        x = self.maxpool(convs)
        x = self.projections(x)

        x = x + residual
        x = x.transpose(1, 2)
        for h in self.highways:
            x = h(x)

        input_lengths = input_lengths.cpu().numpy()
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=total_length)

        return outputs

    def inference(self, x_phone):
        if self.split_tone:
            x_phone, x_tone = x_phone[:, 0, :], x_phone[:, 1, :]
            x = self.embedding_phone(x_phone)
            x_tone = self.embedding_tone(x_tone)
            x = torch.cat([x, x_tone], dim=-1)
        else:
            x = self.embedding_phone(x_phone)

        x = x.transpose(1, 2)
        residual = x
        convs = []
        for k in range(len(self.bank_kernels)):
            convs += [self.conv_bank[k](x)]
        convs = torch.cat(convs, dim=1)
        x = self.maxpool(convs)
        x = self.projections(x)

        x = x + residual
        x = x.transpose(1, 2)
        for h in self.highways:
            x = h(x)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.hparams = hparams
        self.split_tone = hparams.split_tone
        if self.split_tone:
            self.embedding_tone = nn.Embedding(6, hparams.tone_embedding)
            self.embedding_phone = nn.Embedding(hparams.num_chars,
                                                hparams.encoder_embedding_dim - hparams.tone_embedding)
        else:
            self.embedding_phone = nn.Embedding(hparams.num_chars, hparams.encoder_embedding_dim)

        self.convolutions = nn.ModuleList()
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            self.convolutions.append(conv_layer)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim, hparams.encoder_embedding_dim//2, 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x_phone, input_lengths):
        """
        :param x_phone: phones index
        :param input_lengths: unpadded input lengths
        :return: encoder_outputs: encoder outputs
        """
        if self.split_tone:
            x_phone, x_tone = x_phone[:, 0, :], x_phone[:, 1, :]
            x = self.embedding_phone(x_phone)
            x_tone = self.embedding_tone(x_tone)
            x = torch.cat([x, x_tone], dim=-1)
        else:
            x = self.embedding_phone(x_phone)

        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.dropout(F.elu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu().numpy()
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=total_length)

        return outputs

    def inference(self, x_phone):
        """
        :param x_phone: phones index
        :return: encoder_outputs: encoder outputs
        """
        if self.split_tone:
            x_phone, x_tone = x_phone[:, 0, :], x_phone[:, 1, :]
            x = self.embedding_phone(x_phone)
            x_tone = self.embedding_tone(x_tone)
            x = torch.cat([x, x_tone], dim=-1)
        else:
            x = self.embedding_phone(x_phone)

        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.dropout(F.elu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.attention_dim = hparams.attention_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dims = hparams.prenet_dims
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.attention_mode = hparams.attention_mode
        self.feed_back_last = hparams.feed_back_last

        self.prenet = Prenet(
            self.n_mel_channels if self.feed_back_last else self.n_mel_channels * self.n_frames_per_step,
            self.prenet_dims)

        self.attention_rnn = nn.LSTMCell(self.prenet_dims[-1] + self.encoder_embedding_dim, self.attention_rnn_dim)

        if self.attention_mode == 'GMM':
            self.kernel = hparams.gmm_kernel
            self.attention_layer = GMMAttention(
                hparams.attention_rnn_dim, hparams.attention_dim,
                hparams.gmm_kernel, hparams.delta_bias, hparams.sigma_bias)
        else:
            self.attention_layer = ForwardAttentionV2(
                hparams.attention_rnn_dim, self.encoder_embedding_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(self.attention_rnn_dim + self.encoder_embedding_dim, self.decoder_rnn_dim)

        self.linear_projection = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            self.n_mel_channels * self.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.n_mel_channels * self.n_frames_per_step,
            self.n_frames_per_step, bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = memory.new_zeros(B, self.n_mel_channels if self.feed_back_last else self.n_mel_channels*self.n_frames_per_step)
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = memory.new_zeros(B, self.attention_rnn_dim)
        self.attention_cell = memory.new_zeros(B, self.attention_rnn_dim)

        self.decoder_hidden = memory.new_zeros(B, self.decoder_rnn_dim)
        self.decoder_cell = memory.new_zeros(B, self.decoder_rnn_dim)

        self.attention_context = memory.new_zeros(B, self.encoder_embedding_dim)
        self.attention_weights = memory.new_zeros(B, MAX_TIME)
        if self.attention_mode == "GMM":
            self.mu = memory.new_zeros(B, self.kernel, 1)  # [B, K, 1]
            self.t = torch.arange(MAX_TIME, device=memory.device)
            self.t = self.t.expand(B, self.kernel, MAX_TIME).float()
        else:
            self.attention_weights_cum = memory.new_zeros(B, MAX_TIME)
            self.log_alpha = memory.new_zeros(B, MAX_TIME).fill_(-float(1e4))
            self.log_alpha[:, 0].fill_(0.)
            self.processed_memory = self.attention_layer.memory_layer(memory)

        self.memory = memory
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.contiguous().view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)

        if self.feed_back_last:
            decoder_inputs = decoder_inputs[:, :, -self.n_mel_channels:]

        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()
        gate_outputs = gate_outputs.view(gate_outputs.size(0), -1)
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)

        if self.attention_mode == "GMM":
            self.attention_context, self.attention_weights, self.mu = self.attention_layer(
                self.attention_hidden, self.memory, self.mu, self.t, self.mask)
        else:
            attention_weights_cat = torch.cat(
                [self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)], dim=1)
            self.attention_context, self.attention_weights, self.log_alpha = self.attention_layer(
                self.attention_hidden, self.memory, self.processed_memory,
                attention_weights_cat, self.mask, self.log_alpha)
            self.attention_weights_cum += self.attention_weights

        decoder_input = torch.cat([self.attention_hidden, self.attention_context], dim=1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)

        decoder_output = self.linear_projection(decoder_hidden_attention_context)
        stop_input = torch.cat([self.decoder_hidden, decoder_output], dim=1)
        gate_prediction = self.gate_layer(F.dropout(stop_input, 0.1, self.training))

        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs [B, n_mels, T_out]
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(memory, mask=~get_mask_from_lengths(memory_lengths, max_len=memory.size(1)))

        mel_outputs, gate_outputs, alignments = [], [], []

        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs += [mel_output]
            gate_outputs += [gate_output]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [alignment]

            if (torch.sigmoid(gate_output.data) > self.gate_threshold).any():
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output[:, -self.n_mel_channels:] if self.feed_back_last else mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.hparams = hparams
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder = Encoder(hparams)
        # self.encoder = CBHG(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = ConvPostnet(hparams)

    def forward(self, phones, mels, text_lengths, output_lengths):
        encoder_outputs = self.encoder(phones, text_lengths)
        s_prob, e_prob = None, None

        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, s_prob, e_prob]

    def inference(self, phones, speaker_id):
        encoder_outputs = self.encoder.inference(phones)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs.squeeze(0), mel_outputs_postnet.squeeze(0)
