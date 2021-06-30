# from https://github.com/NVIDIA/tacotron2
# Modified by Ajinkya Kulkarni


from torch import nn
import numpy as np
import torch
from torch.nn import functional as F
import hparams as hp
from model import get_mask_from_lengths

eps = 1e-8

class GuidedAttentionLoss(torch.nn.Module):
    """
    Adapted from https://github.com/mozilla/TTS/blob/master/TTS/tts/layers/losses.py
    """
    def __init__(self, sigma=0.4):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma

    def _make_ga_masks(self, ilens, olens):
        B = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        ga_masks = torch.zeros((B, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            ga_masks[idx, :olen, :ilen] = self._make_ga_mask(ilen, olen, self.sigma)
        return ga_masks

    def forward(self, att_ws, ilens, olens):
        ga_masks = self._make_ga_masks(ilens, olens).to(att_ws.device)
        seq_masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = ga_masks * att_ws
        loss = torch.mean(losses.masked_select(seq_masks))
        return loss

    @staticmethod
    def _make_ga_mask(ilen, olen, sigma):
        grid_x, grid_y = torch.meshgrid(torch.arange(olen).to(olen), torch.arange(ilen).to(ilen))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen)**2 / (2 * (sigma**2)))

    @staticmethod
    def _make_masks(ilens, olens):
        in_masks = get_mask_from_lengths(ilens)
        out_masks = get_mask_from_lengths(olens)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)

class Tacotron2Loss(nn.Module):
    def __init__(self, update_step):
        super(Tacotron2Loss, self).__init__()
        self.expressive_classes = hp.emotion_classes
        self.speaker_classes = hp.speaker_classes
        self.cat_lambda = hp.cat_lambda
        self.speaker_encoder_type = hp.speaker_encoder_type
        self.expressive_encoder_type = hp.expressive_encoder_type
        self.update_step = update_step
        self.kl_lambda = hp.kl_lambda
        self.kl_incr = hp.kl_incr
        self.kl_step = hp.kl_step
        self.kl_step_after = hp.kl_step_after
        self.kl_max_step = hp.kl_max_step

        self.cat_incr = hp.cat_incr
        self.cat_step = hp.cat_step
        self.cat_step_after = hp.cat_step_after
        self.cat_max_step = hp.cat_max_step

        self.n_frames_per_step = hp.n_frames_per_step
        self.attention_mode = hp.attention_mode
        self.guided_sigma = hp.guided_sigma
        self.guided_loss = GuidedAttentionLoss(self.guided_sigma)
        self.mel_type = hp.mel_type
        self.L1_Criterion = nn.L1Loss(reduction="mean")
        self.MSE_Criterion = nn.MSELoss(reduction="mean")
        self.BCEWithLogits_Criterion = nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=torch.tensor(hp.pos_weight) if hp.pos_weight > 0 else None)

    def indices_to_one_hot(self, data, n_classes):
        targets = data.contiguous().view(-1)
        return torch.eye(n_classes, device=targets.device)[targets]

    def KL_loss(self, mu, var):
        return torch.mean(0.5 * torch.sum(torch.exp(var) + mu ** 2 - 1. - var, 1))

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()

    def get_encoder_loss(self, id_, prob_, classes_, cat_lambda, kl_lambda, encoder_type):
        cat_target = self.indices_to_one_hot(id_, classes_)

        if (encoder_type == 'gst' or encoder_type == 'x-vector') and cat_lambda != 0.0:
            loss = cat_lambda * (-self.entropy(cat_target, prob_) - np.log(0.1))
        elif (encoder_type == 'vae' or encoder_type == 'gst_vae') and (cat_lambda != 0.0 or kl_lambda != 0.0):
            loss = cat_lambda * (-self.entropy(cat_target, prob_[2]) - np.log(0.1)) + \
                   kl_lambda * self.KL_loss(prob_[0], prob_[1])
        elif encoder_type == 'gmvae' and (cat_lambda != 0.0 or kl_lambda != 0.0):
            loss = self.gaussian_loss(prob_[0], prob_[1], prob_[2], prob_[3], prob_[4])*kl_lambda + (-self.entropy(cat_target, prob_[5]) - np.log(0.1))*cat_lambda
        else:
            loss = 0.0

        return loss

    def update_lambda(self, iteration):
        iteration += 1
        if iteration % self.update_step == 0:
            self.kl_lambda = self.kl_lambda + self.kl_incr
            self.cat_lambda = self.cat_lambda + self.cat_incr

        if iteration <= self.kl_max_step and iteration % self.kl_step == 0:
            kl_lambda = self.kl_lambda
        elif iteration > self.kl_max_step and iteration % self.kl_step_after == 0:
            kl_lambda = self.kl_lambda
        else:
            kl_lambda = 0.0

        if iteration <= self.cat_max_step and iteration % self.cat_step == 0:
            cat_lambda = self.cat_lambda
        elif iteration > self.cat_max_step and iteration % self.cat_step_after == 0:
            cat_lambda = self.cat_lambda
        else:
            cat_lambda = 0.0

        return min(1.0, kl_lambda), min(1.0, cat_lambda)

    def log_normal(self, x, mu, var):
        if eps > 0.0:
            var = var + eps
        return -0.5 * torch.sum(np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))

    def forward(self, iteration, model_output, targets, s_id, e_id):

        kl_lambda, cat_lambda = self.update_lambda(iteration)

        mel_target, gate_target, input_lengths, mel_length = targets[0], targets[1], targets[2], targets[3]
        mel_out, mel_out_postnet, gate_out, alignments, s_prob, e_prob = model_output

        # tacotron losses
        l1_loss, mse_loss = 0.0, 0.0
        gate_loss, align_loss = 0.0, 0.0
        speaker_loss, expressive_loss = 0.0, 0.0

        # prepare masks
        mel_masks = get_mask_from_lengths(mel_length, max_len=mel_out.size(2))
        mel_masks = mel_masks.unsqueeze(1).expand(-1, mel_out.size(1), -1)

        # gate_target = gate_target.contiguous().view(gate_target.size(0), -1, self.n_frames_per_step)[..., 0]
        # mel_step_lengths = torch.ceil(mel_length.float() / self.n_frames_per_step).long()
        # stop_masks = get_mask_from_lengths(mel_step_lengths, max_len=mel_out.size(2) // self.n_frames_per_step)
        stop_masks = get_mask_from_lengths(mel_length, max_len=mel_out.size(2))

        # using masked_select losses
        mel_out = mel_out.masked_select(mel_masks)
        mel_out_postnet = mel_out_postnet.masked_select(mel_masks)
        gate_out = gate_out.masked_select(stop_masks)

        mel_target = mel_target.masked_select(mel_masks)
        gate_target = gate_target.masked_select(stop_masks)
        
        # Masked losses
        # l1_loss = self.L1_Criterion(mel_out, mel_target) + self.L1_Criterion(mel_out_postnet, mel_target)
        mse_loss = self.MSE_Criterion(mel_out, mel_target) + self.MSE_Criterion(mel_out_postnet, mel_target)
        gate_loss = self.BCEWithLogits_Criterion(gate_out, gate_target)

        if self.guided_sigma > 0 and self.attention_mode[:3] != "GMM":
            align_loss = self.guided_loss(
                alignments, input_lengths, torch.ceil(mel_length.float() / self.n_frames_per_step).long())

        # speaker_encoder_loss
        if s_prob is not None:
            speaker_loss = self.get_encoder_loss(
                s_id, s_prob, self.speaker_classes, cat_lambda, kl_lambda, self.speaker_encoder_type)

        # expressive_encoder_loss
        if e_prob is not None:
            expressive_loss = self.get_encoder_loss(
                e_id, e_prob, self.expressive_classes, cat_lambda, kl_lambda, self.expressive_encoder_type)

        return l1_loss + mse_loss + gate_loss + align_loss + speaker_loss + expressive_loss