import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import AdvancedPatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        return x.transpose(*self.dims).contiguous() if self.contiguous else x.transpose(*self.dims)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        self.patch_embedding = AdvancedPatchEmbedding(
            d_model=configs.d_model,
            patch_len=patch_len,
            stride=stride,
            padding=padding,
            dropout=configs.dropout,
            hierarchical_levels=2,  # Can be adjusted
            merge_method='conv'      # Options: 'conv', 'linear'
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, 
                                    attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2)
            )
        )

        # Adjusted head_nf calculation for hierarchical patching
        reduced_patches = int((configs.seq_len - patch_len) / stride + 2)
        reduced_patches = reduced_patches // (2 ** (2-1))  # Account for 2 hierarchical levels
        self.head_nf = configs.d_model * reduced_patches
        
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, 
                                  head_dropout=configs.dropout)
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len, 
                                  head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def encode(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out)
        enc_out = enc_out.view(-1, n_vars, enc_out.shape[1], enc_out.shape[2])
        return enc_out.permute(0, 1, 3, 2)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = (x_enc - means) / torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)

        enc_out = self.encode(x_enc)
        dec_out = self.head(enc_out).permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1) + \
                 means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
        x_enc /= stdev

        enc_out = self.encode(x_enc)
        dec_out = self.head(enc_out).permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1) + \
                 means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = (x_enc - means) / stdev

        enc_out = self.encode(x_enc)
        dec_out = self.head(enc_out).permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1) + \
                 means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = (x_enc - means) / stdev

        enc_out = self.encode(x_enc)
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.view(output.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        elif self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
