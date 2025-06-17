import torch
import torch.nn as nn
import numpy as np
from layers.Embed import PatchEmbedding, TruncateModule
from einops import rearrange
import torch.nn.functional as F

class TwoStageTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, space_tokenlayer):
        super(TwoStageTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = 0.05
        self.activation = 'gelu'
        self.num_layers = 4
        self.space_tokenlayer = space_tokenlayer

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_ff,
            dropout=self.dropout,
            activation=self.activation
        )
        
        self.time_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        space_encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=d_ff,
            dropout=self.dropout,
            activation=self.activation
        )

        self.space_encoder = nn.TransformerEncoder(
            space_encoder_layer,
            num_layers=self.num_layers
        )

    def forward(self, enc_in):
        B, D, T = enc_in.shape
        enc_out = self.time_encoder(enc_in)
        enc_out = rearrange(enc_out, 'B D T -> B T D', B=B)

        if D == 14:
            enc_out, _ = self.space_tokenlayer(enc_out)

        enc_out = self.space_encoder(enc_out)
        return enc_out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_scales = 3
        self.seq_len = 32
        self.pred_len = 16
        self.revin = 0
        self.embed_type = 0
        self.ch_ind = 1
        self.stride = 32
        self.patch_len = 32
        self.d_model = 512
        self.d_ff = 256
        self.dropout = 0.1
        self.activation = 'gelu'
        self.sp_d_model = 512
        self.sp_num = 14
        process_layer = nn.Identity()
        self.local_token_layer = PatchEmbedding(
            self.seq_len, self.d_model, self.patch_len, self.stride, self.dropout, process_layer,
            pos_embed_type='sincos',
            learnable=True,
            ch_ind=self.ch_ind
        )

        self.space_token_layer = PatchEmbedding(
            self.d_model, self.sp_d_model, self.sp_num, self.sp_num, self.dropout, process_layer,
            pos_embed_type='sincos',
            learnable=True,
            ch_ind=self.ch_ind
        )

        self.final_linear = nn.Linear(self.num_scales, 1)
        self.final_dropout = nn.Dropout(self.dropout)

        self.dropouts = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.patch_merging = nn.ModuleList()
        self.transformer_encoders = nn.ModuleList()

        for i in range(self.num_scales):
            current_d_model = int(self.d_model / (2 ** i))
            self.transformer_encoders.append(
                TwoStageTransformerEncoder(current_d_model, self.d_ff, self.space_token_layer)
            )
            self.dropouts.append(nn.Dropout(self.dropout))
            self.linears.append(nn.Linear(current_d_model * self.sp_d_model, 1))
            if i < self.num_scales - 1:
                self.patch_merging.append(nn.Linear(2 * self.sp_d_model, self.sp_d_model))

    def forward(self, x_enc):
        x_enc = x_enc.cuda()
        B, _, M = x_enc.shape
        x_enc = rearrange(x_enc, 'B T D -> B D T', B=B)
        enc_out, _ = self.local_token_layer(x_enc, None)

        B, _, M = enc_out.shape
        encoder_outputs = []
        
        for i, encoder in enumerate(self.transformer_encoders):
            enc_out = encoder(enc_out)
            encoder_outputs.append(self.dropouts[i](
                self.linears[i](rearrange(enc_out, 'B T D -> B (T D)', B=B))
            ))
            if i < len(self.patch_merging):
                enc_out = self.merge_patches(enc_out, i)

        dec_out = self.final_linear(rearrange(encoder_outputs, 'N B R -> B (N R)', N=self.num_scales))
        dec_out = self.final_dropout(dec_out)
        return dec_out

    def merge_patches(self, x, i):
        B, _, M = x.shape
        x = rearrange(x, 'b (n p) d -> b n (p d)', p=2)
        x = self.patch_merging[i](x)
        x = rearrange(x, 'B T D -> B D T', B=B)
        return x

    def loss_function(self, prediction, target):
        eps = 1e-6
        reg_loss = torch.sqrt(F.mse_loss(prediction.cuda(), target.cuda(), reduction='mean'))
        return reg_loss
