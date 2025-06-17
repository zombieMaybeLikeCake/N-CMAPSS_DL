import torch
import torch.nn as nn
import numpy as np
from layers.BiMamba4TS_layers import Encoder, EncoderLayer
from layers.Embed import PatchEmbedding, TruncateModule
from einops import rearrange
from mamba_plus import Mamba
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,corr=None):
        super(Model, self).__init__()
        # self.seq_len = configs.seq_len
        # self.pred_len = configs.pred_len
        # self.revin = configs.revin
        # self.embed_type = configs.embed_type
        self.seq_len = 32
        self.pred_len = 16
        self.revin = 0
        self.embed_type = 0
        self.ch_ind = 1
        self.stride=32
        self.patch_len=32
        self.d_model=256
        self.d_state=8
        self.d_conv=2
        self.e_fact=1
        self.d_ff=256
        self.dropout=0.1
        self.activation='gelu'
        self.bi_dir=1
        self.residual=1
        self.e_layers=2
        self.enc_in=14
        self.sp_d_model=14
        # if configs.SRA:
        #     self.ch_ind = self.SRA(corr, configs.threshold)
        # else:
        #     self.ch_ind = configs.ch_ind

        # patching
        # being able to divide indicates that there is no need to forcefully padding
        # if configs.seq_len % configs.stride==0:
        # print("self.seq_len % self.stride:",self.seq_len % self.stride)
        # if self.seq_len % self.stride==0:
            
        self.patch_num = int((self.seq_len - self.patch_len)/self.stride + 1)
        process_layer = nn.Identity()
        # else:
        #     # padding at tail
        #     if configs.padding_patch=="end":
        #         padding_length = configs.stride - (configs.seq_len % configs.stride)
        #         self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 2)
        #         process_layer = nn.ReplicationPad1d((0, padding_length))
        #     # if not padding, then execute cutting
        #     else:
        #         truncated_length = configs.seq_len - (configs.seq_len % configs.stride)
        #         self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 1)
        #         process_layer = TruncateModule(truncated_length)
        self.local_token_layer = PatchEmbedding(self.seq_len, self.d_model, self.patch_len, self.stride, self.dropout, process_layer,
                                                pos_embed_type='sincos',
                                                learnable=True,
                                                ch_ind=self.ch_ind)
        self.spaceencoder = Encoder(
            [
                EncoderLayer(
                    Mamba(d_model=self.sp_d_model, 
                        #   batch_size=configs.batch_size,
                          d_state=self.d_state, 
                          d_conv=self.d_conv, 
                          expand=self.e_fact,
                          use_fast_path=False),
                    Mamba(d_model=self.sp_d_model, 
                        #   batch_size=configs.batch_size,
                          d_state=self.d_state, 
                          d_conv=self.d_conv, 
                          expand=self.e_fact,
                          use_fast_path=False),
                    self.sp_d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                    bi_dir=self.bi_dir,
                    residual=self.residual==1
                ) for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.sp_d_model)
        )
        self.timeencoder=Encoder(
            [
                EncoderLayer(
                    Mamba(d_model=self.d_model, 
                        #   batch_size=configs.batch_size,
                          d_state=self.d_state, 
                          d_conv=self.d_conv, 
                          expand=self.e_fact,
                          use_fast_path=False),
                    Mamba(d_model=self.d_model, 
                        #   batch_size=configs.batch_size,
                          d_state=self.d_state, 
                          d_conv=self.d_conv, 
                          expand=self.e_fact,
                          use_fast_path=False),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                    bi_dir=self.bi_dir,
                    residual=self.residual==1
                ) for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        head_nf = self.d_model * self.patch_num
        self.head = Flatten_Head(False, self.enc_in, head_nf, self.pred_len)
        # self.final_linear=nn.Linear(self.d_model*self.sp_d_model,1)
        # self.final_linear=nn.Linear(self.d_model*self.sp_d_model,1)
        self.final_dropout=nn.Dropout(self.dropout)
    def SRA(self, corr, threshold):
        high_corr_matrix = corr >= threshold
        num_high_corr = np.maximum(high_corr_matrix.sum(axis=1)-1, 0)
        positive_corr_matrix = corr >= 0
        num_positive_corr = np.maximum(positive_corr_matrix.sum(axis=1)-1, 0)
        max_high_corr = num_high_corr.max()
        max_positive_corr = num_positive_corr.max()
        r = max_high_corr/max_positive_corr
        print('SRA -> channel mixing' if r >= 1 - threshold else 'channel independent')
        return 0 if r >= 1 - threshold else 1

    # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    def forward(self, x_enc):
        x_enc=x_enc.cuda()
        # B L M
        B, _, M = x_enc.shape
        # [B, T, D] -> [B, T, 128]
        x_enc=rearrange(x_enc, 'B T D -> B D T', B=B)
        # print("x_out size",x_enc.size())
        # print("x_enc.permute(0, 2, 1) size",x_enc.permute(0, 2, 1).size())
        # print("x_enc size",x_enc.size())
        enc_out, _ = self.local_token_layer(x_enc,None)
        # print("enc_out size",enc_out.size())
        enc_out = self.timeencoder(enc_out)
        # print("enc_out size",enc_out.size())

        enc_out=rearrange(enc_out, 'B D T -> B T D', B=B)
        # print("enc_out size",enc_out.size())
        enc_out = self.spaceencoder(enc_out)

        enc_out=rearrange(enc_out, 'B D T -> B (D T)', B=B)
        # print("final_out size",enc_out.size())
        # output: [B*M, N, D] or [B*N, M, D] -> [B x M x H]
        dec_out = self.final_linear(enc_out)
        dec_out = self.final_dropout(dec_out)
        return dec_out
    def loss_function(self,prediction, target):
        eps = 1e-6
        reg_loss = torch.sqrt(F.mse_loss(prediction.cuda(), target.cuda(), reduction='mean'))
        return reg_loss


class Flatten_Head(nn.Module):

    # self.enc_in=14
    # self.head = Flatten_Head(False, self.enc_in, head_nf, self.pred_len)
    # head_nf = self.d_model * self.patch_num
    # 
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0.):
        super().__init__()
        
        self.ch_ind = individual
        self.n_vars = n_vars
        
        if self.ch_ind:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
        self.final_linear=nn.Linear(target_window,1)
            
    def forward(self, x):                           # x: [B, C, D, N]
        if self.ch_ind:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])    # z: [B, D * N]
                z = self.linears[i](z)              # z: C x [B, H]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)           # x: [B, C, H]
            x = self.final_linear(x)
        else:
            x = self.flatten(x)                     # x: [B, C, D * N]
            x = self.linear(x)                      # x: [B, C, H]
            x = self.dropout(x)
            x = self.final_linear(x)
        return x
