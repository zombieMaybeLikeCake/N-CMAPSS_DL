import torch
import torch.nn as nn
import numpy as np
from layers.BiMamba4TS_layers import Encoder, EncoderLayer
from layers.Embed import PatchEmbedding, TruncateModule
from einops import rearrange
from mamba_plus import Mamba
import torch.nn.functional as F

class TwoStageBiMambaEncoder(nn.Module):
    def __init__(self,d_model,d_ff,space_tokenlayer):
        super(TwoStageBiMambaEncoder, self).__init__()
        self.d_model=d_model
        self.d_state=8
        self.d_conv=4
        self.e_fact=4
        self.d_ff=d_ff
        self.dropout=0.05
        self.activation='gelu'
        self.bi_dir=1
        self.residual=2
        self.e_layers=4
        self.sp_d_model=64
        self.space_tokenlayer=space_tokenlayer
        self.spaceencoder = Encoder(
            [
                EncoderLayer(
                    Mamba(d_model=self.sp_d_model, 
                          d_state=self.d_state, 
                          d_conv=self.d_conv, 
                          expand=self.e_fact,
                          use_fast_path=False),
                    Mamba(d_model=self.sp_d_model, 
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
    def forward(self,enc_in):
        B, D, T = enc_in.shape
        enc_out = self.timeencoder(enc_in)
        enc_out = rearrange(enc_out, 'B D T -> B T D', B=B)
        if D == 14:
            enc_out, _ =self.space_tokenlayer(enc_out)
        enc_out = self.spaceencoder(enc_out)
        # enc_out = rearrange(enc_in, 'B D T -> B T D', B=B)
        # enc_out = self.spaceencoder(enc_out)
        # enc_out = rearrange(enc_out, 'B D T -> B T D', B=B)
        # enc_out = self.timeencoder(enc_in)
        # enc_out = rearrange(enc_out, 'B T D -> B D T', B=B)
        return enc_out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.seq_len = configs.seq_len
        # self.pred_len = configs.pred_len
        # self.revin = configs.revin
        # self.embed_type = configs.embed_type

        self.num_scales = 4
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
        self.sp_num=14
        self.sp_d_model=64
        self.patch_num = int((self.seq_len - self.patch_len)/self.stride + 1)
        process_layer = nn.Identity()
        self.local_token_layer = PatchEmbedding(self.seq_len, self.d_model, self.patch_len, self.stride, self.dropout, process_layer,
                                                pos_embed_type='sincos',
                                                learnable=True,
                                                ch_ind=self.ch_ind)
        self.space_token_layer = PatchEmbedding(self.d_model, self.sp_d_model, self.sp_num, self.sp_num, self.dropout, process_layer,
                                                pos_embed_type='sincos',
                                                learnable=True,
                                                ch_ind=self.ch_ind)
        # self.final_linear=nn.Linear(self.d_model*self.sp_d_model,1)
        self.final_linear=nn.Linear(self.num_scales,1)
        self.final_dropout=nn.Dropout(self.dropout)
        self.dropouts = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.patch_merging = nn.ModuleList()
        self.BiMambaEncoders = nn.ModuleList()
        for i in range(self.num_scales):
            current_d_model = int(self.d_model / (2 ** i))
            # current_d_model = self.d_model
            self.BiMambaEncoders.append(TwoStageBiMambaEncoder(current_d_model,self.d_ff,self.space_token_layer))
            self.dropouts.append(nn.Dropout(self.dropout))
            self.linears.append(nn.Linear(current_d_model*self.sp_d_model,1))
            if i < self.num_scales - 1:
                # self.patch_merging.append(nn.Linear(2 * current_d_model, next_d_model))
                self.patch_merging.append(nn.Linear(2 * self.sp_d_model, self.sp_d_model))
    def forward(self, x_enc):
        x_enc=x_enc.cuda()
        # B T S
        B, _, M = x_enc.shape
        # [B, T, S] -> [B, T, D(128)]
        x_enc=rearrange(x_enc, 'B T D -> B D T', B=B)
        # print("x_out size",x_enc.size())
        # print("x_enc.permute(0, 2, 1) size",x_enc.permute(0, 2, 1).size())
        # print("x_enc size",x_enc.size())
        enc_out, _ = self.local_token_layer(x_enc,None)
        
        B, _, M = enc_out.shape
        encoder_outputs = []
        # for i, encoder in enumerate(zip(self.BiMambaEncoders)):\
        for i, encoder in enumerate(self.BiMambaEncoders):

            enc_out = encoder(enc_out)
            # encoder_outputs.append(self.linears[i](x.mean(dim=1)))
            encoder_outputs.append(self.dropouts[i](self.linears[i](rearrange(enc_out, 'B T D -> B (T D)', B=B))))
            # print("enc_out size",enc_out.size())
            if i < len(self.patch_merging): 
                enc_out = self.merge_patches(enc_out, i)
        # print("enc_out size",enc_out.size())
        # print("enc_out size",enc_out.size())
        # print("final_out size",enc_out.size())
        # output: [B*M, N, D] or [B*N, M, D] -> [B x M x H]
        dec_out=self.final_linear(rearrange(encoder_outputs, 'N B R -> B (N R)', N=self.num_scales))
        dec_out = self.final_dropout(dec_out)
        return dec_out
    def merge_patches(self, x, i):
        B, _, M = x.shape
        # print("x size",x.size())
        x = rearrange(x, 'b (n p) d -> b n (p d)', p=2)# 將相鄰的patch合併
        x = self.patch_merging[i](x)  # 進行線性變換
        # print("after liear x size",x.size())
        x = rearrange(x, 'B T D -> B D T', B=B)
        return x
    def loss_function(self,prediction, target):
        eps = 1e-6
        reg_loss = torch.sqrt(F.mse_loss(prediction.cuda(), target.cuda(), reduction='mean'))
        return reg_loss
