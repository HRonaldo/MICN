import torch
import torch.nn as nn
from models.embed import DataEmbedding,positional_encoding
from models.local_global import Seasonal_Prediction
from models.RevIN import RevIN
from models.Other import FourierLayer,series_decomp_multi
from models.ffc_1d import FFC_BN_ACT_1d,BasicBlock_1d

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class MICN(nn.Module):
    def __init__(self, dec_in, c_out, seq_len, label_len, out_len,
                 d_model=256, n_heads=8,d_layers=3,
                 dropout=0.2,embed='fixed', freq='h',
                 device=torch.device('cuda:0'), mode='regre',
                 decomp_kernel=[33], conv_kernel=[12, 24], isometric_kernel=[18, 6],patch_len=4,stride=2,padding_patch='end'
                 ,pe = 'zeros',learn_pe=True,individual=False,revin=True,affine=True,subtract_lasy=True,head_dropout=0,
                 ratio_gin=0.5,ratio_gout=0.5):
        super(MICN, self).__init__()

        self.pred_len = out_len
        self.seq_len = seq_len
        self.c_out = c_out
        self.decomp_kernel = decomp_kernel
        self.mode = mode

        self.decomp_multi = series_decomp_multi(kernel_size=[4, 8, 12])
        self.dropout = nn.Dropout(dropout)

        # norm(revin)
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(dec_in, affine=affine, subtract_last=subtract_lasy)

        # patch and embedding
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((seq_len - patch_len)/stride + 1)
        pred_patch_num = int((self.pred_len - patch_len) / stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)

        # embedding
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        self.encoder = Seasonal_Prediction(embedding_size=d_model, n_heads=n_heads, dropout=dropout,
                                     d_layers=d_layers, decomp_kernel=decomp_kernel, c_out=c_out, conv_kernel=conv_kernel,
                                     isometric_kernel=isometric_kernel, device=device,patch_num=patch_num,pred_patch_num=pred_patch_num,patch_len=patch_len)

        self.regression = nn.Linear(seq_len, out_len)
        self.regression.weight = nn.Parameter((1/out_len) * torch.ones([out_len, seq_len]), requires_grad=True)

        self.individual = individual
        self.n_vars = dec_in
        self.head_nf = d_model * patch_num
        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, self.pred_len, head_dropout=head_dropout)
        self.projection = nn.Linear(d_model, c_out)

        self.seasonality_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        self.conv1 = FFC_BN_ACT_1d(d_model, d_model, kernel_size=1, ratio_gin=ratio_gin, ratio_gout=ratio_gout)
        self.conv2 = FFC_BN_ACT_1d(d_model,d_model,kernel_size=3,padding=1,ratio_gin=ratio_gin,ratio_gout=ratio_gout)
        self.conv3 = FFC_BN_ACT_1d(d_model,d_model,kernel_size=1,ratio_gin=ratio_gin,ratio_gout=ratio_gout)

        in_cg = int(d_model * ratio_gin)
        in_cl = d_model - in_cg

        self.split_channels = [in_cg,in_cl]

        block = []
        # for i in range(d_layers):
        #     block.append(FFC_BN_ACT_1d(d_model, d_model, kernel_size=1, ratio_gin=ratio_gin, ratio_gout=ratio_gout))
        for i in range(d_layers):
            block.append(BasicBlock_1d(inplanes=d_model))
        self.ffc_conv = nn.Sequential(*block)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        n_vars = x.shape[2]
        # # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')

        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        # trend-cyclical prediction block: regre or mean
        # if self.mode == 'regre':
        #     seasonal_init_enc, trend = self.decomp_multi(x_enc)
        #     trend = self.regression(trend.permute(0,2,1)).permute(0, 2, 1)
        # elif self.mode == 'mean':
        #     mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        #     seasonal_init_enc, trend = self.decomp_multi(x_enc)
        #     trend = torch.cat([trend[:, -self.seq_len:, :], mean], dim=1)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)



        # [bs × seq_len × n_vars]
        x = seasonality

        x = x.permute(0,2,1)
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x = x.premute(0,1,3,2)
        x = self.W_P(x)

        # CI
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u + self.W_pos)
        # dec_out = u


        u = u.permute(0,2,1)
        u = torch.split(u, split_size_or_sections=self.split_channels, dim=1)
        u = self.ffc_conv(u)
        out = torch.cat(u,dim=1).permute(0,2,1)

        # out = self.encoder(u)


        z = out
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        z = z.permute(0, 1, 3, 2)
        z = self.head(z)
        z = z.permute(0,2,1)


        z = z[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]

        if self.revin:
            z = self.revin_layer(z, 'denorm')
        return z


        # embedding
        # seasonal_init_dec [ bs × (seq_l + pre_l) × m ]
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        # seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.seq_len:, :], zeros], dim=1)
        # dec_out = self.dec_embedding(seasonal_init_dec, x_mark_dec)
        # dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
        #
        # return dec_out

