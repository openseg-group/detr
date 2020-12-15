# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import os
import pdb


class CrossAttentionLayer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 factor=2):
        super(CrossAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.latent_channels = in_channels // factor

        self.f_k, self.f_q, self.f_d, self.f_u = self._make_self_attention(in_channels, in_channels//factor)
        self.g_k, self.g_q, self.g_d, self.g_u = self._make_self_attention(in_channels, in_channels//factor)
        self.h_k, self.h_q, self.h_d, self.h_u = self._make_self_attention(in_channels, in_channels//factor)
        self.t_k, self.t_q, self.t_d, self.t_u = self._make_self_attention(in_channels, in_channels//factor)

    def _make_self_attention(self, in_channels, latent_channels):
        key_transform = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels,
                kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(latent_channels),
            nn.ReLU(inplace=True)
        )
        query_transform = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels,
                kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(latent_channels),
            nn.ReLU(inplace=True)
        )
        down_transform = nn.Conv2d(in_channels, latent_channels,
            kernel_size=1, stride=1, padding=0)
        up_transform = nn.Conv2d(latent_channels, in_channels,
            kernel_size=1, stride=1, padding=0)

        return key_transform, query_transform, down_transform, up_transform

    def forward(self, x_f, x_g, x_h, x_t):

        batch_size, channel, h, w = x_f.size()

        v_f = self.f_d(x_f).view(batch_size, h, w, self.latent_channels)
        v_g = self.g_d(x_g).view(batch_size, h, w, self.latent_channels)
        v_h = self.h_d(x_h).view(batch_size, h, w, self.latent_channels)
        v_t = self.t_d(x_t).view(batch_size, h, w, self.latent_channels)

        v = torch.cat((v_f[:,:,:,:,None], v_g[:,:,:,:,None], v_h[:,:,:,:,None], v_t[:,:,:,:,None]), 4)
        v = v.permute(0, 1, 2, 4, 3)

        q_f = self.f_q(x_f).view(batch_size, h, w, self.latent_channels)
        q_g = self.g_q(x_g).view(batch_size, h, w, self.latent_channels)
        q_h = self.h_q(x_h).view(batch_size, h, w, self.latent_channels)
        q_t = self.t_q(x_t).view(batch_size, h, w, self.latent_channels)
        q = torch.cat((q_f[:,:,:,:,None], q_g[:,:,:,:,None], q_h[:,:,:,:,None], q_t[:,:,:,:,None]), 4)
        q = q.permute(0, 1, 2, 4, 3)

        k_f = self.f_k(x_f).view(batch_size, h, w, self.latent_channels)
        k_g = self.g_k(x_g).view(batch_size, h, w, self.latent_channels)
        k_h = self.h_k(x_h).view(batch_size, h, w, self.latent_channels)
        k_t = self.t_k(x_t).view(batch_size, h, w, self.latent_channels)
        k = torch.cat((k_f[:,:,:,:,None], k_g[:,:,:,:,None], k_h[:,:,:,:,None], k_t[:,:,:,:,None]), 4)

        sim_map = torch.matmul(q, k)
        sim_map = (self.latent_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, v)
        context = context.permute(0, 1, 2, 4, 3).contiguous()

        context_f = context[:,:,:,:,0]
        context_g = context[:,:,:,:,1]
        context_h = context[:,:,:,:,2]
        context_t = context[:,:,:,:,3]

        out_f = x_f + self.f_u(context_f.permute(0, 3, 1, 2))
        out_g = x_g + self.g_u(context_g.permute(0, 3, 1, 2))
        out_h = x_h + self.h_u(context_h.permute(0, 3, 1, 2))
        out_t = x_t + self.t_u(context_t.permute(0, 3, 1, 2))

        return out_f, out_g, out_h, out_t


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

        self.stride_4x_proj = nn.Sequential(
            nn.Conv2d(18, 256, kernel_size=1),
            nn.SyncBatchNorm(256),
            nn.ReLU(inplace=True)
        )
        self.stride_8x_proj = nn.Sequential(
            nn.Conv2d(36, 256, kernel_size=1),
            nn.SyncBatchNorm(256),
            nn.ReLU(inplace=True)
        )
        self.stride_16x_proj = nn.Sequential(
            nn.Conv2d(72, 256, kernel_size=1),
            nn.SyncBatchNorm(256),
            nn.ReLU(inplace=True)
        )
        self.stride_32x_proj = nn.Sequential(
            nn.Conv2d(144, 256, kernel_size=1),
            nn.SyncBatchNorm(256),
            nn.ReLU(inplace=True)
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, src_list, mask_list, pos_embed_list):

        # encoder
        _, _, h_4x, w_4x = src_list[0].size()
        feat1 = self.stride_4x_proj(src_list[0])
        feat2 = F.interpolate(self.stride_8x_proj(src_list[1]), size=(h_4x, w_4x), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(self.stride_16x_proj(src_list[2]), size=(h_4x, w_4x), mode="bilinear", align_corners=True)

        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)

        memory_4x = self.encoder(src, feat_list=[feat1, feat2, feat3], src_key_padding_mask=mask, pos=pos_embed)
        mask_4x = mask_list[0]
        mask_4x = mask_4x.flatten(1)
        pos_embed_4x = pos_embed_list[0].flatten(2).permute(2, 0, 1)

        # decoder
        hs = self.decoder(tgt, memory_4x, memory_key_padding_mask=mask_4x,
                        pos=pos_embed_4x, query_pos=query_embed)
        return hs.transpose(1, 2), memory_4x.permute(1, 2, 0).view(bs, c, h_4x, w_4x)



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.fuse_output_proj = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.SyncBatchNorm(256),
            nn.ReLU(inplace=True)
        )

        self.cross_atten = CrossAttentionLayer(256)

    def forward(self, src, feat_list,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None
                ):
        bs, c, h, w = src.shape
        _, _, h_4x, w_4x = feat_list[0].shape

        output = src
        output_4x = feat_list[0]
        output_8x = feat_list[1]
        output_16x = feat_list[2]

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            output_32x = F.interpolate(output.permute(1, 2, 0).view(bs, c, h, w), size=(h_4x, w_4x), mode="bilinear", align_corners=True)
            output_4x, output_8x, output_16x, output_32x = self.cross_atten(output_4x, output_8x, output_16x, output_32x)
            output = F.interpolate(output_32x, size=(h, w), mode="bilinear", align_corners=True)
            output = output.flatten(2).permute(2, 0, 1)

        output_32x = F.interpolate(output.permute(1, 2, 0).view(bs, c, h, w), size=(h_4x, w_4x), mode="bilinear", align_corners=True)
        output = torch.cat([output_4x, output_8x, output_16x, output_32x], 1)
        output = self.fuse_output_proj(output)
        output = output.flatten(2).permute(2, 0, 1)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
