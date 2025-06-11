# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeedForward, MultiHeadAttention
from recbole.model.loss import BPRLoss
import copy

class FrequencyLayer(nn.Module):
    def __init__(self, hidden_dropout_prob, hidden_size, c, beta):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.c = c // 2 + 1
        if beta > 0:
            self.beta = beta
        elif beta == 0:
            self.beta = 0
        elif beta == -1:
            self.beta = nn.Parameter(torch.randn(1, 1, 1))
        else:
            self.beta = nn.Parameter(torch.randn(1, 1, hidden_size))


    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        
        high_pass = input_tensor - low_pass
        
        sequence_emb_fft = low_pass + (self.beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class BSARecLayer(nn.Module):
    def __init__(self, 
                 n_heads, 
                 hidden_size, 
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 layer_norm_eps,
                 alpha,
                 c,
                 filter_type,
                 beta
                 ):
        super(BSARecLayer, self).__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.layer_norm_eps =layer_norm_eps
        self.filter_layer = FrequencyLayer(hidden_dropout_prob, hidden_size, c, beta)
        self.attention_layer = MultiHeadAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.alpha = alpha 
        self.filter_type = filter_type
        self.gating_linear = nn.Linear(in_features=hidden_size*2, out_features=1, bias=False)
        self.gating_sigmoid = nn.Sigmoid()

    def item_pred_gating(self, a, b):
        concat = torch.cat((a,b), dim=-1)
        output = self.gating_linear(concat)
        output = self.gating_sigmoid(output)
        return output
    
    def forward(self, input_tensor, attention_mask):
        if self.filter_type == 'only_fd':
            dsp = self.filter_layer(input_tensor)
            hidden_states = dsp
        elif self.filter_type == 'only_sa':
            gsp = self.attention_layer(input_tensor, attention_mask)
            hidden_states = gsp
        elif self.filter_type == 'BSARec':
            dsp = self.filter_layer(input_tensor)
            gsp = self.attention_layer(input_tensor, attention_mask)
            # gating = self.item_pred_gating(dsp, gsp)
            # hidden_states = gating * dsp + ( 1 - gating ) * gsp
            hidden_states = self.alpha * dsp + ( 1 - self.alpha ) * gsp

        return hidden_states
    
class BSARecBlock(nn.Module):
    def __init__(self, 
                 n_heads, 
                 hidden_size, 
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 layer_norm_eps,
                 alpha,
                 c,
                 hidden_act,
                 intermediate_size,
                 filter_type, 
                 beta):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(n_heads, hidden_size, 
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 layer_norm_eps,
                 alpha,
                 c,
                 filter_type,
                 beta)
        self.feed_forward = FeedForward(hidden_size, intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class BSARecEncoder(nn.Module):
    def __init__(self, 
                 n_layers,
                 n_heads, 
                 hidden_size, 
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 layer_norm_eps,
                 alpha,
                 c,
                 hidden_act,
                 intermediate_size,
                 filter_type,
                 beta):
        super(BSARecEncoder, self).__init__()
        block = BSARecBlock(
                 n_heads, 
                 hidden_size, 
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 layer_norm_eps,
                 alpha,
                 c,
                 hidden_act,
                 intermediate_size,
                 filter_type,
                 beta)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers
    
class BSARec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(BSARec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.c = config['C']
        self.alpha = config['alpha']
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.filter_type = config["filter_type"]
        self.beta = config["beta"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = BSARecEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            intermediate_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            c = self.c,
            alpha = self.alpha,
            filter_type = self.filter_type,
            beta = self.beta
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
