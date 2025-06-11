import torch
import torch.nn as nn
import copy
from recbole.model.abstract_recommender import SequentialRecommender

class DistFilterLayer(nn.Module):
    def __init__(self, args):
        super(DistFilterLayer, self).__init__()
        self.mean_complex_weight = nn.Parameter(
            torch.randn(1, args.max_seq_length // 2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.cov_complex_weight = nn.Parameter(
            torch.randn(1, args.max_seq_length // 2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)

        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layer_norm = LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_mean_tensor, input_cov_tensor):
        batch, seq_len, hidden = input_mean_tensor.shape

        mean_x = torch.fft.rfft(input_mean_tensor, dim=1, norm='ortho')
        mean_weight = torch.view_as_complex(self.mean_complex_weight)
        mean_x = mean_x * mean_weight
        mean_sequence_emb_fft = torch.fft.irfft(mean_x, n=seq_len, dim=1, norm='ortho')
        mean_hidden_states = self.out_dropout(mean_sequence_emb_fft)
        mean_hidden_states = self.layer_norm(mean_hidden_states + input_mean_tensor)

        cov_x = torch.fft.rfft(input_cov_tensor, dim=1, norm='ortho')
        cov_weight = torch.view_as_complex(self.cov_complex_weight)
        cov_x = cov_x * cov_weight
        cov_sequence_emb_fft = torch.fft.irfft(cov_x, n=seq_len, dim=1, norm='ortho')
        cov_hidden_states = self.out_dropout(cov_sequence_emb_fft)
        cov_hidden_states = self.layer_norm(cov_hidden_states + input_cov_tensor)

        return mean_hidden_states, cov_hidden_states

class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.filter_layer = DistFilterLayer(args)
        self.activation_func = nn.ELU()

    def forward(self, mean_hidden_states, cov_hidden_states):
        mean_filter_output, cov_filter_output = self.filter_layer(mean_hidden_states, cov_hidden_states)
        return mean_filter_output, self.activation_func(cov_filter_output) + 1

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, mean_hidden_states, cov_hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            mean_hidden_states, cov_hidden_states = layer_module(mean_hidden_states, cov_hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append([mean_hidden_states, cov_hidden_states])
        if not output_all_encoded_layers:
            all_encoder_layers.append([mean_hidden_states, cov_hidden_states])
        return all_encoder_layers


class VanillaAttention(nn.Module):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights

class DLFSRecModel(SequentialRecommender):

    def __init__(self, config, dataset):
        super(DLFSRecModel, self).__init__(config, dataset)
        
        self.hidden_size = config['hidden_size']

        self.item_mean_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.item_cov_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        self.side_mean_dense = nn.Linear(args.feature_size, args.attribute_hidden_size)
        self.side_cov_dense = nn.Linear(args.feature_size, args.attribute_hidden_size)

        if args.fusion_type == 'concat':
            self.mean_fusion_layer = nn.Linear(args.attribute_hidden_size + args.hidden_size, args.hidden_size)
            self.cov_fusion_layer = nn.Linear(args.attribute_hidden_size + args.hidden_size, args.hidden_size)

        elif args.fusion_type == 'gate':
            self.mean_fusion_layer = VanillaAttention(args.hidden_size, args.hidden_size)
            self.cov_fusion_layer = VanillaAttention(args.hidden_size, args.hidden_size)

        self.mean_layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.cov_layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)
        self.elu = torch.nn.ELU()

        self.apply(self.init_weights)

    def forward(self, input_ids, input_context):

        mean_id_emb = self.item_mean_embeddings(input_ids)
        cov_id_emb = self.item_cov_embeddings(input_ids)

        input_attrs = self.args.items_feature[input_ids]
        mean_side_dense = self.side_mean_dense(torch.cat((input_context, input_attrs), dim=2))
        cov_side_dense = self.side_cov_dense(torch.cat((input_context, input_attrs), dim=2))

        if self.args.fusion_type == 'concat':
            mean_sequence_emb = self.mean_fusion_layer(torch.cat((mean_id_emb, mean_side_dense), dim=2))
            cov_sequence_emb = self.cov_fusion_layer(torch.cat((cov_id_emb, cov_side_dense), dim=2))
        elif self.args.fusion_type == 'gate':
            mean_concat = torch.cat(
                [mean_id_emb.unsqueeze(-2), mean_side_dense.unsqueeze(-2)], dim=-2)
            mean_sequence_emb, _ = self.mean_fusion_layer(mean_concat)
            cov_concat = torch.cat(
                [cov_id_emb.unsqueeze(-2), cov_side_dense.unsqueeze(-2)], dim=-2)
            cov_sequence_emb, _ = self.cov_fusion_layer(cov_concat)
        else:
            mean_sequence_emb = mean_id_emb + mean_side_dense
            cov_sequence_emb = cov_id_emb + cov_side_dense

        mask = (input_ids > 0).long().unsqueeze(-1).expand_as(mean_sequence_emb)
        mean_sequence_emb = mean_sequence_emb * mask
        cov_sequence_emb = cov_sequence_emb * mask

        mean_sequence_emb = self.dropout(self.mean_layer_norm(mean_sequence_emb))
        cov_sequence_emb = self.elu(self.dropout(self.cov_layer_norm(cov_sequence_emb))) + 1

        item_encoded_layers = self.item_encoder(mean_sequence_emb,
                                                cov_sequence_emb,
                                                output_all_encoded_layers=True)
        sequence_mean_output, sequence_cov_output = item_encoded_layers[-1]

        return sequence_mean_output, sequence_cov_output

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



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