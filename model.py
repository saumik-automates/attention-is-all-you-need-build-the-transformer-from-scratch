"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    token_to_id = {}
    
    for i, token in enumerate(specials):
        token_to_id[token] = i
    
    idx = len(specials)
    for token in " ".join(sentences).split():
        if token in token_to_id:
            continue
        token_to_id[token] = idx
        idx += 1

    return token_to_id

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    id_to_token = {}
    for token, id_ in token_to_id.items():
        id_to_token[id_] = token

    return id_to_token

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    def convert_token_to_id(token):
        if token in token_to_id:
            return token_to_id[token]
        else:
            return token_to_id[unk_token]

    return [convert_token_to_id(token) for token in sentence.split()]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    return [id_to_token[id_] for id_ in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    l = len(ids)
    if l >= max_len:
        return ids[:max_len]
    else:
        return ids + [pad_id]*(max_len-l)

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    return embeddings*math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    out = torch.zeros((d_model//2,))
    for i in range(0, d_model//2):
        out[i] = 10000**(-2*i/d_model)
    return out

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    return torch.arange(0, max_len, dtype=torch.float32).reshape(-1, 1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    
    sin_values = torch.sin(position*div_term)
    out = pe.clone()
    out[..., 0::2] = sin_values
    return out

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    
    cos_values = torch.cos(position*div_term)
    out = pe.clone()
    out[..., 1::2] = cos_values
    return out

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    
    div_term = compute_positional_div_term(d_model)
    positions = build_position_index_column(max_len)

    pe = torch.zeros((max_len, d_model))

    pe = fill_even_indices_with_sin(pe, positions, div_term)
    pe = fill_odd_indices_with_cos(pe, positions, div_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    L = embedded_batch.shape[1]
    return embedded_batch + positional_encoding[:L, :]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    attn_mask = (token_ids != pad_id)
    B, L = token_ids.shape[0], token_ids.shape[1]
    return attn_mask.reshape((B, 1, 1, L))

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    return torch.tril(torch.ones((seq_len, seq_len))).bool().reshape((1, 1, seq_len, seq_len))

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    return query @ torch.transpose(key, -2, -1)

# Step 18 - scale_attention_scores
import math

def scale_attention_scores(scores, d_k):
    return scores/math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # Only insert singleton dimensions if scores are 4D (Multi-head attention)
    # and the mask is 2D (batch_size, src_len)
    if mask is not None and mask.dim() == 2 and scores.dim() == 4:
        mask = mask.unsqueeze(1).unsqueeze(2)
    return scores.masked_fill(~mask, float('-inf'))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    exp_masked_scores = torch.exp(masked_scores)
    sum_last_dim_exp_masked_scores = torch.sum(exp_masked_scores, dim=-1, keepdim=True)
    return exp_masked_scores/torch.clamp(sum_last_dim_exp_masked_scores, min=1e-10)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    
    d_k = key.shape[-1]
    scaled_attn_scores = scale_attention_scores(compute_raw_attention_scores(query, key), d_k)
    masked_attn_scores = mask_attention_scores_with_neg_inf(scaled_attn_scores, mask) if mask is not None else scaled_attn_scores
    attn_weights = softmax_attention_weights(masked_attn_scores)
    return apply_attention_weights_to_values(attn_weights, value), attn_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    B, L, d_model = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    d_k= d_model//num_heads
    return tensor.reshape((B, L, num_heads, d_k))

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    return split_tensor.transpose(1, 2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    B, H, L, d_k = multi_head_tensor.shape[0], multi_head_tensor.shape[1], multi_head_tensor.shape[2], multi_head_tensor.shape[3]
    return multi_head_tensor.transpose(1, 2).reshape(B, L, H*d_k)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    if bias is None:
        return x @ weight.T
    else:
        return x @ weight.T + bias

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    return apply_linear_projection(x, w_q, b_q), apply_linear_projection(x, w_k, b_k), apply_linear_projection(x, w_v, b_v)

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    q_h = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    k_h = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))
    v_h = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))
    return q_h, k_h, v_h

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    return apply_linear_projection(merge_heads_back_to_model_dim(context), w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    Q = apply_linear_projection(query, w_q, None)
    K = apply_linear_projection(key, w_k, None)
    V = apply_linear_projection(value, w_v, None)
    Qh, Kh, Vh = split_qkv_into_heads(Q, K, V, num_heads)
    context, _ = multi_head_scaled_dot_product_attention(Qh, Kh, Vh, mask)
    return merge_heads_and_project_output(context, w_o, None)

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    return torch.maximum(torch.tensor(0), x @ w1 + b1)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    return apply_ffn_second_linear(apply_ffn_first_linear_and_relu(x, w1, b1), w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    return torch.mean(x, axis=-1, keepdim=True), torch.var(x, axis=-1, keepdim=True, correction=0)

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mu, var = compute_layer_norm_mean_and_variance(x)
    x_cap = (x - mu)/torch.sqrt(var+eps)
    return gamma*x_cap + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    return normalize_and_scale_with_gamma_beta(residual_input + sublayer_output, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    return keep_mask*(x/keep_prob)

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    x_mha = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(x, x_mha, gamma, beta)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    x_ffn = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    return apply_residual_add_and_norm(x, x_ffn, gamma, beta)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    x_sa = encoder_layer_self_attention_sublayer(x, layer_params["w_q"], layer_params["w_k"], \
                                                    layer_params["w_v"], layer_params["w_o"], \
                                                    layer_params["attn_gamma"], layer_params["attn_beta"], \
                                                    num_heads, src_mask)
    x_ff = encoder_layer_feed_forward_sublayer(x_sa, layer_params["w1"], layer_params["b1"], \
                                                     layer_params["w2"], layer_params["b2"], \
                                                     layer_params["ffn_gamma"], layer_params["ffn_beta"])
    return x_ff

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    for encoder_layer_params in encoder_layer_params_list:
        x = assemble_encoder_layer(x, encoder_layer_params, num_heads, src_mask)
    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    y_mha = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)
    return apply_residual_add_and_norm(y, y_mha, gamma, beta)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    y_ca = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(y, y_ca, gamma, beta)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    y_ffn = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    return apply_residual_add_and_norm(y, y_ffn, gamma, beta)

# Step 46 - assemble_decoder_layer
import torch
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    y_sa = decoder_layer_masked_self_attention_sublayer(y, layer_params["w_q_self"], layer_params["w_k_self"], \
                                                    layer_params["w_v_self"], layer_params["w_o_self"], \
                                                    layer_params["self_gamma"], layer_params["self_beta"], \
                                                    num_heads, tgt_mask)
    y_ca = decoder_layer_cross_attention_sublayer(y_sa, encoder_output, layer_params["w_q_cross"], layer_params["w_k_cross"], \
                                                    layer_params["w_v_cross"], layer_params["w_o_cross"], \
                                                    layer_params["cross_gamma"], layer_params["cross_beta"], \
                                                     num_heads, src_mask)
    return decoder_layer_feed_forward_sublayer(y_ca, layer_params["w1"], layer_params["b1"], \
                                                     layer_params["w2"], layer_params["b2"], \
                                                     layer_params["ffn_gamma"], layer_params["ffn_beta"])

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    for decoder_layer_param in decoder_layer_params_list:
        y = assemble_decoder_layer(y, encoder_output, decoder_layer_param, num_heads, src_mask, tgt_mask)
    return y

# Step 48 - apply_final_output_projection (not yet solved)
# TODO: implement

# Step 49 - tie_output_projection_to_token_embeddings (not yet solved)
# TODO: implement

# Step 50 - apply_log_softmax_over_vocab (not yet solved)
# TODO: implement

# Step 51 - run_transformer_forward (not yet solved)
# TODO: implement

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

