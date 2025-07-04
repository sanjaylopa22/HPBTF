import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
import torch.nn.functional as F
from math import sqrt
import pennylane as qml


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FuzzyAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, delta=0.5):
        super(FuzzyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.delta = delta  # Fuzziness factor

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        Forward pass for the Fuzzy Attention.

        :param queries: Query tensor of shape (B, L, H, E)
        :param keys: Key tensor of shape (B, S, H, D)
        :param values: Value tensor of shape (B, S, H, D)
        :param attn_mask: Mask to apply attention to certain locations
        :param tau: Temperature scaling factor (optional)
        :param delta: Fuzziness adjustment factor (optional, defaults to self.delta)
        :return: The computed attention output and optionally the attention weights
        """
        # Default delta value if not provided
        delta = delta or self.delta
        
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        # Compute attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Add gaussian fuzziness to the attention scores
        # fuzzy_scores = scores + delta * torch.randn_like(scores) 
        
        #Perturb scores using a scaled sigmoid function
        fuzzy_scores = scores + delta * torch.sigmoid(scores) 
        
        # Scale the noise 
        # fuzzy_scores = scores + delta * torch.randn_like(scores) * scores.abs() 
        
        # Learnable fuzziness
        #self.fuzziness_param = nn.Parameter(torch.tensor(1.0))
        #fuzziness = delta * self.fuzziness_param * torch.randn_like(scores)
        #fuzzy_scores = scores + fuzziness

        # Apply softmax to the adjusted fuzzy scores
        A = self.dropout(F.softmax(scale * fuzzy_scores, dim=-1))

        # Compute the output using the attention weights
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class QuantumInspiredAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False, num_heads=3):
        super(QuantumInspiredAttention, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale or 1.0 / math.sqrt(64), dtype=torch.float32))
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm(normalized_shape=64)
        self.attn_gate = nn.Sigmoid()

        # Define the linear layer to apply after the attention
        self.linear = nn.Linear(64, 64)  # Linear layer for output embeddings

        # Learnable scaling factor for normalization
        self.scale_factor = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def normalize_states(states):
        norm = torch.norm(states, dim=-1, keepdim=True)
        return states / (norm + 1e-6)

    def compute_superpositions(self, states, num_heads):
        B, L, E = states.shape
        attention_weights = F.softmax(torch.matmul(states, states.transpose(-2, -1)) / sqrt(E), dim=-1)
        superpositions = torch.einsum("bld,bld->blb", attention_weights, states)  # Use attention to create superpositions
        return self.dropout(superpositions)

    def compute_entangled_attention(self, states, superpositions):
        entangled_scores = torch.einsum("bld,hbd->hbl", states, superpositions)  # Compute entanglement score
        attention_scores = torch.abs(entangled_scores) ** 2
        return F.softmax(attention_scores, dim=-1)

    def update_embeddings(self, states, attention_scores):
        weighted_states = torch.einsum("hbl,bld->hbd", attention_scores, states)  # [H, B, D]
        updated_embeddings = weighted_states.mean(dim=0)  # Aggregate across heads
        
        # Apply linear transformation and GELU activation
        updated_embeddings = F.gelu(self.linear(updated_embeddings))
        
        return updated_embeddings

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, E = queries.shape
        quantum_queries = self.normalize_states(queries)
        quantum_keys = self.normalize_states(keys)

        # Generate superposition states for each head
        superposition_queries = self.compute_superpositions(quantum_queries, self.num_heads)
        superposition_keys = self.compute_superpositions(quantum_keys, self.num_heads)

        # Compute entangled attention scores
        attention_scores = self.compute_entangled_attention(quantum_queries, superposition_queries)

        if attn_mask is not None:
            attention_scores = attention_scores.masked_fill(attn_mask[:, None, None, :], -float('inf'))

        scaled_attention_scores = F.softmax(self.scale_factor * attention_scores, dim=-1)
        scaled_attention_scores = self.dropout(scaled_attention_scores)

        # Update embeddings with the new linear transformation and GELU activation
        updated_embeddings = self.update_embeddings(values, scaled_attention_scores)
        updated_embeddings = self.layer_norm(updated_embeddings + queries)  # Residual connection

        if self.output_attention:
            return updated_embeddings, scaled_attention_scores
        else:
            return updated_embeddings, None



class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
        

class TimeAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, max_len=100, covariate=False, flash_attention=False):
        super(TimeAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.covariate = covariate
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection, kwargs=dict(max_len=max_len),
                                          partial_factor=(0.0, 0.5),)
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)

    def forward(self, queries, keys, values, attn_mask, n_vars, n_tokens, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        seq_id = torch.arange(n_tokens * n_vars)
        seq_id = repeat(seq_id, 'n -> b h n', b=B, h=H)

        queries, keys = self.qk_proj(
            queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1. / sqrt(E)

        var_id = repeat(torch.arange(n_vars),
                        'C -> (C n_tokens)', n_tokens=n_tokens)
        var_id = repeat(var_id, 'L -> b h L', b=B, h=1).to(queries.device)

        attn_bias = self.attn_bias(var_id, var_id)

        if self.mask_flag:
            if attn_mask is None:
                if self.covariate:
                    attn_mask = TimerCovariateMask(
                        B, n_vars, n_tokens, device=queries.device)
                else:
                    attn_mask = TimerMultivariateMask(
                        B, n_vars, n_tokens, device=queries.device)
            attn_mask = attn_bias.masked_fill(attn_mask.mask, float("-inf"))
        else:
            attn_mask = attn_bias

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            scores += attn_mask
            
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None
            

class QuantumAttention(nn.Module):
    def __init__(self, num_qubits=4, mask_flag=True, scale=None, attention_dropout=0.1,
                 output_attention=False, entanglement_factor=0.5):
        """
        QuantumAttention module with Variational Quantum Eigensolver (VQE) for attention score computation.

        :param num_qubits: Number of qubits for the quantum circuit.
        :param mask_flag: Boolean flag to indicate whether to use masking.
        :param scale: Scaling factor for the attention scores.
        :param attention_dropout: Dropout rate for attention probabilities.
        :param output_attention: Boolean flag to return attention weights if True.
        :param entanglement_factor: A scalar controlling the weight of the entanglement term.
        """
        super(QuantumAttention, self).__init__()
        self.num_qubits = num_qubits
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.entanglement_factor = entanglement_factor

    def variational_circuit(self, params):
        """Quantum circuit for attention score calculation using a VQE approach."""
        for i in range(self.num_qubits):
            qml.RY(params[i], wires=i)

        # Entanglement using CNOT gates
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
       
        return qml.expval(qml.PauliZ(0))  # Expectation value as attention score
   
    def compute_quantum_attention(self, x):
        """Computes attention scores using a quantum variational circuit."""
        params = torch.rand(self.num_qubits, requires_grad=True)  # Random initialization
        qnode = qml.QNode(self.variational_circuit, self.dev, interface='torch')
        return qnode(params)

    def forward(self, queries, keys, values, attn_mask=None):
        """
        Forward pass for the Quantum Attention using VQE.

        :param queries: Query tensor of shape (B, L, H, E)
        :param keys: Key tensor of shape (B, S, H, D)
        :param values: Value tensor of shape (B, S, H, D)
        :param attn_mask: Mask to apply attention to certain locations
        :return: The computed attention output and optionally the attention weights
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Quantum-based attention score calculation
        superposition_scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        quantum_scores = self.compute_quantum_attention(superposition_scores)
       
        entanglement_scores = torch.einsum("blhd,bshe->bhls", values, keys)
        scores = quantum_scores + self.entanglement_factor * entanglement_scores

        # Apply mask (if any)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.triu(torch.ones(L, L), diagonal=1).bool().to(queries.device)
            scores.masked_fill_(attn_mask, -float('inf'))

        # Normalize scores using softmax
        A = self.dropout(F.softmax(scale * scores, dim=-1))

        # Compute the weighted sum of values
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class HybridChannelAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False,
                 mode='hybrid', num_heads=8, alpha=1.0, beta=0.0):
        super(HybridChannelAttention, self).__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.mode = mode  # 'ci', 'cd', 'hybrid'
        self.num_heads = num_heads
        self.alpha = nn.Parameter(torch.tensor(alpha))  # learnable scalar
        self.beta = nn.Parameter(torch.tensor(beta))    # learnable scalar

        if self.mode == 'cd':
            # Learnable matrix A for channel-channel dependency
            self.A = nn.Parameter(torch.eye(num_heads))  # [H, H]

    def compute_channel_mask(self, Q, K):
        """
        Computes correlation matrix R ∈ [-1, 1] for each channel and builds A.
        Args:
            Q, K: [B, H, L, D]
        Returns:
            A: [H, H] matrix for hybrid masking
        """
        # Mean over batch and time to get one [H, D] vector for Q and K
        Qh = Q.mean(dim=0).mean(dim=1)  # [H, D]
        Kh = K.mean(dim=0).mean(dim=1)  # [H, D]

        # Pearson correlation: R[i, j] = corr(Qh[i], Kh[j])
        Q_norm = Qh - Qh.mean(dim=1, keepdim=True)
        K_norm = Kh - Kh.mean(dim=1, keepdim=True)

        Q_std = Q_norm.norm(dim=1, keepdim=True) + 1e-6
        K_std = K_norm.norm(dim=1, keepdim=True) + 1e-6

        Q_hat = Q_norm / Q_std
        K_hat = K_norm / K_std

        R = torch.matmul(Q_hat, K_hat.T)  # [H, H] ∈ (-1,1)

        # Apply sigmoid(α·R + β)
        M = torch.sigmoid(self.alpha * R + self.beta)  # [H, H]
        return M

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        Args:
            queries, keys, values: [B, L, H, D]
            attn_mask: [B, H, L, L]
        """
        B, L, H, D = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1. / sqrt(D)

        # Base attention: [B, H, L, L]
        scores = torch.einsum("blhd,bshd->bhls", queries, keys)

        # Channel attention modifier A ∈ ℝ^{H×H}
        if self.mode == 'ci':
            A = torch.eye(H, device=queries.device)
        elif self.mode == 'cd':
            A = self.A  # learnable
        elif self.mode == 'hybrid':
            A = self.compute_channel_mask(queries.transpose(1, 2), keys.transpose(1, 2))  # [H, H]
        else:
            raise ValueError("Invalid mode. Use 'ci', 'cd', or 'hybrid'.")

        # Apply A to attention scores
        # scores: [B, H, L, L] × A[H, H] → weighted sum across heads
        scores = torch.einsum("bhls,hk->bkls", scores, A)  # [B, K, L, S]

        # Mask if needed
        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), float('-inf'))

        A_soft = torch.softmax(scale * scores, dim=-1)  # [B, H, L, S]
        A_soft = self.dropout(A_soft)

        # Output values: [B, L, H, D]
        V = torch.einsum("bhls,bshd->blhd", A_soft, values)

        if self.output_attention:
            return V.contiguous(), A_soft
        else:
            return V.contiguous(), None
