import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import repeat, rearrange, einsum
import gin
import math
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def activation_switch(activation: str) -> callable:
    if activation == "leaky_relu":
        return F.leaky_relu
    elif activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    # elif activation == "adaptive":
    #     return SlowAdaptiveRational()
    else:
        raise ValueError(f"Unrecognized `activation` func: {activation}")

class RNNEncoder(nn.Module):
    def __init__(self,
                 args,
                 # network size
                 layers_before_gru=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 latent_dim=32,
                 # actions, states, rewards
                 action_dim=2,
                 action_embed_dim=10,
                 state_dim=2,
                 state_embed_dim=10,
                 reward_size=1,
                 reward_embed_size=5,
                 ):
        super(RNNEncoder, self).__init__()

        self.args = args
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.reparameterise = self._sample_gaussian

        # embed action, state, reward
        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
        self.reward_encoder = utl.FeatureExtractor(reward_size, reward_embed_size, F.relu)

        # fully connected layers before the recurrent cell
        curr_input_dim = action_embed_dim + state_embed_dim + reward_embed_size
        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(curr_input_dim, layers_before_gru[i]))
            curr_input_dim = layers_before_gru[i]

        # recurrent unit
        # TODO: TEST RNN vs GRU vs LSTM
        self.gru = nn.GRU(input_size=curr_input_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          )

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_dim = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(curr_input_dim, layers_after_gru[i]))
            curr_input_dim = layers_after_gru[i]

        # output layer
        self.fc_mu = nn.Linear(curr_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_input_dim, latent_dim)

    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            raise NotImplementedError  # TODO: double check this code, maybe we should use .unsqueeze(0).expand((num, *logvar.shape))
            std = torch.exp(0.5 * logvar).repeat(num, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)

    def reset_hidden(self, hidden_state, done):
        """ Reset the hidden state where the BAMDP was done (i.e., we get a new task) """
        if hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        hidden_state = hidden_state * (1 - done)
        return hidden_state

    def prior(self, batch_size, sample=True):

        # TODO: add option to incorporate the initial state

        # we start out with a hidden state of zero
        hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(device)

        h = hidden_state
        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            h = F.relu(self.fc_after_gru[i](h))

        # outputs
        latent_mean = self.fc_mu(h)
        latent_logvar = self.fc_logvar(h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        return latent_sample, latent_mean, latent_logvar, hidden_state

    def forward(self, actions, states, rewards, hidden_state, return_prior, sample=True, detach_every=None):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        """

        # we do the action-normalisation (the the env bounds) here
        actions = utl.squash_action(actions, self.args)

        
    
        # shape should be: sequence_len x batch_size x hidden_size
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))
        if hidden_state is not None:
            # if the sequence_len is one, this will add a dimension at dim 0 (otherwise will be the same)
            hidden_state = hidden_state.reshape((-1, *hidden_state.shape[-2:]))

        if return_prior:
            # if hidden state is none, start with the prior
            prior_sample, prior_mean, prior_logvar, prior_hidden_state = self.prior(actions.shape[1])
            hidden_state = prior_hidden_state.clone()

        # extract features for states, actions, rewards
        ha = self.action_encoder(actions)
        hs = self.state_encoder(states)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=2)


        # forward through fully connected layers before GRU
        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))

        #raise RuntimeError(f"Debugging: h.shape = {h.shape} ha.shape = {ha.shape} hs.shape = {hs.shape} hr.shape = {hr.shape} asr = {actions.shape} {states.shape} {rewards.shape}")


        if detach_every is None:
            # GRU cell (output is outputs for each time step, hidden_state is last output)
            output, _ = self.gru(h, hidden_state)
        else:
            output = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                curr_input = h[i*detach_every:i*detach_every+detach_every]  # pytorch caps if we overflow, nice
                curr_output, hidden_state = self.gru(curr_input, hidden_state)
                output.append(curr_output)
                # detach hidden state; useful for BPTT when sequences are very long
                hidden_state = hidden_state.detach()
            output = torch.cat(output, dim=0)
        gru_h = output.clone()

        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            gru_h = F.relu(self.fc_after_gru[i](gru_h))

        # outputs
        latent_mean = self.fc_mu(gru_h)
        latent_logvar = self.fc_logvar(gru_h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        if return_prior:
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((prior_logvar, latent_logvar))
            output = torch.cat((prior_hidden_state, output))

        if latent_mean.shape[0] == 1:
            latent_sample, latent_mean, latent_logvar = latent_sample[0], latent_mean[0], latent_logvar[0]

        #raise RuntimeError(f"samplemeanlogavr : {latent_sample.shape} {latent_mean.shape} {latent_logvar.shape}")

        return latent_sample, latent_mean, latent_logvar, output

# AMAGO TRANFORMER
class Normalization(nn.Module):
    def __init__(self, method: str, d_model: int):
        super().__init__()
        assert method in ["layer", "none"]
        if method == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif method == "none":
            self.norm = lambda x: x
        self.method = method

    def forward(self, x):
        return self.norm(x)


@gin.configurable(allowlist=["window_size"])
class FlashAttention(nn.Module):
    def __init__(
        self,
        causal: bool = True,
        attention_dropout: float = 0.0,
        window_size: tuple[int, int] = (-1, -1),
    ):
        super().__init__()
        self.dropout = attention_dropout
        self.causal = causal
        self.window_size = window_size

    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        qkv = qkv.to(torch.bfloat16)
        if key_cache is None or val_cache is None or cache_seqlens is None:
            out = flash_attn.flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
                window_size=self.window_size,
            )
        else:
            assert not self.training
            q, k, v = qkv.unbind(2)
            out = flash_attn.flash_attn_with_kvcache(
                q=q,
                k_cache=key_cache,
                v_cache=val_cache,
                cache_seqlens=cache_seqlens,
                k=k,
                v=v,
                causal=self.causal,
                window_size=self.window_size,
            )
        return out


class SigmaReparam(nn.Module):
    """
    https://arxiv.org/pdf/2303.06296.pdf Appendix C
    """

    def __init__(self, d_in, d_out, bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_out, d_in), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(d_out), requires_grad=True) if bias else None
        u = torch.randn(d_out)
        self.register_buffer("u", u / u.norm(dim=0))
        v = torch.randn(d_in)
        self.register_buffer("v", v / v.norm(dim=0))
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # same as nn.Linear
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                u = (self.W @ self.v).float()
                self.u.data = u / u.norm(dim=0)
                v = (self.W.T @ self.u).float()
                self.v.data = v / v.norm(dim=0)
        sigma = einsum(self.u, self.W, self.v, "d, d c , c->")
        W_hat = self.gamma / sigma * self.W
        out = F.linear(x, W_hat, self.b)
        return out


class VanillaAttention(nn.Module):
    def __init__(self, causal: bool = True, attention_dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.causal = causal
        self._mask = None

    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        assert (
            key_cache is None and val_cache is None
        ), "VanillaAttention does not support `fast_inference` mode"
        queries, keys, values = torch.unbind(qkv, dim=2)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self._mask is None or self._mask.shape != (B, 1, L, L):
            self._mask = torch.triu(
                torch.ones((B, 1, L, L), dtype=torch.bool, device=qkv.device),
                diagonal=1,
            )
        if self.causal:
            scores.masked_fill_(self._mask, -torch.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V


@gin.configurable(allowlist=["head_scaling", "sigma_reparam"])
class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_qkv,
        n_heads,
        dropout_qkv=0.0,
        head_scaling: bool = True,
        sigma_reparam: bool = True,
    ):
        super().__init__()
        self.attention = attention
        FF = SigmaReparam if sigma_reparam else nn.Linear
        self.qkv_projection = FF(d_model, 3 * d_qkv * n_heads, bias=False)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.out_projection = FF(d_qkv * n_heads, d_model)
        self.head_scaler = nn.Parameter(
            torch.ones(1, 1, n_heads, 1), requires_grad=head_scaling
        )
        self.n_heads = n_heads

    def forward(self, sequence, key_cache=None, val_cache=None, cache_seqlens=None):
        qkv = self.dropout_qkv(self.qkv_projection(sequence))
        qkv = rearrange(
            qkv,
            "batch len (three d_qkv heads) -> batch len three heads d_qkv",
            heads=self.n_heads,
            three=3,
        )
        out = self.head_scaler * self.attention(
            qkv=qkv,
            key_cache=key_cache,
            val_cache=val_cache,
            cache_seqlens=cache_seqlens,
        )
        out = rearrange(out, "batch len heads dim -> batch len (heads dim)")
        out = self.out_projection(out)
        return out


@gin.configurable(denylist=["activation", "norm", "dropout_ff"])
class TransformerLayer(nn.Module):
    """
    Pre-Norm Self-Attention
    """

    def __init__(
        self,
        self_attention,
        d_model: int,
        d_ff: int,
        dropout_ff: float = 0.1,
        activation: str = "leaky_relu",
        norm: str = "layer",
        sigma_reparam: bool = True,
        normformer_norms: bool = True,
    ):
        super().__init__()
        self.self_attention = self_attention
        FF = SigmaReparam if sigma_reparam else nn.Linear
        self.ff1 = FF(d_model, d_ff)
        self.ff2 = FF(d_ff, d_model)
        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = (
            Normalization(method=norm, d_model=d_model)
            if normformer_norms
            else lambda x: x
        )
        self.norm3 = Normalization(method=norm, d_model=d_model)
        self.norm4 = (
            Normalization(method=norm, d_model=d_ff)
            if normformer_norms
            else lambda x: x
        )
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.activation = activation_switch(activation)

    def forward(self, self_seq, key_cache=None, val_cache=None, cache_seqlens=None):
        q1 = self.norm1(self_seq)  # pre-norm
        q1 = self.self_attention(
            q1, key_cache=key_cache, val_cache=val_cache, cache_seqlens=cache_seqlens
        )
        q1 = self.norm2(q1)  # normformer extra norm 1
        self_seq = self_seq + q1
        q1 = self.norm3(self_seq)  # regular norm
        # normformer extra norm 2
        q1 = self.norm4(self.activation(self.ff1(q1)))
        q1 = self.dropout_ff(self.ff2(q1))
        self_seq = self_seq + q1
        return self_seq


class Cache:
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
    ):
        self.data = torch.zeros(
            (batch_size, max_seq_len, n_heads, head_dim), dtype=dtype, device=device
        )
        # make silent bugs in k/v cache... much louder
        self.data[:] = torch.nan

    def __len__(self):
        return self.data.shape[1]

    def roll_back(self, idx):
        roll = self.data[idx, 1:].clone()
        self.data[idx, :-1] = roll
        self.data[idx, -1] = torch.nan  # no silent bugs


class TformerHiddenState:
    def __init__(
        self, key_cache: list[Cache], val_cache: list[Cache], timesteps: torch.Tensor
    ):
        assert isinstance(key_cache, list) and len(key_cache) == len(val_cache)
        assert timesteps.dtype == torch.int32
        self.n_layers = len(key_cache)
        self.key_cache = key_cache
        self.val_cache = val_cache
        self.timesteps = timesteps

    def reset(self, idxs):
        self.timesteps[idxs] = 0

    def update(self):
        self.timesteps += 1
        for i, timestep in enumerate(self.timesteps):
            if timestep == len(self.key_cache[0]):
                for k, v in zip(self.key_cache, self.val_cache):
                    k.roll_back(i)
                    v.roll_back(i)
                self.timesteps[i] -= 1

    def __getitem__(self, layer_idx):
        assert layer_idx < self.n_layers
        return (
            self.key_cache[layer_idx].data,
            self.val_cache[layer_idx].data,
            self.timesteps,
        )


class FixedPosEmb(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, pos_idxs: torch.LongTensor):
        B, L = pos_idxs.shape
        emb = torch.zeros(
            (B, L, self.d_model), device=pos_idxs.device, dtype=torch.float32
        )
        coeff = torch.exp(
            (
                torch.arange(0, self.d_model, 2, device=emb.device, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )
        )
        emb[..., 0::2] = torch.sin(pos_idxs.float().unsqueeze(-1) * coeff)
        emb[..., 1::2] = torch.cos(pos_idxs.float().unsqueeze(-1) * coeff)
        return emb


class Transformer(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        max_pos_idx: int,
        d_model: int = 128,
        d_ff: int = 512,
        d_emb_ff: int = None,
        n_heads: int = 4,
        layers: int = 3,
        dropout_emb: float = 0.05,
        dropout_ff: float = 0.05,
        dropout_attn: float = 0.00,
        dropout_qkv: float = 0.00,
        attention: str = "flash",
        activation: str = "leaky_relu",
        norm: str = "layer",
        causal: bool = True,
        pos_emb: str = "learnable",
    ):
        super().__init__()
        assert attention in ["flash", "vanilla"]
        assert pos_emb in ["learnable", "fixed"]

        # embedding
        if pos_emb == "learnable":
            self.position_embedding = nn.Embedding(
                max_pos_idx + 1, embedding_dim=d_model
            )
        elif pos_emb == "fixed":
            self.position_embedding = FixedPosEmb(d_model)
        d_emb_ff = d_emb_ff or d_model
        self.inp = nn.Linear(inp_dim, d_model)
        self.dropout = nn.Dropout(dropout_emb)

        self.head_dim = d_model // n_heads
        assert self.head_dim in range(8, 129, 8)
        self.n_heads = n_heads
        self.n_layers = layers
        Attn = FlashAttention if attention == "flash" else VanillaAttention

        def make_layer():
            return TransformerLayer(
                self_attention=AttentionLayer(
                    attention=Attn(causal=causal, attention_dropout=dropout_attn),
                    d_model=d_model,
                    d_qkv=self.head_dim,
                    n_heads=self.n_heads,
                    dropout_qkv=dropout_qkv,
                ),
                d_model=d_model,
                d_ff=d_ff,
                dropout_ff=dropout_ff,
                activation=activation,
                norm=norm,
            )

        self.layers = nn.ModuleList([make_layer() for _ in range(layers)])
        self.norm = Normalization(method=norm, d_model=d_model)
        self.d_model = d_model

    @property
    def emb_dim(self):
        return self.d_model

    def forward(self, seq, pos_idxs, hidden_state: None | TformerHiddenState):
        if self.training:
            assert hidden_state is None
        batch, length, dim = seq.shape
        h = hidden_state or [[None, None, None] for _ in range(self.n_layers)]

        # emedding
        pos_emb = self.position_embedding(pos_idxs)
        traj_emb = self.inp(seq)
        traj_emb = self.dropout(traj_emb + pos_emb)

        # self-attention
        for i, layer in enumerate(self.layers):
            traj_emb = layer(traj_emb, *h[i])
        traj_emb = self.norm(traj_emb)

        if hidden_state is not None:
            # controls the sequence length of the k/v cache
            hidden_state.update()

        return traj_emb, hidden_state

#project results back to d_model for residuals - need to do 
class TransformerEncoder(nn.Module):
    def __init__(self,
                 args,
                 layers_before_transformer=(),
                 layers_after_transformer = (),
                 d_model=128,
                 n_heads=3,
                 num_transformer_layers=3,
                 d_ff=512,
                 latent_dim=32,

                 # actions, states, rewards
                 action_dim=2,
                 action_embed_dim=10,
                 state_dim=2,
                 state_embed_dim=10,
                 reward_size=1,
                 reward_embed_size=5,
                 max_seq_len=100,
                 ):
        super(TransformerEncoder, self).__init__()

        self.args = args
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.reparameterise = self._sample_gaussian
        self.max_seq_len = max_seq_len

        # embed action, state, reward
        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
        self.reward_encoder = utl.FeatureExtractor(reward_size, reward_embed_size, F.relu)

        # fully connected layers before the transformer
        curr_input_dim = action_embed_dim + state_embed_dim + reward_embed_size
        self.fc_before_transformer = nn.ModuleList([])
        for i in range(len(layers_before_transformer)):
            self.fc_before_transformer.append(nn.Linear(curr_input_dim, layers_before_transformer[i]))
            curr_input_dim = layers_before_transformer[i]

        self.positional_embedding = nn.Linear(max_seq_len, d_model)
        self.input_projection = nn.Linear(curr_input_dim, d_model)
        #use for testing skip connection
        #self.project_to_dim = nn.Linear(curr_input_dim, d_model)
        
        # make x transformer layers
        self.transformer_layers = nn.ModuleList(   
        [TransformerLayer(
                self_attention=AttentionLayer(
                        attention=VanillaAttention(causal=True, attention_dropout=0.1),
                        d_model=d_model,
                        d_qkv=d_model//n_heads,
                        n_heads=n_heads,
                        dropout_qkv=0.1,
                ),
                d_model=d_model,
                d_ff=d_ff,
                dropout_ff=0.1,
                activation="relu",
                norm="layer"
            ) 
        for i in range(num_transformer_layers)])

        self.norm = Normalization(method="layer", d_model=d_model)

        # fully connected layers after the transformer
        curr_input_dima = d_model
        self.fc_after_transformer = nn.ModuleList([])
        for i in range(len(layers_after_transformer)):
            self.fc_after_transformer.append(nn.Linear(curr_input_dima, layers_after_transformer[i]))
            curr_input_dima = layers_after_transformer[i]

        # output layer
        self.fc_mu = nn.Linear(curr_input_dima, latent_dim)
        self.fc_logvar = nn.Linear(curr_input_dima, latent_dim)

    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            raise NotImplementedError  # TODO: double check this code, maybe we should use .unsqueeze(0).expand((num, *logvar.shape))
            std = torch.exp(0.5 * logvar).repeat(num, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)
        
    def prior(self, batch_size, sample=True):
        # We start out with a latent state initialized to zero
        hidden_state = torch.zeros((1,batch_size, self.d_model), requires_grad=True).to(device)
        # Forward through fully connected layers after Transformer
        for i in range(len(self.fc_after_transformer)):
            print(hidden_state.shape)
            hidden_state = F.relu(self.fc_after_transformer[i](hidden_state))

        # Outputs
        latent_mean = self.fc_mu(hidden_state)
        latent_logvar = self.fc_logvar(hidden_state)
        
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        return latent_sample, latent_mean, latent_logvar, hidden_state

    #hidden_state and detach_every=None are vestiges from RNNEncoder just used for reuse of metalearner
    def forward(self, actions, states, rewards, hidden_state, return_prior=False, sample=True, detach_every=None):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For feeding in entire trajectories, sequence_len > 1.
        """

        actions = utl.squash_action(actions, self.args)

        # Embed actions, states, rewards
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))

        if return_prior:
            # if hidden state is none, start with the prior
            prior_sample, prior_mean, prior_logvar, prior_hidden_state = self.prior(actions.shape[1])

        # extract features for states, actions, rewards
        ha = self.action_encoder(actions)
        hs = self.state_encoder(states)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=2)

        # Forward through fully connected layers before Transformer
        for fc_layer in self.fc_before_transformer:
            h = F.relu(fc_layer(h))

        # positional embeddings
        # h.shape init 1 1 16
        pos_idxs = torch.arange(self.max_seq_len).unsqueeze(0).unsqueeze(1).to(h.device).float()
        # adding the position to the vector, then passing to pos_embedding layer (so that position info is integrated with state,reward,action)
        paddedh = F.pad(h, (0, self.max_seq_len - h.shape[2]))
        
        # combined position embed below
        pos_emb = self.positional_embedding(paddedh + pos_idxs)
        # separated position embed below
        # pos_emb = self.positional_embedding(pos_idxs)
        h = self.input_projection(h) + pos_emb

        #for residuals (for a proper transformer arch)
        positionally_encoded_embedding = h

        # Forward through Transformer layers
        for layer in self.transformer_layers:
            h = self.norm(h)
            h = layer(h)
        
        #for residuals (for a proper transformer arch)
        #dont forget norms
        h += positionally_encoded_embedding
        attention_output = h
        
        h = self.norm(h)

        # Forward through fully connected layers after Transformer
        for fc_layer in self.fc_after_transformer:
            h = F.relu(fc_layer(h))
        #dont forget norms
        #h = self.project_to_dim(h)
        h += attention_output
        h = self.norm(h)

        # Outputs
        latent_mean = self.fc_mu(h)
        latent_logvar = self.fc_logvar(h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        if return_prior:
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((prior_logvar, latent_logvar))
        
        #reduce extra dim
        latent_sample = latent_sample.squeeze(0)
        latent_mean = latent_mean.squeeze(0)
        latent_logvar = latent_logvar.squeeze(0)
        
        return latent_sample, latent_mean, latent_logvar, h