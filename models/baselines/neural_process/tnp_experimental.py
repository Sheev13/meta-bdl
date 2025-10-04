import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import warnings
from base_networks.base_architectures import MLP
from ...likelihoods import GaussianLikelihood

from typing import List, Optional

class MHALayer(nn.Module):
    """Represents a general-purpose dot-product multi-head attention layer."""
    def __init__(self, d_emb: int, num_heads: int = 8):
        super().__init__()
        if d_emb % num_heads != 0:
            raise ValueError("Transformer embedding dimension must be divisible by the number of heads.")
        d_head = d_emb // num_heads

        self.q_proj = nn.Linear(d_emb, d_emb, bias=True)
        self.k_proj = nn.Linear(d_emb, d_emb, bias=True)
        self.v_proj = nn.Linear(d_emb, d_emb, bias=True)
        self.out_proj = nn.Linear(d_emb, d_emb, bias=True)

        self.d_emb = d_emb
        self.num_heads = num_heads
        self.d_head = d_head

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q, k, and v are all shape (samples, n, d_emb)
        samples, n = q.shape[:2]
        q = q.reshape(samples, n, self.num_heads, self.d_head).transpose(1, 2) # shape (samples, num_heads, n, d_head)
        k = k.reshape(samples, n, self.num_heads, self.d_head).transpose(1, 2) # "
        v = v.reshape(samples, n, self.num_heads, self.d_head).transpose(1, 2) # "

        scores = q @ k.transpose(-2, -1) / (self.d_head**0.5) # shape (samples, num_heads, n, n)
        attn_weights = F.softmax(scores, dim=-1)

        attn_output = attn_weights @ v # shape (samples, num_heads, n, d_head)
        return attn_output.transpose(1, 2).contiguous().reshape(samples, n, self.d_emb)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        attn = self.scaled_dot_product_attention(q, k, v)

        return self.out_proj(attn)
    

class MHABlock(nn.Module):
    def __init__(self, d_emb: int, num_heads: int = 8, nonlinearity: nn.Module = nn.ReLU()):
        super().__init__()
        self.mha = MHALayer(d_emb, num_heads)
        self.ln1 = nn.LayerNorm(d_emb)
        self.ln2 = nn.LayerNorm(d_emb)
        self.ff = MLP(3*[d_emb], nonlinearity=nonlinearity)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        attn = self.mha(q, k, v)
        res1 = attn + q
        ln1 = self.ln1(res1)
        ff = self.ff(ln1)
        res2 = ff + ln1
        ln2 = self.ln2(res2)

        return ln2


class MHSABlock(MHABlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, qkv: torch.Tensor):
        return super().forward(qkv, qkv, qkv)
    

class MHCABlock(MHABlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        return super().forward(q, kv, kv)


class EQTNPBlock(nn.Module):
    def __init__(self, d_emb: int, num_heads: int = 8, nonlinearity: nn.Module = nn.ReLU()):
        super().__init__()
        self.mhsa = MHSABlock(d_emb, num_heads, nonlinearity)
        self.mhca = MHCABlock(d_emb, num_heads, nonlinearity)

    def forward(self, Zc: torch.Tensor, Zt: torch.Tensor):
        if len(Zc.shape) == 2:
            Zc = Zc.unsqueeze(0)
        if len(Zt.shape) == 2:
            Zt = Zt.unsqueeze(0)

        Zc = self.mhsa(Zc)
        Zt = self.mhca(Zt, Zc)

        return Zc, Zt
    

class DualSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
    
    def forward(self, Zc, Zt):
        for m in self.modules_list:
            Zc, Zt = m(Zc, Zt)
        return Zc, Zt


class EQTNP(nn.Module):
    def __init__(self,
                 x_dim: Optional[int] = None,
                 y_dim: Optional[int] = None,
                 num_blocks: int = 2,
                 d_emb: int = 64,
                 num_heads: int = 8,
                 nonlinearity: nn.Module = nn.ReLU()
                ):
        super().__init__()
        if x_dim is None or y_dim is None:
            raise ValueError("User failed to provide dimensionality of inputs/outputs to EQTNP class.")
        self.ctxt_tokeniser = MLP([x_dim+y_dim, d_emb, d_emb], nonlinearity)
        self.trgt_tokeniser = MLP([x_dim, d_emb, d_emb], nonlinearity)
        
        transformer_blocks = [EQTNPBlock(d_emb, num_heads, nonlinearity) for _ in range(num_blocks)]
        self.transformer = DualSequential(*transformer_blocks)

        self.decoder = MLP([d_emb, d_emb, 2*y_dim], nonlinearity)

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_blocks = num_blocks
        self.d_emb = d_emb
        self.num_heads = num_heads
        self.nonlinearity = nonlinearity

    def certify_shapes(self, a: torch.Tensor, is_output: bool = False):
        d_dim = self.y_dim if is_output else self.x_dim
        if len(a.shape) == 0:
            a = a.unsqueeze(0).unsqueeze(0) # user forgot to unsqueeze anything and has passed a single element
        if len(a.shape) == 1:
            if d_dim == 1:
                a = a.unsqueeze(-1) # user forgot to unsqueeze for data dimensionality of 1
            else:
                a = a.unsqueeze(0) # user forgot to unsqueeze for dataset size of 1

        return a

    def forward(self, Xt: torch.Tensor, Xc: torch.Tensor, Yc: torch.Tensor):
        # the following just ensure each tensor is (n, d_dim) 
        #   where n is either n_c or n_t and d_dim is either x_dim or y_dim.
        Xt = self.certify_shapes(Xt)
        Xc = self.certify_shapes(Xc)
        Yc = self.certify_shapes(Yc, is_output=True)

        Zc = torch.cat((Xc, Yc), dim=-1) # shape (n_c, x_dim+y_dim)
        Zc = self.ctxt_tokeniser(Zc)
        Zt = self.trgt_tokeniser(Zt)

        Zc, Zt = self.transformer(Zc, Zt)

        y_t_params = self.decoder(Zt)
        means, stds = y_t_params[:,:self.y_dim], 0.001+0.999*nn.functional.softplus(y_t_params[:,self.y_dim:])
        return torch.distributions.Normal(means, stds)


    def loss(self, Xc, yc, Xt, yt, **redundant_kwargs):
        """Predictive log likelihood of targets given contexts"""
        predictive = self(Xt, Xc, yc)
        ll = predictive.log_prob(yt)

        metrics = {
            "ll": ll.detach().item(),
        }
            
        return - ll, metrics