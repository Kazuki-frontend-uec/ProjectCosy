import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# FSQ Whisper Tokenizer: FSQ = Finit Scalar Quantizer
# -----------------------------
class FSQWhisperTokenizer(nn.Module):
    """
    FSQ:
    Forward 順伝搬:
        入力: [N, dim]  or [B, T, dim]
    Returns 戻り値:
        recon: same shape as x (quantized->reconstructed)
        indices: LongTensor of shape [N, D] or [B, T, D] (per-factor indices)
        vq_loss: scalar (codebook+commit losses)
    """
    def __init__(
        self,
        num_factors=8,
        K=4,
        dim=512,
        decay=0.99,
        eps=1e-5,
        use_ema=True
        ):
        super().__init__()
        assert dim % num_factors == 0, "dim must be divisible by num_factors"
        self.D = num_factors
        self.K = K
        self.dim = dim
        self.subdim = dim // num_factors
        self.decay = decay
        self.eps = eps
        self.use_ema = use_ema

        # codebook: [D, K, subdim]
        embed = torch.randn(self.D, self.K, self.subdim) * 0.01
        self.register_buffer("embedding", embed)  # buffer so saved in state_dict
        if use_ema:
            self.register_buffer("cluster_size", torch.zeros(self.D, self.K))
            self.register_buffer("embed_avg", embed.clone())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [..., dim] where ... can be (N,) or (B,T,)
        returns recon (same shape), indices (same leading dims + D), vq_loss (scalar)
        """
        orig_shape = x.shape
        need_unsqueeze = False
        if x.dim() == 3:
            # [B, T, dim] => flatten to [B*T, dim]
            B, T, _ = x.shape
            flat = x.view(-1, self.dim)
            leading_shape = (B, T)
        elif x.dim() == 2:
            flat = x
            leading_shape = (flat.shape[0],)
        else:
            raise ValueError("x must be [N,dim] or [B,T,dim]")

        # reshape to [N, D, subdim]
        N = flat.shape[0]
        sub = flat.view(N, self.D, self.subdim)  # [N, D, subdim]

        # sub_flat = sub.view(N * self.D, self.subdim)             # [N*D, subdim]
        # emb_flat = self.embedding.view(self.D * self.K, self.subdim)  # [D*K, subdim]

        emb = self.embedding  # [D, K, subdim]
        # sub: [N, D, subdim] -> [N, D, 1, subdim]
        sub_exp = sub.unsqueeze(2)  # [N, D, 1, subdim]
        # emb: [D, K, subdim] -> [1, D, K, subdim]
        emb_exp = emb.unsqueeze(0)  # [1, D, K, subdim]
        # compute L2 distances: [N, D, K], (a - b)^2 = a^2 + b^2 - 2ab
        # dist = torch.sum((sub_exp - emb_exp) ** 2, dim=-1)  # [N, D, K]
        sub_norm = (sub ** 2).sum(dim=-1, keepdim=True)       # [N, D, 1]
        emb_norm = (emb ** 2).sum(dim=-1).unsqueeze(0)        # [1, D, K]
        cross = torch.einsum("ndm,dkm->ndk", sub, emb)        # [N, D, K]
        dist = sub_norm + emb_norm - 2 * cross

        # argmin to get indices per factor
        idx = torch.argmin(dist, dim=-1)  # [N, D] long

        # idx: [N, D] -> quantized vectors: [N, D, subdim]
        quant = self.embedding[torch.arange(self.D).unsqueeze(0), idx]  # [N, D, subdim]

        # reconstructed flat vectors: [N, dim]
        z_q = quant.view(N, self.dim)

        commitment_cost = 0.25
        vq_loss = F.mse_loss(z_q.detach(), flat) + commitment_cost * F.mse_loss(z_q, flat.detach())

        z_q_st = flat + (z_q - flat).detach()  # gradient flows into flat (encoder) only

        # EMA update (only during training)
        if self.training and self.use_ema:
            with torch.no_grad():
                one_hot = F.one_hot(idx, num_classes=self.K).type_as(flat)  # [N, D, K]
                # sum over N: [D, K]
                counts = one_hot.sum(dim=0)  # [D, K]
                embed_sum = torch.einsum("ndk,ndm->dkm", one_hot, sub)  # [D, K, subdim]

                # EMA updates
                self.cluster_size.data.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # normalize
                n = self.cluster_size.sum(dim=1, keepdim=True)  # [D,1]
                # avoid divide by zero
                cluster_size = self.cluster_size + self.eps
                cluster_size = cluster_size / cluster_size.sum(dim=1, keepdim=True) * n
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(-1)
                self.embedding.data.copy_(embed_normalized)

        # reshape outputs back to original leading dims
        if x.dim() == 3:
            recon = z_q_st.view(B, T, self.dim)
            indices = idx.view(B, T, self.D)
        else:
            recon = z_q_st.view(N, self.dim)
            indices = idx.view(*leading_shape, self.D) if isinstance(leading_shape, tuple) and len(leading_shape)==1 else idx.view(*leading_shape, self.D)

        return recon, indices, vq_loss
