import torch
import math

# Inverse dim formula to find dim based on number of rotations
def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    """
    对于波⻓⼤于原最⼤序列⻓度的那些低频分量, 不应该对他们进⾏外推, 模型本身也未见过，所以需要通过内插， 让模型见过它们内插缩小后的相对位置

    d = D/2* log_{base}(L/(2*pi*num_rotations)) 
    """
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

# Find dim range bounds based on rotations
def find_correction_range(low_rot, # 32, 
                          high_rot, # 1
                          dim, base=10000, max_position_embeddings=2048):
    low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)  # Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity
    # linear_func: [dim]
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

"""
yarn:
    sqrt(1/t) = 0.1*ln(s) + 1
"""
def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

class LlamaYaRNScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1, original_max_position_embeddings=2048, extrapolation_factor=1, attn_factor=1, 
                 beta_fast=32, # 高频外推，啥也不干
                 beta_slow=1, # 低频内插，缩小theta_d
                 finetuned=False, 
                 device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast # 32 # 高频外推，啥也不干
        self.beta_slow = beta_slow # 1, # 低频内插，缩小theta_d

        self.yarn(device)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        # postion_idx: [max_seq_len]
        position_idx = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # freqs: [max_seq_len, head_dim//2]
        # inv_freqs: [dim//2]
        #freqs = torch.einsum("i,j->ij", position_idx, self.inv_freq)
        freqs = position_idx[:, None] @ self.inv_freq[None, :]
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()

        self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            # 超出了max_seq_len_cached, 需要重新计算cos和sin
            self.max_seq_len_cached = seq_len

            position = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            #freqs = torch.einsum("i,j->ij", position, self.inv_freq)
            freqs = position[:,None] @ self.inv_freq[None, :]
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    def yarn(self, device):
        # pos_freqs: base^(2i/D) for i = 0, 1, ..., dim/2 - 1
        # pos_freqs: [dim//2]
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)

        # inv_freq_extrapolation: base^(-2i/D),外推
        inv_freq_extrapolation = 1.0 / pos_freqs # 外推就是啥也不干！！！

        # 内插，inv_freq_interpolation，即为theta_d: base^(-2i/D),内插, 将缩小theta_d, 减缓旋转的角度为以前的1/scale
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        alpha_freq_low, beta_freq_high = find_correction_range(self.beta_fast,  # 32
                                                               self.beta_slow,
                                                               self.dim,
                                                               self.base,
                                                               self.original_max_position_embeddings)
        # 中间部分，高频外推，低频内插
        # r(d): 训练⻓长度内旋转的周期个数, r(i)=L/(2*pi*base^(2i/d))=L/(2*pi)*base^(-2i/d)
        # gamma_rd: 
        gamma_rd = linear_ramp_mask(alpha_freq_low, beta_freq_high, self.dim // 2).float().to(device)

        # NOTE:这里就是 除以factor?
        inv_freq_mask = (1 - gamma_rd) * self.extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
        # 内插 + 外推
        """
        1. r(d)>beta, 高频外推,就是保持原rope不变
        2. alpha<r(d)<beta, 中间部分，高频外推，低频内插
        3. r(d)<alpah=1，低频内插,缩小旋转角度，降低旋转频率，让模型见过它们内插缩小后的相对位置
        """
        # inv_freqs: [dim//2]
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq)

        """
        m_scale: 
           sqrt(1/t) = 0.1*ln(s) + 1
        """
        self.mscale = float(get_mscale(self.scale) * self.attn_factor) # Get n-d magnitude scaling corrected for interpolation