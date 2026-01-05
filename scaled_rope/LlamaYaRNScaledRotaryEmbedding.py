import torch
import math
"""
github:
https://github.com/hkxIron/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py

yarn:
    sqrt(1/t) = 0.1*ln(s) + 1
"""
# Inverse dim formula to find dim based on number of rotations
def find_correction_dim_by_rotation(num_rotations:int, dim:int, base=10000, max_position_embeddings=2048):
    """
    这个函数的作用是：给定一个旋转次数（频率阈值），计算出对应的维度索引（dim index）。

    :param num_rotations: 在长度L内，旋转的圈数 rotation = r = L/lambda
    :param dim: 为序列的最大维度
    :param max_position_embeddings: 最长序列的token数

    对于波⻓⼤于原最⼤序列⻓度的那些低频分量, 不应该对他们进⾏外推, 模型本身也未见过，所以需要通过内插，让模型见过它们内插缩小后的相对位置
    公式推导：
    lambda = 2*pi/w = 2*pi/theta_d = 2*pi/base^(-2*i/D)
    rotation = L/lambda
    =>
    rotation =  L/(2*pi*base^(2*i/D))
    =>
    i = D/2* log_{base}(L/(2*pi*num_rotations))
    =>
    i = D/2* log(L/(2*pi*num_rotations))/log(base)
    """
    return dim/2 * math.log(max_position_embeddings/(num_rotations * 2 * math.pi), base=base)

# Find dim range bounds based on rotations
def find_correction_dim_range(fast_rotation:int,  # 高频部分，32,
                              low_rotation:int,  # 低频部分1
                              dim:int,
                              base=10000,
                              max_position_embeddings=2048):
    """
    这个函数定义了混合区间的边界。

    :param fast_rotation: (beta_fast_rotation = 32)：在这个旋转次数以上的（高频部分, 即dim低维部分），不需要插值，保持原样（外推）。
    :param low_rotation:(beta_slow_rotation = 1)：在这个旋转次数以下的（极低频部分,即dim高维部分），必须完全插值（内插），否则模型没见过这么长的相对距离。
    :param dim:
    :param base:
    :param max_position_embeddings:
    :return: 返回值：low 和 high 是维度的索引范围。在这个范围内的维度，将混合使用“外推”和“内插”。

    """
    low_dim = math.floor(find_correction_dim_by_rotation(fast_rotation, dim, base, max_position_embeddings))
    high_dim = math.ceil(find_correction_dim_by_rotation(low_rotation, dim, base, max_position_embeddings))
    return max(low_dim, 0), min(high_dim, dim-1)  # Clamp values just in case


def linear_ramp_mask(min, max, dim:int):
    """
    这是一个平滑过渡函数。 ramp: 斜坡

    它生成一个形状为 [dim] 的向量。
    在 min 之前的值为 0。
    在 max 之后的值为 1。
    中间部分从 0 线性增加到 1。
    作用：用于控制“内插”和“外推”的混合比例。

    """
    if min == max:
        max += 0.001  # Prevent singularity
    # linear_func: [dim]
    linear_func = (torch.arange(end=dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

"""
yarn:
    sqrt(1/t) = 0.1*ln(s) + 1, 该值恒>1
"""
def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

class LlamaYaRNScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim:int,
                 max_position_embeddings=2048,
                 base=10000,
                 scale=1,
                 original_max_position_embeddings=2048,
                 extrapolation_factor=1,
                 attn_factor=1, # attention的缩放因子，默认为1
                 beta_fast_rotation=32,  # 高频外推，在指定长度内完整的旋转圈数, 低维度
                 beta_slow_rotation=1,  # 低频内插，缩小theta_d, 高维度
                 finetuned=False,
                 device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        # 外推
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast_rotation = beta_fast_rotation # NOTE: 高频的旋转圈数阈值=32, 高频外推, 在这个旋转次数以上的（高频部分, 即dim低维部分），不需要插值，保持原样（外推）。 高频外推，啥也不干
        self.beta_slow_rotation = beta_slow_rotation # NOTE: 低频的旋转圈数阈值=1, 低频内插，缩小theta_d, 在这个旋转次数以下的（极低频部分,即dim高维部分），必须完全插值（内插）

        self.yarn(device)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        # postion_idx: [max_seq_len]
        position_idx = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # freqs: [max_seq_len, head_dim//2]
        # inv_freqs: [dim//2]
        #freqs = torch.einsum("i,j->ij", position_idx, self.inv_freq)
        # 或 freqs = torch.outer(position_idx, self.inv_freq)
        freqs = position_idx[:, None] @ self.inv_freq[None, :]
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()

        # NOTE: mscale: 幅值缩放因子, mscale = 0.1 * ln(scale) + 1.0
        #   [seq_len, dim/2] -> [batch=1, head_num=1, seq_len, dim]
        self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_dim]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            # NOTE: 超出了max_seq_len_cached, 需要重新计算cos(m*theta)和sin(m*theta)
            self.max_seq_len_cached = seq_len

            position = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            #freqs = torch.einsum("i,j->ij", position, self.inv_freq)
            # 或者 freqs = torch.outer(position, self.inv_freq)
            freqs = position[:,None] @ self.inv_freq[None, :]
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            # 这里的 * self.mscale 是 YaRN 特有的, 这意味着在 forward 过程中，输入的 x 不需要再手动乘 mscale，直接应用 RoPE 就会自动带上幅值缩放。
            self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    def yarn(self, device):
        # 即theta(d), 物理意义为角速度
        # pos_freqs: base^(2i/D) for i = 0, 1, ..., dim/2 - 1
        # pos_freqs: [dim//2]
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)

        # NOTE: inv_freq_extrapolation = theta(d) base^(-2i/D),外推, 外推就是啥也不变
        inv_freq_extrapolation = 1.0 / pos_freqs

        # NOTE: 内插，inv_freq_interpolation = theta(d)/s, s = L'/L, 为缩放的长度
        #   减缓旋转的角度为以前的1/scale,
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        alpha_low_dim_of_fast_rotation, beta_high_dim_of_slow_rotation = find_correction_dim_range(self.beta_fast_rotation,  # 高频旋转
                                                                                                   self.beta_slow_rotation,  # 低频旋转
                                                                                                   self.dim,
                                                                                                   self.base,
                                                                                                   self.original_max_position_embeddings)
        # 中间部分，高频外推，低频内插
        # r(d): 训练⻓长度内旋转的周期个数, r(i)=L/(2*pi*base^(2i/d))=L/(2*pi)*base^(-2i/d)
        # gamma(rd) = 1, if r(d)> beta_rotation_fast, 外推
        # gamma(rd) = 0, if r(d)< beta_rotation_slow, 内插
        # gamma(rd) = (r(d)-beta_rotation_slow)/(beta_rotation_fast-beta_rotation_slow), if beta_rotation_slow<r(d)<beta_rotation_fast, 中间部分，内插+外推
        # gamma_rd_in_dim: 在低维度时为0, 在高维度处为1, 从左到右逐渐增大
        gamma_rd_in_dim = linear_ramp_mask(alpha_low_dim_of_fast_rotation, beta_high_dim_of_slow_rotation, dim=self.dim // 2).float().to(device)


        # gamma_rd_in_dim: 在低维度时为0, 在高维度处为1
        # 因此gamma_rd_mask需要使用 1 - gamma_rd_in_dim
        # 解释：低维度是高频分量，需要外推， 高维度是低频分量，需要内插
        gamma_rd_mask = (1 - gamma_rd_in_dim) * self.extrapolation_factor # Get n-d rotational scaling corrected for extrapolation

        """
        计算yarn的新inv_freqs
        1. r(d)>beta, 低维度，高频外推,就是保持原rope不变
        2. alpha<r(d)<beta, 中间部分，高频外推，低频内插
        3. r(d)<alpah=1，高维度, 低频内插,缩小旋转角度，降低旋转频率，让模型见过它们内插缩小后的相对位置
        """
        # inv_freqs: [dim//2]
        inv_freq = inv_freq_extrapolation * gamma_rd_mask + inv_freq_interpolation * (1 - gamma_rd_mask)
        self.register_buffer("inv_freq", inv_freq)

        """
        m_scale: sqrt(1/t) = 0.1*ln(s) + 1
        注意：m_scale会直接乘到q*k的内积中，放大进入softmax之间的值，出现嬴者通吃 
        """
        self.mscale = float(get_mscale(self.scale) * self.attn_factor) # Get n-d magnitude scaling corrected for interpolation