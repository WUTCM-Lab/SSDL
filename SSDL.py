import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

NUM_CLASS = 3
CHANNEL = 224

class SpectralRoPE(nn.Module):
    def __init__(self, shape, dim_head, num_heads):
        super().__init__()
        self.H, self.W = shape
        self.dim_head = dim_head
        self.num_heads = num_heads

        self.pos_proj = nn.Sequential(
            nn.Conv1d(dim_head, dim_head * 2, kernel_size=3, padding=1, groups=dim_head),
            nn.GELU(),
            nn.Conv1d(dim_head * 2, dim_head, kernel_size=1)
        )

        self.omega = nn.Parameter(torch.linspace(0.1, 10, dim_head // 2))

        self.pos_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim_head, dim_head, kernel_size=1),
            nn.Sigmoid()
        )

    def _get_spectral_pe(self, L):
        grid = torch.arange(L, device=self.omega.device).float()
        rel_pos = grid[:, None] - grid[None, :]  # [L, L]

        freqs = self.omega[None, None, :] * rel_pos[..., None]  # [L, L, d/2]
        pe = torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # [L, L, d]

        pe = rearrange(pe, 'l1 l2 d -> d (l1 l2)')
        pe = self.pos_proj(pe.unsqueeze(0)).squeeze(0)  # [d, L*L]
        pe = rearrange(pe, 'd (l1 l2) -> l1 l2 d', l1=L)

        gate = self.pos_gate(rearrange(pe, 'l1 l2 d -> d (l1 l2)'))
        pe = pe * rearrange(gate, '(l1 l2) d-> l1 l2 d', l1=L)

        return pe  # [L, L, d]

    def forward(self, x):
        B, nh, N, d = x.shape
        L = int(N ** 0.5)

        pe = self._get_spectral_pe(L)  # [L, L, d]

        x = rearrange(x, 'b h (l1 l2) d -> b h l1 l2 d', l1=L)

        x = x + torch.einsum('b h m n d, m n d -> b h m n d',
                             x, pe) * (d ** -0.25)

        return rearrange(x, 'b h l1 l2 d -> b h (l1 l2) d')


class SpectralAttention(nn.Module):
    def __init__(self, peak_width=3., valley_width=3.):
        super().__init__()
        self.log_peak_width = nn.Parameter(torch.tensor(math.log(peak_width)))
        self.log_valley_width = nn.Parameter(torch.tensor(math.log(valley_width)))

    def forward(self, x):
        # x: (B, C, N)
        B, C, N = x.shape

        diff = x.diff(dim=-1)
        sign_change = diff.diff(dim=-1).sign()
        peaks = (sign_change < 0).nonzero(as_tuple=True)[-1] + 1
        valleys = (sign_change > 0).nonzero(as_tuple=True)[-1] + 1

        pos = torch.arange(N, device=x.device).float()

        def gaussian(centers, width):
            dist = pos.unsqueeze(0) - centers.unsqueeze(1)  # (n_centers, N)
            return torch.exp(-(dist / width.exp()) ** 2)

        attn = torch.zeros_like(x)
        if len(peaks) > 0:
            attn += gaussian(peaks.float(), self.log_peak_width).sum(dim=0)
        if len(valleys) > 0:
            attn += gaussian(valleys.float(), self.log_valley_width).sum(dim=0)

        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)

        return attn

class SpectralAttention_SpectralLinearAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.H, self.W = input_resolution

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.elu = nn.ELU()

        self.rope = SpectralRoPE(
            shape=(self.H, self.W),
            dim_head=self.head_dim,
            num_heads=num_heads
        )

        self.proj = nn.Linear(dim, dim)
        self.spec_att = SpectralAttention()

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.H, self.W

        # SpectralAttention
        x_spec = rearrange(x, 'b (h w) c -> b c (h w)', h=H)

        spec_weight = self.spec_att(x_spec)
        x = torch.einsum('b n c, b c n -> b n c', x, spec_weight)

        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads) for t in qkv]

        q, k = self.elu(q) + 1.0, self.elu(k) + 1.0
        q, k = self.rope(q), self.rope(k)

        scale = (self.head_dim * N) ** -0.5
        context = torch.einsum('b h n d, b h n e -> b h d e', q, k) * scale
        attn_out = torch.einsum('b h n d, b h d e -> b h n e', v, context) * scale

        out = rearrange(attn_out, 'b h n d -> b n (h d)')
        return self.proj(out)

class Classifier(nn.Module):

    def __init__(self, in_features=100, num_classes=NUM_CLASS):
        super().__init__()

        self.attn_pool1 = nn.Sequential(
            nn.Conv1d(in_features, 1, kernel_size=1),
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(100, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # (B, in_features, seq_len)

        features = x  # (B, hidden_dim, seq_len)

        attn_weights = self.attn_pool1(features)  # (B, 1, seq_len)

        return self.classifier1(attn_weights.squeeze(1))

class SpatialDiffAttention(nn.Module):

    def __init__(self, d_channels=CHANNEL, num_heads=5):
        super().__init__()
        assert d_channels % (2 * num_heads) == 0, "d_channels must be divisible by 2*num_heads"
        self.num_heads = num_heads
        self.head_dim = d_channels // (num_heads)  # 减半维度
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(d_channels, 2 * d_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(d_channels, 2 * d_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(d_channels, d_channels, kernel_size=1)  # 保持单路V

        self.lambda_init = nn.Parameter(torch.tensor(0.4))
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))

        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(d_channels, d_channels)
        self.conv3x3 = nn.Conv2d(d_channels, d_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(d_channels, 1, kernel_size=1)
    def forward(self, D):
        B, C, H, W = D.shape
        N = H * W

        x_h = self.pool_h(D).reshape(B, -1, H, 1)
        x_w = self.pool_w(D).reshape(B, -1, 1, W)
        hw = x_h.expand(-1, -1, -1, W) + x_w.expand(-1, -1, H, -1)
        spatial_enh = self.gn(D * torch.sigmoid(hw))

        Q = self.q_proj(spatial_enh).view(B, 2, self.num_heads, self.head_dim, N)
        K = self.k_proj(D).view(B, 2, self.num_heads, self.head_dim, N)
        V = self.v_proj(D).view(B, self.num_heads, self.head_dim, N)

        q1, q2 = Q[:, 0], Q[:, 1]  # [B, h, d, N]
        k1, k2 = K[:, 0], K[:, 1]  # [B, h, d, N]
        v = V

        attn1 = torch.einsum('bhdn,bhdm->bhnm', q1, k1) * self.scaling
        attn2 = torch.einsum('bhdn,bhdm->bhnm', q2, k2) * self.scaling

        lambda_1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum())
        lambda_2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum())
        lambda_full = (lambda_1 - lambda_2 ).clamp(0, 1) + self.lambda_init

        attn_weights = F.softmax(attn1 - lambda_full * attn2, dim=-1)

        out = torch.einsum('bhnm,bhdm->bhdn', attn_weights, v)  # 平均双路V
        out = out.reshape(B, C, H, W)

        x1 = self.agp(out).sigmoid()
        x2 = self.conv3x3(out)
        cross_enh = x2 * x1.expand_as(x2)

        attn_map = self.conv_out(cross_enh).sigmoid()
        return attn_map


class SpatialDiffEnhancement(nn.Module):

    def __init__(self, d_channels=CHANNEL, e_channels=CHANNEL*2):
        super(SpatialDiffEnhancement, self).__init__()

        self.SDA = SpatialDiffAttention(d_channels=d_channels, num_heads=8)

        self.e_calibrate = nn.Sequential(
            nn.Conv2d(e_channels, e_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(e_channels),
            nn.ReLU()
        )

    def forward(self, D, E):
        attn_map = self.SDA(D)  # (B,1,H,W)
        E_enhanced = E * attn_map.expand_as(E)
        E_enhanced = self.e_calibrate(E_enhanced) + E
        return E_enhanced


