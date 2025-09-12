import torch
from torch.jit import Final
import torch.nn as nn
from typing import Type
import logging


logger = logging.getLogger('project_log')

class FeedForwardModule(nn.Module):
    '''Confermer FeedForward Layer.
    Args:
        d_model (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    '''
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            input (torch.Tensor): with shape (batch, seqLength, d_model).

        Return:
            torch.Tensor: output, with shape (batch, seqLength, d_model).
        '''
        return self.sequential(input)
    

class ConvolutionModule(nn.Module):
    '''Confermer convolution module.
    Args: 
        input_dim (int): input dimension.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): whether to add bias term to each convolution layer. (DefaultL False)
        use_group_norm (bool, optional): use group normal (instance normal) or batchnorm. (Default False (batchnorm))
    '''
    def __init__(self, d_model: int, depthwise_kernel_size: int, dropout: float = 0.0, 
                bias: bool = False, use_group_norm: bool = False) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 !=0:
            raise ValueError('depthwise_kernel_size must be odd to achieve same padding.')

        self.layernorm = nn.LayerNorm(d_model)
        self.sequential = nn.Sequential(
            # Pointwise Convolution
            nn.Conv1d(d_model, 2*d_model, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GLU(dim=1),
            # Depthwise Convolution
            nn.Conv1d(d_model, d_model, depthwise_kernel_size, stride=1, 
                     padding=(depthwise_kernel_size-1)//2, groups=d_model, bias=bias),
            torch.nn.GroupNorm(num_groups=1, num_channels=d_model)
            if use_group_norm
            else torch.nn.BatchNorm1d(d_model),
            nn.SiLU(),
            # Pointwise Convolution
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            input (torch.Tensor): with shape of (batch, seqLence, d_model).
        Returns:
            torch.Tensor: with shape of (batch, seqLence, d_model).
        '''
        x = self.layernorm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)

        return x.transpose(1, 2)
        

class PositionEncoding(nn.Module):
    ''' Absolute positional encoding.
    '''
    def __init__(self, d_model: int, max_len: int = 1000) -> None:
        super().__init__()
        self.pe = torch.zeros((1, max_len, d_model))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1,1) / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)

        self.pe[:, :, 0::2] = torch.sin(X)
        self.pe[:, :, 1::2] = torch.cos(X)

    def forward(self, num_patches) -> torch.Tensor:
        ''' 
        Args:
            X (torch.Tensor): with shape of (batch, seqLence, d_model).
        Return:
            position embedding: torch.Tensor with shape of (1, seqLence, d_model).
        '''
        # if cls_token:
        #     pos_emb = self.pe[:, :num_patches, :] 
        #     cls_token = torch.zeros((1, 1, pos_emb.shape[-1]))
        #     pos_emb = torch.cat([cls_token, pos_emb], dim=1)
        # else:
        pos_emb = self.pe[:, :num_patches, :]

        return pos_emb

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_distance=32):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.rel_bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, seq_len):
        # Create relative distance matrix: shape (T, T)
        context_position = torch.arange(seq_len)[:, None]    # shape (T, 1)
        memory_position = torch.arange(seq_len)[None, :]     # shape (1, T)
        relative_position = memory_position - context_position  # shape (T, T)
        relative_position = relative_position.clamp(-self.max_distance, self.max_distance)  # clamp to [-max_distance, max_distance]
        relative_position += self.max_distance  # shift to positive indices
        relative_position = relative_position.to(self.rel_bias.weight.device)

        # Get relative bias: shape (T, T, H)
        rel_bias = self.rel_bias(relative_position)  # (T, T, H)
        rel_bias = rel_bias.permute(2, 0, 1)  # (H, T, T)
        return rel_bias
    
class MultiheadAttentionWithRelBias(nn.Module):
    ''' Multi-head attention with relative position bias. (T5 style relative position bias)
    This module implements multi-head attention with relative position bias.
    Args:
        dim (int): input dimension.
        num_heads (int): number of attention heads.
        qkv_bias (bool): whether to add bias to qkv linear layer. (Default: False)
        qk_norm (bool): whether to apply layer norm to q and k. (Default: False)
        proj_bias (bool): whether to add bias to projection linear layer. (Default: True)
        attn_drop (float): dropout probability for attention weights. (Default: 0.0)
        proj_drop (float): dropout probability for projection layer. (Default: 0.0)
        norm_layer (Type[nn.Module]): normalization layer type. (Default: nn.LayerNorm)
        max_distance (int): maximum distance for relative position bias. (Default: 32)
    '''
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            max_distance: int = 32,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        # self.proj_drop = nn.Dropout(proj_drop)   # dropout have been added attention module in ConformerBlock

        # Relative position embedding bias
        self.max_distance = max_distance
        self.rel_bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, x: torch.Tensor, average_attn_weights=True) -> torch.Tensor:
        B, L, C = x.shape
        device = x.device
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)      # q, k, v: (B, H, L, D)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn_logits  = q @ k.transpose(-2, -1)

        # Compute relative positions (L, L)
        relative_position = torch.arange(L, device=device).unsqueeze(1) - torch.arange(L, device=device).unsqueeze(0)
        relative_position = relative_position.clamp(-self.max_distance, self.max_distance) + self.max_distance  # shift to [0, 2*max_dist]

        # Get relative bias: shape (L, L, H)
        rel_bias = self.rel_bias(relative_position)  # (L, L, H)
        rel_bias = rel_bias.permute(2, 0, 1)  # (H, L, L)
        rel_bias_expanded = rel_bias.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, L, L)
        attn_logits  = attn_logits  + rel_bias_expanded   # Add relative bias

        attn_weights = attn_logits.softmax(dim=-1)  # (B, H, L, L)
        attn = self.attn_drop(attn_weights)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)   # Average over heads
        return x, attn_weights
    
    
class ConformerBlock(nn.Module):
    ''' Confermer Block unit.
    Args: 
        d_model (int): model dimension.
        ffn_dim_factor (int): factor to scale the hidden dimension of feedforward module.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        feedford_dropout (float, optional): dropout probability for feedforward module. (Default: 0.0)
        pos_encode_dropout (float, optional): dropout probability for position encoding. (Default: 0.0)
        attention_type (str, optional): type of attention module, 'RelativePositionBias' or 'Standard'. (Default: 'RelativePositionBias')
        attention_dropout (float, optional): dropout probability for attention module. (Default: 0.0)
        conv_dropout (float, optional): dropout probability for convolution module. (Default: 0.0)
        use_group_norm (bool, optional): whether to use group normalization in convolution module. (Default: False)
        convolution_first (bool, optional): whether to apply convolution module before attention module. (Default: False)
        max_distance (int, optional): maximum distance for relative position bias. (Default: 32)
    
    '''
    def __init__(self, d_model: int, 
                 ffn_dim_factor: int, 
                 num_attention_heads: int, 
                 depthwise_conv_kernel_size: int,
                 feedford_dropout: float = 0.0, 
                 pos_encode_dropout: float = 0.0, 
                 attention_type: str = 'RelativePositionBias', 
                 attention_dropout: float = 0.0, 
                 conv_dropout: float = 0.0, 
                 use_group_norm: bool = False, 
                 convolution_first: bool = False,
                 max_distance: int = 32,
                ) -> None:
        super().__init__()


        self.ffn1 = FeedForwardModule(d_model, d_model*ffn_dim_factor, dropout=feedford_dropout)
        # self.pos_enccoding = PositionEncoding(d_model, dropout=pos_encode_dropout)
        self.self_atten_layernorm = nn.LayerNorm(d_model)

        # relative position bias with multi-head attention
        if attention_type == 'RelativePositionBias':
            self.attn = MultiheadAttentionWithRelBias(d_model, num_attention_heads, attn_drop=attention_dropout, max_distance=max_distance)
        # Standard Multi-head self attention
        else:
            self.attn = nn.MultiheadAttention(d_model, num_attention_heads, dropout=attention_dropout, batch_first=True)
        self.attn_drop = nn.Dropout(attention_dropout)

        self.conv_module = ConvolutionModule(d_model, depthwise_conv_kernel_size,
                                            dropout=conv_dropout, bias=True, use_group_norm=use_group_norm)

        self.ffn2 = FeedForwardModule(d_model, d_model*ffn_dim_factor, dropout=feedford_dropout)
        self.final_layernorm = nn.LayerNorm(d_model)
        self.convolution_first = convolution_first

        self.relAttn = True if isinstance(self.attn, MultiheadAttentionWithRelBias) else False

        # logger.info(f"Attention layer used in ConformerLayer: {self.attn.__class__.__name__}")

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = self.conv_module(input)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor, return_attention=False, avg_attn_heads=True) -> torch.Tensor:
        ''' 
        Args:
            input (torch.Tensor): with shape of (batch, seqLence, d_model)
        Returns:
            torch.Tensor: with shape of (batch, seqLence, d_model)
        '''
        # FeedForward Module
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        # Convolutiom Module if convlution first
        if self.convolution_first:
            x = self._apply_convolution(x)

        # Multi-head self attention 
        residual = x
        x = self.self_atten_layernorm(x)

        if self.relAttn:
            x, attn_weights = self.attn(x, average_attn_weights=avg_attn_heads)   # relative position bias MHSA
        else:
            x, attn_weights = self.attn(query=x, key=x, value=x, average_attn_weights=avg_attn_heads)  # Standard MHSA
        x = self.attn_drop(x)
        x = x + residual

        # convolution module 
        if not self.convolution_first:
            x = self._apply_convolution(x)

        # Feedforward Module
        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual
        # x = x + residual
        
        x = self.final_layernorm(x)

        if return_attention:
            return x, attn_weights
        return x
        

if __name__ == '__main__':

    pass