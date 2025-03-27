import torch
import torch.nn as nn

# from FeedForwardModule import FeedForwardModule
# from ConvolutionModule import ConvolutionModule
# import PositionEncoding


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
            nn.Conv1d(d_model, 2*d_model, kernel_size=1, stride=1, padding=0,bias=bias),
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
        

# class PositionEncoding(nn.Module):
#     ''' Absolute positional encoding.
#     '''
#     def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 1000) -> None:
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.pe = torch.zeros((1, max_len, d_model))
#         X = torch.arange(max_len, dtype=torch.float32).reshape(
#             -1,1) / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)

#         self.pe[:, :, 0::2] = torch.sin(X)
#         self.pe[:, :, 1::2] = torch.cos(X)

#     def forward(self, X) -> torch.Tensor:
#         ''' 
#         Args:
#             X (torch.Tensor): with shape of (batch, seqLence, d_model).
#         Return:
#             torch.Tensor with shape of (batch, seqLence, d_model).
#         '''
#         X = X + self.pe[:, X.shape[1], :].to(X.device) 
#         return self.dropout(X)
    
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

    def forward(self, num_patches, cls_token=False) -> torch.Tensor:
        ''' 
        Args:
            X (torch.Tensor): with shape of (batch, seqLence, d_model).
        Return:
            position embedding: torch.Tensor with shape of (1, seqLence, d_model).
        '''
        if cls_token:
            pos_emb = self.pe[:, :num_patches, :] 
            cls_token = torch.zeros((1, 1, pos_emb.shape[-1]))
            pos_emb = torch.cat([cls_token, pos_emb], dim=1)
        else:
            pos_emb = self.pe[:, :num_patches, :]

        return pos_emb


class ConformerLayer(nn.Module):
    ''' Confermer layer unit.
    Args: 
        d_model (int): model dimension.
        
    '''
    def __init__(self, d_model: int, 
                 ffn_dim_factor: int, 
                 num_attention_heads: int, 
                 depthwise_conv_kernel_size: int,
                 feedford_dropout: float = 0.0, 
                 pos_encode_dropout: float = 0.0, 
                 attention_dropout: float = 0.0, 
                 conv_dropout: float = 0.0, 
                 use_group_norm: bool = False, 
                 convolution_first: bool = False,
                ) -> None:
        super().__init__()

        self.ffn1 = FeedForwardModule(d_model, d_model*ffn_dim_factor, dropout=feedford_dropout)
        # self.pos_enccoding = PositionEncoding(d_model, dropout=pos_encode_dropout)
        self.self_atten_layernorm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_attention_heads, dropout=attention_dropout)
        self.attn_drop = nn.Dropout(attention_dropout)

        self.conv_module = ConvolutionModule(d_model, depthwise_conv_kernel_size,
                                            dropout=conv_dropout, bias=True, use_group_norm=use_group_norm)

        self.ffn2 = FeedForwardModule(d_model, d_model*ffn_dim_factor, dropout=feedford_dropout)
        self.final_layernorm = nn.LayerNorm(d_model)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        # input = input.transpose(0, 1)
        input = self.conv_module(input)
        # input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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

        # # Absolute Position ecoding 
        # x = self.pos_enccoding(x)

        # Multi-head self attention 
        residual = x
        x = self.self_atten_layernorm(x)
        # print(f'shape of x before atention: {x.shape}')
        x, _ = self.attn(query=x, key=x, value=x, need_weights=False)
        # print(f'shape of x after attention: {x.shape}')
        x = self.attn_drop(x)
        x = x + residual

        # convolution module 
        if not self.convolution_first:
            x = self._apply_convolution(x)

        # Feedforward Module
        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual
        
        x = self.final_layernorm(x)

        return x
        