import torch
import torch.nn as nn


class MLP_head(nn.Module):
    ''' Classifier for MLP head.
    '''
    def __init__(self, input_dim: int,
                hid_dim: int,
                n_class: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, n_class)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ''' 
        Args:
            input (torch.Tensor): with shape of (batch, seqlence*d_model)
        Return:
            output (torch.Tensor): with shape of (batch, n_class)
        '''
        return self.mlp(inputs)
    
class MLP_glob_pool_head(nn.Module):
    ''' Classifier for MLP head with global pooling.
    '''
    def __init__(self, input_dim: int,
                n_class: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, n_class),
            nn.Dropout(dropout)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ''' 
        Args:
            input (torch.Tensor): with shape of (batch, seqlence, d_model)
        Return:
            output (torch.Tensor): with shape of (batch, n_class)
        '''
        # global pooling
        inputs = inputs.mean(dim=1)

        return self.mlp(inputs)

class FCN_head(nn.Module):
    ''' Classifier for FCN head.
    '''
    def __init__(self, in_channels: int, n_class: int, kernel_size: int) -> None:
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, n_class, kernel_size=kernel_size, stride=2),nn.ReLU(),
            # nn.BatchNorm1d(n_class),
            nn.Conv1d(n_class, n_class, kernel_size=3, stride=2),nn.ReLU(),
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, inputs):
        ''' 
        Args:
            input (torch.Tensor): with shape of (batch, seqLence, d_model)
        Return:
            output (torch.Tensor): with shape of (batch, n_class)
        '''
        return self.sequential(inputs)
    
class MLP_CLS(nn.Module):
    ''' Classifier for MLP head (input is one patch of class token).
    '''
    def __init__(self, d_model: int,
                n_class: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            # nn.LayerNorm(d_model),
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, n_class),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ''' 
        Args:
            input (torch.Tensor): with shape of (batch, seqLength, d_model)
        Return:
            output (torch.Tensor): with shape of (batch, n_class)
        '''
        cls_token = inputs[:, 0, :]             # (batch, d_model)
        return self.mlp(cls_token)
