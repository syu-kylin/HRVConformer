import torch
import torch.nn as nn

from model.ConformerLayer import ConformerLayer, PositionEncoding
from Classifier import MLP_head, FCN_head, MLP_glob_pool_head

class ConformerNet(nn.Module):
    ''' Confermer Encoder module.
    '''

    def __init__(self, input_dim: int = 1200,
                 patch_size: int = 80,
                 d_model: int = 144, 
                 num_layers: int = 2,
                 ff_dim_factor: int = 4,
                 num_attention_heads: int = 6,
                 feedford_dropout: float = 0.0,
                 attention_dropout: float = 0.0, 
                 pos_encode_dropout: float = 0.0,
                 conv_dropout: float = 0.0, 
                 conv_kernel_size: int = 31,
                 classifier_head: str = 'fcn',
                 fcn_head_kernel_size: int = 3,
                 mlp_hid_dim: int = 200, 
                 mlp_dropout: float = 0.0,
                 n_class: int = 2, 
                ) -> None:
        super().__init__()

        assert input_dim % patch_size == 0, 'input_dim must be divisible by patch_size'
        self.seq_length = int(input_dim // patch_size)
        self.input_projection = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size, padding=0)

        # self.pos_enccoding = PositionEncoding(d_model, dropout=pos_encode_dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_length+1, d_model), requires_grad=False)
        self.pos_encoding = PositionEncoding(d_model)

        self.layers = nn.ModuleList([ConformerLayer(
            d_model=d_model,
            ffn_dim_factor=ff_dim_factor,
            num_attention_heads=num_attention_heads,
            depthwise_conv_kernel_size=conv_kernel_size,
            feedford_dropout=feedford_dropout,
            pos_encode_dropout=pos_encode_dropout,
            attention_dropout=attention_dropout,
            conv_dropout=conv_dropout,
            use_group_norm=False,
            convolution_first=False,
        ) for _ in range(num_layers)]) 

        if classifier_head == 'mlp':
            self.head = MLP_head(self.seq_length*d_model, mlp_hid_dim, n_class, mlp_dropout)
            self.cls_enable = False
        elif classifier_head == 'fcn':
            self.head = FCN_head(in_channels=self.seq_length, n_class=n_class, kernel_size=fcn_head_kernel_size)
            self.cls_enable = False
        elif classifier_head == 'mlp_glob_pool':
            self.head = MLP_glob_pool_head(d_model, n_class, mlp_dropout)
            self.cls_enable = True

        self.init_weights()

    def init_weights(self):
        '''Model weights initialization function.
        '''
        torch.nn.init.normal_(self.cls_token, std=.02)

        # fix the sin-cos pos embedding (cls_token=True produce seq_length+1 pos embedding)
        pos_emb = self.pos_encoding(self.seq_length, cls_token=True)     # (1, seq_length+1, d_model)
        self.pos_emb.data.copy_(pos_emb)

        # intitialize the weights of the model layers
        self.apply(lambda x: init_weights(x))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ''' 
        Args:
            inputs (torch.Tensor): with shape of (batch, seqLence, input_dim)
        Return:
            outputs (torch.Tensor): with shape of (batch, seqLence, d_model)
        '''
        # 1). Input projection and position encoding
        outputs = self.input_projection(inputs)
        outputs = outputs.permute(0, 2, 1)    # (batch, d_model, seq_length)

        # print('shape of input projection:', outputs.shape)
        # outputs = self.pos_enccoding(outputs.permute(0, 2, 1))

        cls_token = self.cls_token.expand(outputs.shape[0], -1, -1)    # (batch, 1, d_model)
        outputs = torch.cat([cls_token, outputs], dim=1)               # (batch, seq_length+1, d_model)
        outputs = outputs + self.pos_emb                               # (batch, seq_length+1, d_model)

        # 2). Conformer blocks
        for layer in self.layers:
            outputs = layer(outputs)
        # print(f'shap of confermer layer: {outputs.shape}')

        # 3). Classifier head
        if not self.cls_enable:            # not use cls token for classification
            outputs = outputs[:, 1:, :]    # (batch, seq_length, d_model)
        outputs = self.head(outputs)

        return outputs
    
    
def init_weights(module):
    '''Model weights initialization function.
    Args:
        m (nn.modules): model layers.
        init_way (string): layer initialization method (can be:
        'xavier_uniform', 'kaiming_uniform', 'kaiming_normal', 'xavier_uniform_relu',
        'xavier_normal').
    '''

    if isinstance(module, nn.Linear):
        # print(f'Reset parameters of Linear layer: {module}')
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
            # print(f'Reset parameters of Linear bias: {module}')

    elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
        # print(f'Reset parameters of Conv layer: {module}')
        # NOTE conv was left to pytorch default in my original init
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
            # print(f'Reset parameters of conv bias: {module}')
            
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        # print(f'Reset parameters of Norm layer: {module}')
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
       
            
def confermer_net(config):
    model_config = {
        'input_dim': config.input_dim, 'patch_size': config.patch_size,
        'd_model': config.d_model, 'num_layers': config.n_layer,
        'ff_dim_factor': config.ff_dim_factor, 'num_attention_heads': config.num_attention_heads,
        'feedford_dropout': config.dropout, 'attention_dropout': config.dropout,
        'pos_encode_dropout': config.dropout, 'conv_dropout': config.dropout, 
        'conv_kernel_size': config.conv_kernel_size, 'classifier_head': config.classifier_head, 
        'fcn_head_kernel_size': config.fcn_head_kernel_size, 'mlp_hid_dim': config.mlp_hid_dim,
        'mlp_dropout': config.dropout, 'n_class': config.n_class,
    }
    return ConformerNet(**model_config)


if __name__ == "__main__":
    
    confermerNet = ConformerNet()
    # x = torch.randn(2, 15, 80)
    x = torch.randn(2, 1, 1200)
    y = confermerNet(x)
    print(y)
