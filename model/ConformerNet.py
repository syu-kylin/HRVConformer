import torch
import torch.nn as nn
import logging
from model.ConformerBlock import ConformerBlock, PositionEncoding
from model.Classifier import MLP_head, FCN_head, MLP_glob_pool_head, MLP_CLS


logger = logging.getLogger('project_log')
class ConformerNet(nn.Module):
    ''' Confermer Encoder module.
    '''

    def __init__(self, input_dim: int = 1200,
                 patch_size: int = 100,
                 d_model: int = 144, 
                 num_layers: int = 3,
                 ff_dim_factor: int = 4,
                 num_attention_heads: int = 8,
                 attention_type: str = 'RelativePositionBias',
                 fixed_position_embedding: bool = False,
                 feedford_dropout: float = 0.0,
                 attention_dropout: float = 0.0, 
                 pos_encode_dropout: float = 0.0,
                 conv_dropout: float = 0.0, 
                 conv_kernel_size: int = 11,
                 classifier_head: str = 'fcn',
                 fcn_head_kernel_size: int = 11,
                 mlp_hid_dim: int = 200, 
                 mlp_dropout: float = 0.0,
                 n_class: int = 2, 
                ) -> None:
        super().__init__()

        assert input_dim % patch_size == 0, 'input_dim must be divisible by patch_size'
        self.seq_length = int(input_dim // patch_size)
        self.input_projection = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size, padding=0)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, d_model), requires_grad=False)
        # self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_length+1, d_model), requires_grad=False)
        self.pos_encoding = PositionEncoding(d_model)
        self.fixed_position_embedding = fixed_position_embedding

        self.layers = nn.ModuleList([ConformerBlock(
            d_model=d_model,
            ffn_dim_factor=ff_dim_factor,
            num_attention_heads=num_attention_heads,
            depthwise_conv_kernel_size=conv_kernel_size,
            feedford_dropout=feedford_dropout,
            pos_encode_dropout=pos_encode_dropout,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            conv_dropout=conv_dropout,
            use_group_norm=False,
            convolution_first=False,
            max_distance=self.seq_length+1,
        ) for _ in range(num_layers)]) 

        if classifier_head == 'mlp':
            self.head = MLP_head(self.seq_length*d_model, mlp_hid_dim, n_class, mlp_dropout)
            self.cls_enable = False
        elif classifier_head == 'mlp_cls':
            self.head = MLP_CLS(d_model, n_class, mlp_dropout)
            self.cls_enable = True
        elif classifier_head == 'fcn':
            self.head = FCN_head(in_channels=self.seq_length, n_class=n_class, kernel_size=fcn_head_kernel_size)
            self.cls_enable = False
        elif classifier_head == 'mlp_glob_pool':
            self.head = MLP_glob_pool_head(d_model, n_class, mlp_dropout)
            self.cls_enable = True
        
        logger.info(f"ConformerLayer attention moudle: {self.layers[0].attn.__class__.__name__}")
        logger.info(f'ConformerNet classifier head: {self.head.__class__.__name__}')
        self.init_weights()

    def init_weights(self):
        '''
        Model weights initialization function.
        '''
        torch.nn.init.normal_(self.cls_token, std=.02)

        torch.nn.init.normal_(self.pos_emb, std=.02)
        # fix the sin-cos pos embedding (cls_token=True produce seq_length+1 pos embedding)
        pos_emb = self.pos_encoding(self.seq_length)     # (1, seq_length, d_model)
        self.pos_emb.data.copy_(pos_emb)

        # intitialize the weights of the model layers
        self.apply(lambda x: init_weights(x))

    def forward(self, inputs: torch.Tensor, return_attention=False, avg_attn_heads=True) -> torch.Tensor:
        ''' 
        Args:
            inputs (torch.Tensor): with shape of (batch, seqLence, input_dim)
        Return:
            outputs (torch.Tensor): with shape of (batch, seqLence, d_model)
        '''

        # 1). Input projection and position encoding
        all_attn = []
        outputs = self.input_projection(inputs)
        outputs = outputs.permute(0, 2, 1)    # (batch, d_model, seq_length)

        cls_token = self.cls_token.expand(outputs.shape[0], -1, -1)    # (batch, 1, d_model)
        if self.fixed_position_embedding:
            outputs = outputs + self.pos_emb                               # (batch, seq_length, d_model)
        outputs = torch.cat([cls_token, outputs], dim=1)               # (batch, seq_length+1, d_model)

        # 2). Conformer blocks
        for layer in self.layers:
            if return_attention:
                outputs, attn = layer(outputs, return_attention=True, avg_attn_heads=avg_attn_heads) 
                all_attn.append(attn)       # each is (B, H, L, L)
            else:
                outputs = layer(outputs)

        # 3). Classifier head
        if not self.cls_enable:            # not use cls token for classification
            outputs = outputs[:, 1:, :]    # (batch, seq_length, d_model)
        outputs = self.head(outputs)

        if return_attention:
            return outputs, all_attn   # List of (B, H, L, L)
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
       
            
def hrvconformer(config):
    model_config = {
        'input_dim': config.input_dim, 'patch_size': config.patch_size,
        'd_model': config.d_model, 'num_layers': config.n_layer, 'attention_type':config.attention_type,
        'fixed_position_embedding': config.fixed_position_embedding,
        'ff_dim_factor': config.ff_dim_factor, 'num_attention_heads': config.num_attention_heads,
        'feedford_dropout': config.dropout, 'attention_dropout': config.dropout,
        'pos_encode_dropout': config.dropout, 'conv_dropout': config.dropout, 
        'conv_kernel_size': config.conv_kernel_size, 'classifier_head': config.classifier_head, 
        'fcn_head_kernel_size': config.fcn_head_kernel_size, 'mlp_hid_dim': config.mlp_hid_dim,
        'mlp_dropout': config.dropout, 'n_class': config.n_class,
    }
    return ConformerNet(**model_config)


# Backward-compatible alias for earlier experiment scripts.
confermer_net = hrvconformer


if __name__ == "__main__":
    
    confermerNet = ConformerNet(patch_size=100)
    # x = torch.randn(2, 15, 80)
    x = torch.randn(2, 1, 1200)
    y = confermerNet(x)
    print(y)
