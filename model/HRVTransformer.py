import torch
import torch.nn as nn
from torchsummary import summary
import logging

import timm.models.vision_transformer

from Classifier import FCN_head, MLP_glob_pool_head, MLP_CLS


logger = logging.getLogger(__name__)

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

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
            logger.debug(f'pos_emb shape: {pos_emb.shape}')
            logger.debug(f'cls_token shape: {cls_token.shape}')
            pos_emb = torch.cat([cls_token, pos_emb], dim=1)
            # pos_emb = torch.cat([torch.zeros((1, 1, pos_emb.shape[-1])), pos_emb], dim=1)
            logger.debug(f'pos_emb shape: {pos_emb.shape}')
        else:
            pos_emb = self.pe[:, :num_patches, :]

        return pos_emb


class HRVTransformer(nn.Module):
    def __init__(self, input_dim=1200, patch_size=80, num_classes=2, embed_dim=144, depth=2, 
                 num_heads=6, mlp_ratio=4., drop_rate=0., norm_layer=nn.LayerNorm, 
                 classfier_head='fcn', fcn_head_kernel_size=7):
        super(HRVTransformer, self).__init__()

        self.num_classes = num_classes
        self.drop_rate = drop_rate

        assert input_dim % patch_size == 0, 'input_dim must be divisible by patch_size'
        self.seq_length = int(input_dim // patch_size)
        self.patch_embed = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_length + 1, embed_dim), requires_grad=False)
        self.pos_encoding = PositionEncoding(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            timm.models.vision_transformer.Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                proj_drop=drop_rate, attn_drop=drop_rate, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        if classfier_head == 'mlp_glob_pool':
            self.head = MLP_glob_pool_head(input_dim=embed_dim, n_class=num_classes, dropout=drop_rate)
            self.cls_enable = False
        elif classfier_head == 'fcn':
            self.head = FCN_head(in_channels=self.seq_length, n_class=num_classes, kernel_size=fcn_head_kernel_size)
            self.cls_enable = False
        elif classfier_head == 'mlp_cls':
            self.head = MLP_CLS(d_model=embed_dim, n_class=num_classes, dropout=drop_rate)
            self.cls_enable = True

        self.initialize_weights()


    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)

        # fix the sin-cos pos embedding (cls_token=True produce seq_length+1 pos embedding)
        pos_emb = self.pos_encoding(self.seq_length, cls_token=True)     # set cls_token=True produce seq_length+1 patches (1, seq_length+1, d_model)
        self.pos_embed.data.copy_(pos_emb)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        
        B = x.shape[0]
        x = self.patch_embed(x) # (B, embed_dim, seq_length)
        x = x.permute(0, 2, 1)  # (B, seq_length, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if not self.cls_enable:   # not use cls token for classification
            x = x[:, 1:, :]       # (batch, seq_length, d_model)
        x = self.head(x)

        return x
    
def hrv_transformer(config):
    model_config = {
        'input_dim': config.input_dim, 'patch_size': config.patch_size,
        'num_classes': config.n_class, 'embed_dim': config.d_model,
        'depth': config.n_layer, 'num_heads': config.num_attention_heads,
        'mlp_ratio': config.ff_dim_factor, 'drop_rate': config.dropout,
        'classfier_head': config.classifier_head,
        'fcn_head_kernel_size': config.fcn_head_kernel_size,
    }
    return HRVTransformer(**model_config)


if __name__ == '__main__':

    model = HRVTransformer()
    print(model)
    x = torch.randn(1, 1, 1200)
    y = model(x)
    print(y.shape)

    summary(model, (1, 1200), device='cpu')