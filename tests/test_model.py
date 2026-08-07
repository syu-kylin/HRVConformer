import pytest
import torch

from model.ConformerNet import ConformerNet


def test_paper_model_forward_shape():
    model = ConformerNet(
        input_dim=1200,
        patch_size=100,
        d_model=144,
        num_layers=3,
        num_attention_heads=8,
        conv_kernel_size=11,
        fcn_head_kernel_size=11,
    ).eval()

    with torch.no_grad():
        output = model(torch.randn(2, 1, 1200))

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()


def test_attention_output_dimensions():
    model = ConformerNet(
        input_dim=1200,
        patch_size=100,
        d_model=32,
        num_layers=2,
        num_attention_heads=4,
        conv_kernel_size=11,
        fcn_head_kernel_size=3,
    ).eval()

    with torch.no_grad():
        output, attention = model(
            torch.randn(1, 1, 1200), return_attention=True, avg_attn_heads=False
        )

    assert output.shape == (1, 2)
    assert len(attention) == 2
    assert attention[0].shape == (1, 4, 13, 13)


def test_input_must_be_divisible_by_patch_size():
    with pytest.raises(AssertionError, match='input_dim must be divisible'):
        ConformerNet(input_dim=1200, patch_size=128)
