import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


class FocalSparseConv(BaseModule):
    """Focal Sparse Convolution layer.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size of the convolution.
        stride (int): Stride of the convolution.
        padding (int): Padding of the convolution.
        dilation (int): Dilation of the convolution.
        groups (int): Groups of the convolution.
        bias (bool): Whether to use bias.
        norm_cfg (dict): Config of normalization layer.
        focal_sparse_conv (bool): Whether to use focal sparse convolution.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 focal_sparse_conv=True):
        super(FocalSparseConv, self).__init__()
        self.focal_sparse_conv = focal_sparse_conv
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.focal_sparse_conv:
            # Apply focal sparse convolution
            # 1. Compute attention weights
            attention = self._compute_attention(x)
            # 2. Apply attention to input
            x = x * attention
        return self.conv(x)

    def _compute_attention(self, x):
        """Compute attention weights for focal sparse convolution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Attention weights.
        """
        # Compute local variance as attention
        local_mean = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool3d((x - local_mean)**2, kernel_size=3, stride=1, padding=1)
        attention = torch.exp(-local_var)
        return attention


class MemoryEfficientFocalSparseConv(FocalSparseConv):
    """Memory Efficient Focal Sparse Convolution layer.

    This version uses memory-efficient operations to reduce memory usage.
    """

    def forward(self, x):
        """Forward function with memory-efficient operations.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.focal_sparse_conv:
            # Memory-efficient attention computation
            with torch.cuda.amp.autocast():
                attention = self._compute_attention(x)
                x = x * attention
                del attention
        return self.conv(x)

    def _compute_attention(self, x):
        """Compute attention weights with memory-efficient operations.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Attention weights.
        """
        # Compute local statistics in chunks to save memory
        local_mean = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        diff = x - local_mean
        del local_mean
        local_var = F.avg_pool3d(diff**2, kernel_size=3, stride=1, padding=1)
        del diff
        attention = torch.exp(-local_var)
        del local_var
        return attention 
