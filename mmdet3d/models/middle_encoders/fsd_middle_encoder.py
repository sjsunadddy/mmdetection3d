import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.models.layers import FocalSparseConv, MemoryEfficientFocalSparseConv
from mmdet3d.registry import MODELS


@MODELS.register_module()
class FSDMiddleEncoder(BaseModule):
    """Focal Sparse 3D Detector Middle Encoder.

    Args:
        in_channels (int): Input channels.
        sparse_shape (tuple): Shape of sparse tensor.
        order (tuple): Order of operations.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=None):
        super(FSDMiddleEncoder, self).__init__(init_cfg=init_cfg)
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels

        self.conv = FocalSparseConv(
            in_channels,
            in_channels,
            norm_cfg=norm_cfg)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)


@MODELS.register_module()
class FSDv2MiddleEncoder(FSDMiddleEncoder):
    """Memory Efficient Focal Sparse 3D Detector Middle Encoder.

    This version uses memory-efficient operations to reduce memory usage.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 memory_efficient=True,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=None):
        super(FSDv2MiddleEncoder, self).__init__(
            in_channels=in_channels,
            sparse_shape=sparse_shape,
            order=order,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)
        self.memory_efficient = memory_efficient

        # Replace conv with memory-efficient version
        self.conv = MemoryEfficientFocalSparseConv(
            in_channels,
            in_channels,
            norm_cfg=norm_cfg)

    def forward(self, x):
        """Forward function with memory-efficient operations.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        with torch.cuda.amp.autocast():
            return self.conv(x) 
