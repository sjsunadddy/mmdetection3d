import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.models.layers import FocalSparseConv, MemoryEfficientFocalSparseConv
from mmdet3d.registry import MODELS


@MODELS.register_module()
class FSDv2(BaseModule):
    """Focal Sparse 3D Detector v2 backbone.

    Args:
        in_channels (int): Input channels.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of layers in each stage.
        out_channels (list[int]): Output channels of each stage.
        focal_sparse_conv (bool): Whether to use focal sparse convolution.
        feature_propagation (bool): Whether to use feature propagation.
        memory_efficient (bool): Whether to use memory-efficient operations.
    """

    def __init__(self,
                 in_channels,
                 layer_nums,
                 layer_strides,
                 out_channels,
                 focal_sparse_conv=True,
                 feature_propagation=True,
                 memory_efficient=True,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=None):
        super(FSDv2, self).__init__(init_cfg=init_cfg)
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.out_channels = out_channels
        self.focal_sparse_conv = focal_sparse_conv
        self.feature_propagation = feature_propagation
        self.memory_efficient = memory_efficient

        conv_layer = MemoryEfficientFocalSparseConv if memory_efficient else FocalSparseConv

        self.sparse_blocks = nn.ModuleList()
        for i in range(len(layer_nums)):
            layers = []
            for j in range(layer_nums[i]):
                if j == 0:
                    layers.append(
                        conv_layer(
                            in_channels if i == 0 else out_channels[i - 1],
                            out_channels[i],
                            stride=layer_strides[i],
                            padding=1,
                            norm_cfg=norm_cfg,
                            focal_sparse_conv=focal_sparse_conv))
                else:
                    layers.append(
                        conv_layer(
                            out_channels[i],
                            out_channels[i],
                            stride=1,
                            padding=1,
                            norm_cfg=norm_cfg,
                            focal_sparse_conv=focal_sparse_conv))
            self.sparse_blocks.append(nn.Sequential(*layers))

        if feature_propagation:
            self.feature_propagations = nn.ModuleList()
            for i in range(len(layer_nums) - 1):
                self.feature_propagations.append(
                    ConvModule(
                        out_channels[i + 1],
                        out_channels[i],
                        1,
                        stride=1,
                        padding=0,
                        norm_cfg=norm_cfg,
                        act_cfg=None))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            list[torch.Tensor]: List of feature maps.
        """
        outs = []
        for i in range(len(self.layer_nums)):
            x = self.sparse_blocks[i](x)
            outs.append(x)

        if self.feature_propagation:
            for i in range(len(self.layer_nums) - 1, 0, -1):
                if self.memory_efficient:
                    # Memory-efficient feature propagation
                    feat = self.feature_propagations[i - 1](outs[i])
                    feat = F.interpolate(feat, size=outs[i - 1].shape[-3:])
                    outs[i - 1] = outs[i - 1] + feat
                    del feat
                else:
                    outs[i - 1] = outs[i - 1] + self.feature_propagations[i - 1](
                        F.interpolate(outs[i], size=outs[i - 1].shape[-3:]))

        return outs 
