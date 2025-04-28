import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.models.layers import FocalSparseConv, MemoryEfficientFocalSparseConv
from mmdet3d.registry import MODELS


@MODELS.register_module()
class FSDNeck(BaseModule):
    """Focal Sparse 3D Detector Neck.

    Args:
        in_channels (list): Input channels for each level.
        out_channels (list): Output channels for each level.
        num_outs (int): Number of output levels.
        start_level (int): Start level of neck.
        end_level (int): End level of neck.
        add_extra_convs (bool): Whether to add extra convs.
        extra_convs_on_inputs (bool): Whether to add extra convs on inputs.
        relu_before_extra_convs (bool): Whether to add relu before extra convs.
        no_norm_on_lateral (bool): Whether to add norm on lateral.
        norm_cfg (dict): Config for norm layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=None):
        super(FSDNeck, self).__init__(init_cfg=init_cfg)
        assert len(in_channels) == len(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = dict(mode='nearest')

        if end_level == -1:
            self.backbone_end_level = len(in_channels)
            assert num_outs >= len(in_channels) - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            if extra_convs_on_inputs:
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = FocalSparseConv(
                in_channels[i],
                out_channels[i],
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None)
            fpn_conv = FocalSparseConv(
                out_channels[i],
                out_channels[i],
                norm_cfg=norm_cfg)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels[-1]
                extra_fpn_conv = FocalSparseConv(
                    in_channels,
                    out_channels[self.backbone_end_level - 1],
                    norm_cfg=norm_cfg)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (list[torch.Tensor]): Input feature maps.

        Returns:
            list[torch.Tensor]: Output feature maps.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape[2:], **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            if self.add_extra_convs == 'on_input':
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
            elif self.add_extra_convs == 'on_lateral':
                outs.append(self.fpn_convs[used_backbone_levels](laterals[-1]))
            elif self.add_extra_convs == 'on_output':
                outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
            for i in range(used_backbone_levels + 1, self.num_outs):
                if self.relu_before_extra_convs:
                    outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                else:
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


@MODELS.register_module()
class FSDv2Neck(FSDNeck):
    """Memory Efficient Focal Sparse 3D Detector Neck.

    This version uses memory-efficient operations to reduce memory usage.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 memory_efficient=True,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=None):
        super(FSDv2Neck, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            extra_convs_on_inputs=extra_convs_on_inputs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)
        self.memory_efficient = memory_efficient

        # Replace convs with memory-efficient versions
        self.lateral_convs = nn.ModuleList([
            MemoryEfficientFocalSparseConv(
                in_channels[i],
                out_channels[i],
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None)
            for i in range(self.start_level, self.backbone_end_level)
        ])
        self.fpn_convs = nn.ModuleList([
            MemoryEfficientFocalSparseConv(
                out_channels[i],
                out_channels[i],
                norm_cfg=norm_cfg)
            for i in range(self.start_level, self.backbone_end_level)
        ])

        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels[-1]
                extra_fpn_conv = MemoryEfficientFocalSparseConv(
                    in_channels,
                    out_channels[self.backbone_end_level - 1],
                    norm_cfg=norm_cfg)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        """Forward function with memory-efficient operations.

        Args:
            inputs (list[torch.Tensor]): Input feature maps.

        Returns:
            list[torch.Tensor]: Output feature maps.
        """
        with torch.cuda.amp.autocast():
            return super().forward(inputs) 
