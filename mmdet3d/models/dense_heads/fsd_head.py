import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.models.layers import FocalSparseConv, MemoryEfficientFocalSparseConv
from mmdet3d.registry import MODELS
from mmdet.models.dense_heads import AnchorHead


@MODELS.register_module()
class FSDHead(AnchorHead):
    """Focal Sparse 3D Detector Head.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Input channels.
        feat_channels (int): Feature channels.
        use_direction_classifier (bool): Whether to use direction classifier.
        anchor_generator (dict): Config of anchor generator.
        bbox_coder (dict): Config of bbox coder.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of bbox loss.
        loss_dir (dict): Config of direction classification loss.
        norm_cfg (dict): Config of norm layer.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 use_direction_classifier=True,
                 anchor_generator=dict(
                     type='Anchor3DRangeGenerator',
                     ranges=[[0, -40.0, -0.6, 70.4, 40.0, -0.6],
                            [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                            [0, -40.0, -1.78, 70.4, 40.0, -1.78]],
                     sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
                     rotations=[0, 1.57],
                     reshape_out=False),
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls=dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_dir=dict(
                     type='mmdet.CrossEntropyLoss', use_sigmoid=False,
                     loss_weight=0.2),
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=None,
                 **kwargs):
        super(FSDHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            init_cfg=init_cfg,
            **kwargs)
        self.use_direction_classifier = use_direction_classifier
        self.feat_channels = feat_channels

        # Build conv layers
        self.conv_cls = FocalSparseConv(
            in_channels,
            feat_channels,
            norm_cfg=norm_cfg)
        self.conv_reg = FocalSparseConv(
            in_channels,
            feat_channels,
            norm_cfg=norm_cfg)
        if self.use_direction_classifier:
            self.conv_dir_cls = FocalSparseConv(
                in_channels,
                feat_channels,
                norm_cfg=norm_cfg)

        # Build prediction layers
        self.reg_out = nn.Conv2d(feat_channels, self.box_code_size, 1)
        self.cls_out = nn.Conv2d(feat_channels, self.num_anchors * self.cls_out_channels, 1)
        if self.use_direction_classifier:
            self.dir_cls_out = nn.Conv2d(feat_channels, self.num_anchors * 2, 1)

    def forward(self, x):
        """Forward function.

        Args:
            x (list[torch.Tensor]): List of feature maps.

        Returns:
            tuple[list[torch.Tensor]]: List of predictions.
        """
        cls_score = []
        bbox_pred = []
        dir_cls_pred = []

        for i, feature in enumerate(x):
            cls_feat = self.conv_cls(feature)
            reg_feat = self.conv_reg(feature)

            cls_score.append(self.cls_out(cls_feat))
            bbox_pred.append(self.reg_out(reg_feat))

            if self.use_direction_classifier:
                dir_cls_feat = self.conv_dir_cls(feature)
                dir_cls_pred.append(self.dir_cls_out(dir_cls_feat))

        return cls_score, bbox_pred, dir_cls_pred


@MODELS.register_module()
class FSDv2Head(FSDHead):
    """Memory Efficient Focal Sparse 3D Detector Head.

    This version uses memory-efficient operations to reduce memory usage.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 use_direction_classifier=True,
                 memory_efficient=True,
                 anchor_generator=dict(
                     type='Anchor3DRangeGenerator',
                     ranges=[[0, -40.0, -0.6, 70.4, 40.0, -0.6],
                            [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                            [0, -40.0, -1.78, 70.4, 40.0, -1.78]],
                     sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
                     rotations=[0, 1.57],
                     reshape_out=False),
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls=dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_dir=dict(
                     type='mmdet.CrossEntropyLoss', use_sigmoid=False,
                     loss_weight=0.2),
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=None,
                 **kwargs):
        super(FSDv2Head, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            use_direction_classifier=use_direction_classifier,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.memory_efficient = memory_efficient

        # Replace conv layers with memory-efficient versions
        self.conv_cls = MemoryEfficientFocalSparseConv(
            in_channels,
            feat_channels,
            norm_cfg=norm_cfg)
        self.conv_reg = MemoryEfficientFocalSparseConv(
            in_channels,
            feat_channels,
            norm_cfg=norm_cfg)
        if self.use_direction_classifier:
            self.conv_dir_cls = MemoryEfficientFocalSparseConv(
                in_channels,
                feat_channels,
                norm_cfg=norm_cfg)

    def forward(self, x):
        """Forward function with memory-efficient operations.

        Args:
            x (list[torch.Tensor]): List of feature maps.

        Returns:
            tuple[list[torch.Tensor]]: List of predictions.
        """
        with torch.cuda.amp.autocast():
            return super().forward(x) 
