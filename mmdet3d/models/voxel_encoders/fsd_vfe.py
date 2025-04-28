import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.models.layers import FocalSparseConv, MemoryEfficientFocalSparseConv
from mmdet3d.registry import MODELS


@MODELS.register_module()
class FSDVFE(BaseModule):
    """Focal Sparse 3D Detector Voxel Feature Encoder.

    Args:
        in_channels (int): Input channels.
        feat_channels (list[int]): Feature channels.
        with_distance (bool): Whether to use distance feature.
        voxel_size (tuple): Size of voxels.
        with_cluster_center (bool): Whether to use cluster center.
        with_voxel_center (bool): Whether to use voxel center.
        point_cloud_range (tuple): Range of point cloud.
        focal_sparse_conv (bool): Whether to use focal sparse convolution.
        fusion_layer (dict): Config of fusion layer.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 0.2),
                 with_cluster_center=True,
                 with_voxel_center=True,
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 focal_sparse_conv=True,
                 fusion_layer=None,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=None):
        super(FSDVFE, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.with_distance = with_distance
        self.voxel_size = voxel_size
        self.with_cluster_center = with_cluster_center
        self.with_voxel_center = with_voxel_center
        self.point_cloud_range = point_cloud_range
        self.focal_sparse_conv = focal_sparse_conv

        # Feature layers
        self.feat_layers = nn.ModuleList()
        for i in range(len(feat_channels)):
            self.feat_layers.append(
                FocalSparseConv(
                    in_channels if i == 0 else feat_channels[i - 1],
                    feat_channels[i],
                    focal_sparse_conv=focal_sparse_conv,
                    norm_cfg=norm_cfg))

        # Fusion layer
        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)
        else:
            self.fusion_layer = None

    def forward(self, voxel_features, voxel_coords, batch_size):
        """Forward function.

        Args:
            voxel_features (torch.Tensor): Voxel features.
            voxel_coords (torch.Tensor): Voxel coordinates.
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Output features.
        """
        # Process features through layers
        for feat_layer in self.feat_layers:
            voxel_features = feat_layer(voxel_features)

        # Apply fusion layer if specified
        if self.fusion_layer is not None:
            voxel_features = self.fusion_layer(voxel_features)

        return voxel_features


@MODELS.register_module()
class FSDv2VFE(FSDVFE):
    """Memory Efficient Focal Sparse 3D Detector Voxel Feature Encoder.

    This version uses memory-efficient operations to reduce memory usage.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 0.2),
                 with_cluster_center=True,
                 with_voxel_center=True,
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 focal_sparse_conv=True,
                 memory_efficient=True,
                 fusion_layer=None,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=None):
        super(FSDv2VFE, self).__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            with_distance=with_distance,
            voxel_size=voxel_size,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            point_cloud_range=point_cloud_range,
            focal_sparse_conv=focal_sparse_conv,
            fusion_layer=fusion_layer,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)
        self.memory_efficient = memory_efficient

        # Replace feature layers with memory-efficient versions
        self.feat_layers = nn.ModuleList()
        for i in range(len(feat_channels)):
            self.feat_layers.append(
                MemoryEfficientFocalSparseConv(
                    in_channels if i == 0 else feat_channels[i - 1],
                    feat_channels[i],
                    focal_sparse_conv=focal_sparse_conv,
                    norm_cfg=norm_cfg))

    def forward(self, voxel_features, voxel_coords, batch_size):
        """Forward function with memory-efficient operations.

        Args:
            voxel_features (torch.Tensor): Voxel features.
            voxel_coords (torch.Tensor): Voxel coordinates.
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Output features.
        """
        # Process features through layers with memory efficiency
        with torch.cuda.amp.autocast():
            for feat_layer in self.feat_layers:
                voxel_features = feat_layer(voxel_features)

            # Apply fusion layer if specified
            if self.fusion_layer is not None:
                voxel_features = self.fusion_layer(voxel_features)

        return voxel_features 
