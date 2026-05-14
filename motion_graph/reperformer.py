import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from utils import GaussianModel
from utils.data_loader import Database
from utils.utils_gaussians.calc_utils import (
    batch_qvec2rotmat_torch,
    batch_rotmat2qvec_torch,
    build_rotation,
    norm_quaternion,
    quaternion_inverse,
    quaternion_multiply,
)


def compute_relative_motion(src: GaussianModel, dst: GaussianModel) -> torch.Tensor:
    point_count = min(src.get_xyz.shape[0], dst.get_xyz.shape[0])
    if point_count == 0:
        raise ValueError("Cannot compute relative motion from empty Gaussian models.")

    src_rot = norm_quaternion(src.get_rotation[:point_count])
    dst_rot = norm_quaternion(dst.get_rotation[:point_count])
    rel_rotations = quaternion_multiply(dst_rot, quaternion_inverse(src_rot))
    rel_rotations = norm_quaternion(rel_rotations)
    rel_rots = build_rotation(rel_rotations)

    src_xyz = src.get_xyz[:point_count].reshape(-1, 3, 1)
    dst_xyz = dst.get_xyz[:point_count].reshape(-1, 3, 1)
    rel_xyz = dst_xyz - torch.einsum("ijk,ikn->ijn", rel_rots, src_xyz)

    return torch.cat([rel_rots, rel_xyz], dim=2).reshape(-1, 3, 4)


def save_relative_motion(path: Path, rel_trans: torch.Tensor) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rel_trans.detach().to(torch.float32).cpu().reshape(-1, 3, 4), str(path))
    return path


def load_relative_motion(path: Path, device: Optional[torch.device] = None) -> torch.Tensor:
    rel_trans = torch.load(str(path), map_location="cpu").to(torch.float32).reshape(-1, 3, 4)
    if device is not None:
        rel_trans = rel_trans.to(device)
    return rel_trans


def apply_relative_motion(src: GaussianModel, rel_trans: torch.Tensor) -> GaussianModel:
    point_count = min(src.get_xyz.shape[0], rel_trans.shape[0])
    if point_count == 0:
        raise ValueError("Cannot apply relative motion to an empty Gaussian model.")

    device = src.get_xyz.device
    rel_trans = rel_trans.to(device=device, dtype=torch.float32).reshape(-1, 3, 4)
    rel_rots = rel_trans[:point_count, :, :3]
    rel_xyz = rel_trans[:point_count, :, 3]

    src_xyz = src.get_xyz[:point_count].reshape(-1, 3, 1)
    warped_xyz = torch.einsum("ijk,ikn->ijn", rel_rots, src_xyz).squeeze(-1) + rel_xyz

    src_rot_matrix = batch_qvec2rotmat_torch(norm_quaternion(src.get_rotation[:point_count]))
    warped_rot = batch_rotmat2qvec_torch(torch.matmul(rel_rots, src_rot_matrix))

    out = copy.deepcopy(src)
    out._xyz = warped_xyz.detach()
    out._rotation = warped_rot.detach()
    out._opacity = src._opacity[:point_count].detach().clone()
    out._scaling = src._scaling[:point_count].detach().clone()
    out._features_dc = src._features_dc[:point_count].detach().clone()
    out._features_rest = src._features_rest[:point_count].detach().clone()
    return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)
        return self.gamma * out + x


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ReperformerUNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        super().__init__()
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.attention2 = SelfAttention(512)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.attention2(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class ReperformerSkinSynthesizer:
    def __init__(
        self,
        database: Database,
        action: str,
        animation: str,
        sh_degree: int,
    ):
        self.database = database
        self.action = action
        self.animation = animation
        self.sh_degree = int(sh_degree)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("Reperformer transition synthesis requires CUDA.")

        joint_path = database.get_canonical_frame(action, animation, variant="joint")
        morton_assets = database.get_canonical_morton_map(action, animation)
        checkpoint_path = database.get_canonical_unet(action, animation)
        if joint_path is None:
            raise FileNotFoundError(
                f"Missing canonical joint.ply for {action}/{animation}."
            )
        if morton_assets is None:
            raise FileNotFoundError(
                f"Missing canonical morton_map for {action}/{animation}."
            )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Missing canonical state_dict.pth for {action}/{animation}."
            )

        self.joint_gaussian = GaussianModel(self.sh_degree)
        self.joint_gaussian.load_ply(str(joint_path))

        self.gaussian_canonical = GaussianModel(self.sh_degree)
        self.gaussian_canonical.load_ply(str(morton_assets["point_cloud"]))

        self.pos_map = torch.from_numpy(np.load(morton_assets["pos"])).float().to(self.device)
        self.rotation_map = torch.from_numpy(np.load(morton_assets["rotation"])).float().to(self.device)
        self.non_empty_idx = torch.from_numpy(np.load(morton_assets["u"])).long().to(self.device)
        self.non_empty_idy = torch.from_numpy(np.load(morton_assets["v"])).long().to(self.device)
        self.raw_xyz = self.pos_map[self.non_empty_idx, self.non_empty_idy, :].clone().detach()
        self.raw_rot = self.rotation_map[self.non_empty_idx, self.non_empty_idy, :].clone().detach()
        self.graph_weights, self.indices = self._build_skin_to_joint_graph(
            points=self.raw_xyz,
            joints=self.joint_gaussian.get_xyz.detach(),
        )

        self.color_channel = (self.sh_degree + 1) ** 2
        self.gs_motion_unet = ReperformerUNet(3, 4).to(self.device)
        self.gs_geo_unet = ReperformerUNet(3, 4).to(self.device)
        self.gs_color_unet = ReperformerUNet(3, self.color_channel * 3).to(self.device)
        self._load_checkpoint(Path(checkpoint_path))
        self.gs_motion_unet.eval()
        self.gs_geo_unet.eval()
        self.gs_color_unet.eval()
        self.gaussian = GaussianModel(self.sh_degree)
        self.gaussian.active_sh_degree = self.sh_degree

    def _build_skin_to_joint_graph(
        self,
        points: torch.Tensor,
        joints: torch.Tensor,
        k: int = 8,
        length_scale: float = 0.02,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        points_np = points.detach().cpu().numpy()
        joints_np = joints.detach().cpu().numpy()
        neighbor_count = min(int(k), len(joints_np))
        if neighbor_count <= 0:
            raise ValueError(
                f"Canonical joint.ply for {self.action}/{self.animation} is empty."
            )

        nbrs = NearestNeighbors(n_neighbors=neighbor_count, algorithm="kd_tree").fit(joints_np)
        dists, indices = nbrs.kneighbors(points_np)

        dist_t = torch.from_numpy(dists).float().to(self.device)
        idx_t = torch.from_numpy(indices).long().to(self.device)
        weights = torch.exp(-1.0 * dist_t ** 2 / (length_scale ** 2))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return weights.unsqueeze(-1), idx_t

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        payload = torch.load(str(checkpoint_path), map_location=self.device)

        def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            return {
                key.replace("module.", "", 1) if key.startswith("module.") else key: value
                for key, value in state_dict.items()
            }

        def safe_load(model: nn.Module, key: str) -> None:
            state_dict = payload.get(key)
            if state_dict is None:
                raise KeyError(
                    f"Checkpoint {checkpoint_path} is missing key {key!r}."
                )
            model.load_state_dict(normalize_state_dict(state_dict), strict=False)

        safe_load(self.gs_motion_unet, "gs_motion_unet")
        safe_load(self.gs_geo_unet, "gs_geo_unet")
        safe_load(self.gs_color_unet, "gs_color_unet")

    def _unprojection(self, feature_map: torch.Tensor) -> torch.Tensor:
        vertices = feature_map[0][:, self.non_empty_idx, self.non_empty_idy]
        return vertices.permute(1, 0).contiguous()

    def _warp_morton_points(self, rel_trans: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rel_trans = rel_trans.to(self.device, dtype=torch.float32).reshape(-1, 3, 4)
        max_joint_count = rel_trans.shape[0]
        safe_indices = self.indices.clamp(max=max_joint_count - 1)
        valid = self.indices < max_joint_count
        if not torch.all(valid.any(dim=1)):
            raise ValueError(
                "Relative motion does not cover enough joints to warp the canonical morton map."
            )

        gathered = rel_trans[safe_indices]
        weights = self.graph_weights * valid.unsqueeze(-1)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        rotations = gathered[..., :3]
        translations = gathered[..., 3]
        pos_expanded = self.raw_xyz.unsqueeze(1).expand(-1, gathered.shape[1], -1)
        pos_transformed = torch.einsum("gkij,gkj->gki", rotations, pos_expanded) + translations
        pos_out = torch.sum(pos_transformed * weights, dim=1)

        rot_out = torch.einsum("gkij,gk->gij", rotations, weights.squeeze(-1))
        raw_rot_matrix = batch_qvec2rotmat_torch(norm_quaternion(self.raw_rot))
        new_rotations = batch_rotmat2qvec_torch(torch.matmul(rot_out, raw_rot_matrix))
        return pos_out, new_rotations

    def _forward_warpping(self, warped_pos: torch.Tensor) -> GaussianModel:
        pos_map = self.pos_map.clone()
        pos_map[self.non_empty_idx, self.non_empty_idy] = warped_pos.to(self.device)
        pos_map = pos_map.unsqueeze(0).permute(0, 3, 1, 2)

        pos = self._unprojection(pos_map).detach()
        motion_map = self.gs_motion_unet(pos_map)
        geo_map = self.gs_geo_unet(pos_map)
        color_map = self.gs_color_unet(pos_map)

        motion_attributes = self._unprojection(motion_map)
        geo_attributes = self._unprojection(geo_map)
        color_attributes = self._unprojection(color_map)

        self.gaussian._xyz = pos
        self.gaussian._rotation = motion_attributes[:, :4].detach()
        self.gaussian._scaling = geo_attributes[:, :3].detach()
        self.gaussian._opacity = geo_attributes[:, 3:4].detach()
        features = color_attributes.reshape(self.gaussian._xyz.shape[0], self.color_channel, 3)
        self.gaussian._features_dc = features[:, 0:1, :].detach()
        self.gaussian._features_rest = features[:, 1:, :].detach()
        return copy.deepcopy(self.gaussian)

    def synthesize(self, rel_trans: torch.Tensor) -> GaussianModel:
        with torch.inference_mode():
            warped_pos, warped_rot = self._warp_morton_points(rel_trans)
            self.gaussian._xyz = warped_pos.detach()
            self.gaussian._rotation = warped_rot.detach()
            gaussian = self._forward_warpping(self.gaussian.get_xyz)

            # Warm-start style fallback for extremely early checkpoints that may predict near-zero rotation.
            if torch.allclose(gaussian._rotation.abs().sum(dim=1), torch.zeros_like(gaussian._rotation[:, 0]), atol=1e-8):
                gaussian._rotation = warped_rot.detach()

            return gaussian
