import torch
import numpy as np

from .utils import ICloudTransformer
from .ops import rot_scale_3d
from .model import GaussianModel

from typing import Tuple, NamedTuple
try:
	from typing import Self
except:
	from typing_extensions import Self

TAU = float(2*np.pi)

class RobotPose2D(NamedTuple):
	xy_tran :torch.Tensor
	z_cossin:torch.Tensor

	@property
	def z_rot(self) -> torch.Tensor:
		return torch.atan2(self.z_cossin[...,1], self.z_cossin[...,0])

	@staticmethod
	def from_particles(particles:torch.Tensor, polar_tran:bool=False) -> Self:
		z_rot = particles[...,2] * TAU * 45 / 360
		z_cos = torch.cos(z_rot)
		z_sin = torch.sin(z_rot)
		z_cossin = torch.stack((z_cos, z_sin), dim=-1)

		if not polar_tran:
			xy_tran = 0.25 * particles[...,0:2]
		else:
			d_tran = 0.25 * particles[...,0]
			x_tran = d_tran*z_cos
			y_tran = d_tran*z_sin
			xy_tran = torch.stack([x_tran, y_tran], dim=-1)

		return RobotPose2D(xy_tran=xy_tran, z_cossin=z_cossin)

	def __mul__(parent, child:Self) -> Self:
		if child is None:
			return parent

		parent_cos = parent.z_cossin[...,0]
		parent_sin = parent.z_cossin[...,1]
		child_cos = child.z_cossin[...,0]
		child_sin = child.z_cossin[...,1]
		z_cos = parent_cos*child_cos - parent_sin*child_sin
		z_sin = parent_sin*child_cos + parent_cos*child_sin
		z_cossin = torch.stack((z_cos, z_sin), dim=-1)

		child_x = child.xy_tran[...,0]
		child_y = child.xy_tran[...,1]
		x_tran = parent_cos*child_x - parent_sin*child_y
		y_tran = parent_sin*child_x + parent_cos*child_y
		xy_tran = parent.xy_tran + torch.stack([x_tran, y_tran], dim=-1)

		return RobotPose2D(xy_tran=xy_tran, z_cossin=z_cossin)

	def __rmul__(child, parent:Self) -> Self:
		if parent is not None:
			raise TypeError("Invalid type")

		return child

	def to_xlate_rot(self) -> Tuple[torch.Tensor, torch.Tensor]:
		tran = torch.nn.functional.pad(self.xy_tran, (0,1))

		q_cos = 1 + self.z_cossin[...,0]
		q_sin = self.z_cossin[...,1]
		q_adj = torch.sqrt(q_cos*q_cos + q_sin*q_sin)
		q_cos = q_cos / q_adj
		q_sin = q_sin / q_adj

		quat = torch.stack([ q_cos, 0*q_sin, 0*q_sin, 1*q_sin ], dim=-1)

		rot = rot_scale_3d(
			scales=torch.ones(quat.shape[:-1] + (3,), dtype=quat.dtype, device=quat.device),
			quats=quat
		)

		return tran, rot
