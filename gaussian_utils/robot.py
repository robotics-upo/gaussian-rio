import torch
import numpy as np

from .utils import ICloudTransformer, RobotModelParams
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

	def __getitem__(self, idx:int) -> Self:
		return RobotPose2D(xy_tran=self.xy_tran[None,idx], z_cossin=self.z_cossin[None,idx])

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

class RobotModel2D(ICloudTransformer):
	def __init__(self,
		params:RobotModelParams=RobotModelParams(
			init_vel_std=10.0,
			accel_std=1.0,
			angvel_std=0.2,
		)
	):
		self.state = torch.zeros((5,), dtype=torch.float32, device='cuda')
		self.cov = torch.zeros((5,5), dtype=torch.float32, device='cuda')
		self.cov[3,3] = self.cov[4,4] = params.init_vel_std**2
		self._covchol = None

		self.noise_cov = torch.zeros((3,3), dtype=torch.float32, device='cuda')
		self.noise_cov[0,0] = self.noise_cov[1,1] = params.accel_std**2
		self.noise_cov[2,2] = params.angvel_std**2

		self.H = torch.zeros((3,5), dtype=torch.float32, device='cuda')
		self.H[0,0] = self.H[1,1] = self.H[2,2] = 1

		self.I = torch.eye(5, dtype=torch.float32, device='cuda')

	@property
	def covchol(self) -> torch.Tensor:
		if self._covchol is None:
			self._covchol = torch.linalg.cholesky(self.cov + 1e-4 * self.I)
		return self._covchol

	@property
	def pose(self) -> RobotPose2D:
		zcos = torch.cos(self.state[None,2])
		zsin = torch.sin(self.state[None,2])
		return RobotPose2D(xy_tran=self.state[None,0:2], z_cossin=torch.stack((zcos,zsin),dim=1))

	@property
	def speed(self) -> float:
		return float(torch.linalg.norm(self.state[3:5]))

	def advance(self, timediff:float) -> None:
		F = self.I.clone()
		F[0,3] = F[1,4] = timediff

		N = torch.zeros((5,3), dtype=torch.float32, device='cuda')
		N[0,0] = N[1,1] = 0.5*timediff**2
		N[2,2] = N[3,0] = N[4,1] = timediff

		self.state = F @ self.state
		self.cov = F @ self.cov @ F.t() + N @ self.noise_cov @ N.t()
		self._covchol = None

	def transform(self, particles:torch.Tensor) -> torch.Tensor:
		p = self.state[None,:,None] + self.covchol[None,:,:] @ torch.nn.functional.pad(particles[:,0:3,None], (0,0,0,2))
		return p[:,0:3,0]

	def transform_as_pose(self, particles:torch.Tensor) -> RobotPose2D:
		p = self.transform(particles)
		zcos = torch.cos(p[:,2])
		zsin = torch.sin(p[:,2])
		return RobotPose2D(xy_tran=p[:,0:2], z_cossin=torch.stack((zcos,zsin),dim=1))

	# ICloudTransformer
	def transform_cloud(self, cloud:torch.Tensor, particles:torch.Tensor=None) -> torch.Tensor:
		poses = self.transform_as_pose(particles) if particles is not None else self.pose
		xlate,rot = poses.to_xlate_rot()
		return xlate[:,None,:] + (rot[:,None,:] @ cloud[None,:,:,None])[...,0]

	def update(self, new_pose:torch.Tensor, new_cov:torch.Tensor) -> None:
		K = self.cov @ self.H.t() @ torch.linalg.inv(self.H @ self.cov @ self.H.t() + new_cov)
		L = self.I - K @ self.H

		self.state += K @ (new_pose - self.H @ self.state)
		self.cov = L @ self.cov @ L.t() + K @ new_cov @ K.t()
		self._covchol = None
