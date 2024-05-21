import torch
import numpy as np

from .utils import ICloudTransformer, RobotModelParams3D, nzrot_to_mat
from .ops import rot_scale_3d
from .model import GaussianModel

from typing import Tuple, NamedTuple
try:
	from typing import Self
except:
	from typing_extensions import Self

TAU = float(2*np.pi)

class RobotPose3D(NamedTuple):
	xyz_tran :torch.Tensor
	mat_rot  :torch.Tensor

	@property
	def n_angle(self) -> torch.Tensor:
		return torch.acos(self.mat_rot[...,2,2])

	def __getitem__(self, idx:int) -> Self:
		return RobotPose3D(xyz_tran=self.xyz_tran[None,idx], mat_rot=self.mat_rot[None,idx])

class RobotModel3D(ICloudTransformer):
	def __init__(self,
		params:RobotModelParams3D=RobotModelParams3D(
			init_vel_std=10.0,
			xy_accel_std=1.0,
			z_accel_std=0.15,
			n_vel_std=0.02,
			angvel_std=0.2,
		)
	):
		self.state = torch.zeros((9,), dtype=torch.float32, device='cuda')
		self._rotframe = None

		self.cov = torch.zeros((9,9), dtype=torch.float32, device='cuda')
		self.cov[6,6] = self.cov[7,7] = params.init_vel_std**2
		self._covchol = None

		self.noise_cov = torch.zeros((6,6), dtype=torch.float32, device='cuda')
		self.noise_cov[0,0] = self.noise_cov[1,1] = params.xy_accel_std**2
		self.noise_cov[2,2] = params.z_accel_std**2
		self.noise_cov[3,3] = self.noise_cov[4,4] = params.n_vel_std**2
		self.noise_cov[5,5] = params.angvel_std**2

		self.I = torch.eye(9, dtype=torch.float32, device='cuda')
		self.H = self.I[0:6,:]

	@property
	def rotframe(self) -> torch.Tensor:
		if self._rotframe is None:
			self._rotframe = nzrot_to_mat(self.state[3:6])
		return self._rotframe

	@property
	def covchol(self) -> torch.Tensor:
		if self._covchol is None:
			self._covchol = torch.linalg.cholesky((self.cov + 1e-4 * self.I)[0:6,0:6])
		return self._covchol

	@property
	def pose(self) -> RobotPose3D:
		return RobotPose3D(xyz_tran=self.state[None,0:3], mat_rot=self.rotframe[None])

	@property
	def z_rot(self) -> float:
		return float(self.state[5])

	@property
	def speed(self) -> float:
		return float(torch.linalg.norm(self.state[6:9]))

	def advance(self, timediff:float) -> None:
		F = self.I.clone()
		F[0,6] = F[1,7] = F[2,8] = timediff

		N = torch.zeros((9,6), dtype=torch.float32, device='cuda')
		N[0,0] = N[1,1] = N[2,2] = 0.5*timediff**2
		N[3,3] = N[4,4] = N[5,5] = N[6,0] = N[7,1] = N[8,2] = timediff

		noise_cov = self.noise_cov.clone()
		noise_cov[0:3,0:3] = self.rotframe @ noise_cov[0:3,0:3] @ self.rotframe.t()

		self.state = F @ self.state
		self.cov = F @ self.cov @ F.t() + N @ noise_cov @ N.t()
		self._covchol = None

	def transform(self, particles:torch.Tensor) -> torch.Tensor:
		p = self.state[None,0:6,None] + self.covchol[None] @ particles[:,0:6,None]
		return p[...,0]

	def transform_as_pose(self, particles:torch.Tensor) -> RobotPose3D:
		p = self.transform(particles)
		return RobotPose3D(xyz_tran=p[...,0:3], mat_rot=nzrot_to_mat(p[...,3:6]))

	# ICloudTransformer
	def transform_cloud(self, cloud:torch.Tensor, particles:torch.Tensor=None) -> torch.Tensor:
		poses = self.transform_as_pose(particles) if particles is not None else self.pose
		return poses.xyz_tran[:,None,:] + (poses.mat_rot[:,None,:] @ cloud[None,:,:,None])[...,0]

	def update(self, new_pose:torch.Tensor, new_cov:torch.Tensor) -> None:
		K = self.cov @ self.H.t() @ torch.linalg.inv(self.H @ self.cov @ self.H.t() + new_cov)
		L = self.I - K @ self.H

		self.state += K @ (new_pose - self.H @ self.state)
		self.state[0:6] = new_pose
		self._rotframe = None

		self.cov = L @ self.cov @ L.t() + K @ new_cov @ K.t()
		self._covchol = None
