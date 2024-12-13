import torch
import numpy as np

from .utils import ICloudTransformer, RobotModelParams3D
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

	@staticmethod
	def from_tran_rot(tran:torch.Tensor, rot:torch.Tensor) -> Self:
		tran = torch.as_tensor(tran, dtype=torch.float32, device='cuda')[None]
		rot  = torch.as_tensor(rot,  dtype=torch.float32, device='cuda')[None]
		return RobotPose3D(xyz_tran=tran, mat_rot=rot)

	@staticmethod
	def from_xfrm(xfrm:torch.Tensor) -> Self:
		xfrm = torch.as_tensor(xfrm, dtype=torch.float32, device='cuda')
		return RobotPose3D(xyz_tran=xfrm[:,0:3,3], mat_rot=xfrm[:,0:3,0:3])

	@property
	def xfrm_3x4(self) -> torch.Tensor:
		return torch.concat([self.mat_rot, self.xyz_tran[...,None]], dim=-1)

	@property
	def xfrm_4x4(self) -> torch.Tensor:
		ret = torch.nn.functional.pad(self.xfrm_3x4, (0,0,0,1))
		ret[...,3,3] = 1.0
		return ret

	def __getitem__(self, idx:int) -> Self:
		return RobotPose3D(xyz_tran=self.xyz_tran[None,idx], mat_rot=self.mat_rot[None,idx])

	def __add__(lhs, rhs:Self) -> Self:
		rot = lhs.mat_rot @ rhs.mat_rot
		xyz = lhs.xyz_tran + (lhs.mat_rot @ rhs.xyz_tran[...,None])[...,0]
		return RobotPose3D(xyz_tran=xyz, mat_rot=rot)

	def __sub__(lhs, rhs:Self) -> Self:
		rhs_rot_tran = torch.transpose(rhs.mat_rot, 1, 2)
		rot = rhs_rot_tran @ lhs.mat_rot
		xyz = (rhs_rot_tran @ (lhs.xyz_tran - rhs.xyz_tran)[...,None])[...,0]
		return RobotPose3D(xyz_tran=xyz, mat_rot=rot)
