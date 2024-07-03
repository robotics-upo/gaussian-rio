import torch
import numpy as np

from .utils import ICloudTransformer, quat_to_rpy
from .ops import rot_scale_3d
from .model import GaussianModel
from .robot3d import RobotPose3D

from typing import Tuple, NamedTuple
try:
	from typing import Self
except:
	from typing_extensions import Self

TAU = float(2*np.pi)

I_p  = slice(0,3)   # Position vector (world-space)
I_v  = slice(3,6)   # Velocity vector (world-space)
I_g  = slice(6,9)   # Gravity acceleration vector (world-space)
I_ab = slice(9,12)  # Accelerometer bias
I_wb = slice(12,15) # Gyroscope bias
I_q  = slice(15,19) # Attitude quaternion
I_dz = slice(15,18) # Attitude error vector (world-space)

I_ekf = slice(I_p.start,I_q.start)

W_v = slice(0,3)  # Velocity process noise (world-space)
W_z = slice(3,6)  # Attitude error process noise (world-space)
W_a = slice(6,9)  # Accelerometer noise (body-space)
W_w = slice(9,12) # Gyroscope noise (body-space)

SD_statelen = I_q.stop
SD_len = I_dz.stop
SD_noiselen = W_w.stop

def skewsym(v: torch.Tensor) -> torch.Tensor:
	x,y,z = float(v[0]),float(v[1]),float(v[2])
	return torch.as_tensor([
		[ 0.0, -z, +y ],
		[ +z, 0.0, -x ],
		[ -y, +x, 0.0 ],
	], dtype=torch.float32, device='cuda')

def quat_to_mat(q:torch.Tensor) -> torch.Tensor:
	return rot_scale_3d(torch.ones((1,3), dtype=torch.float32, device='cuda'), q[None])[0]

def quat_mult(q0:torch.Tensor, q1:torch.Tensor):
	w0,x0,y0,z0 = q0[...,0], q0[...,1], q0[...,2], q0[...,3]
	w1,x1,y1,z1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
	return torch.stack([
		w0*w1 - x0*x1 - y0*y1 - z0*z1,
		w0*x1 + x0*w1 + y0*z1 - z0*y1,
		w0*y1 + y0*w1 + z0*x1 - x0*z1,
		w0*z1 + z0*w1 + x0*y1 - y0*x1,
	], dim=-1)

def pure_quat_exp(w: torch.Tensor) -> torch.Tensor:
	norm = torch.linalg.norm(w)
	qw,sinc = torch.cos(norm), torch.sinc(norm / (0.5*TAU))
	qv = w*sinc
	return torch.stack([ qw, qv[0], qv[1], qv[2] ], dim=-1)

class Strapdown(ICloudTransformer):
	def __init__(self):
		self.state = torch.zeros((SD_statelen,), dtype=torch.float32, device='cuda')
		self.state[I_g.start+2] = -9.80511
		self.state[I_q.start] = 1
		self._rotframe = None

		self.cov = torch.zeros((SD_len,SD_len), dtype=torch.float32, device='cuda')
		#self.cov[I_g.start+0,I_g.start+0] = 0.01**2
		#self.cov[I_g.start+1,I_g.start+1] = 0.01**2
		#self.cov[I_g.start+2,I_g.start+2] = 0.2**2
		#self.cov[I_ab,I_ab].fill_diagonal_(0.01**2)
		#self.cov[I_ab.start+2,I_ab.start+2] = 0
		self.cov[I_wb,I_wb].fill_diagonal_((0.5*TAU/360)**2) # 0.01
		self.cov[I_wb.start+2,I_wb.start+2] = 0 # To be disabled when a reliable source of yaw is incorporated
		#self.cov[I_dz.start+0,I_dz.start+0] = 0.01**2
		#self.cov[I_dz.start+0,I_dz.start+0] = 0.01**2
		self._covchol = None

		self.I = torch.eye(SD_len, dtype=torch.float32, device='cuda')

	@property
	def rotframe(self) -> torch.Tensor:
		if self._rotframe is None:
			self._rotframe = quat_to_mat(self.state[I_q])
		return self._rotframe

	@property
	def pos(self) -> torch.Tensor:
		return self.state[I_p]

	@property
	def quat(self) -> torch.Tensor:
		return self.state[I_q]

	@property
	def gravity(self) -> torch.Tensor:
		return self.state[I_g]

	@property
	def accel_bias(self) -> torch.Tensor:
		return self.state[I_ab]

	@property
	def gyro_bias(self) -> torch.Tensor:
		return self.state[I_wb]

	def init_vel(self, vel:torch.Tensor, velcov:torch.Tensor) -> None:
		self.state[I_v] = vel
		self.cov[I_v,I_v] = velcov
		self._covchol = None

	def advance(self, timediff:float, accel:torch.Tensor, accel_cov:torch.Tensor, omega:torch.Tensor, omega_cov:torch.Tensor):
		accel = torch.as_tensor(accel, dtype=torch.float32, device='cuda') - self.state[I_ab]
		accel_cov = torch.as_tensor(accel_cov, dtype=torch.float32, device='cuda')
		omega = torch.as_tensor(omega, dtype=torch.float32, device='cuda') - self.state[I_wb]
		omega_cov = torch.as_tensor(omega_cov, dtype=torch.float32, device='cuda')

		R = self.rotframe
		A = skewsym(accel_world := R @ accel)
		g = self.state[I_g]

		self.state[I_p] += timediff*self.state[I_v] + 0.5*(accel_world + g)*timediff**2
		self.state[I_v] += timediff*(accel_world + g)
		self.state[I_q] = quat_mult(self.state[I_q], pure_quat_exp(0.5*omega*timediff))
		self._rotframe = None

		F = torch.eye(SD_len, dtype=torch.float32, device='cuda')
		F[I_p,I_v].fill_diagonal_(timediff)
		F[I_p,I_g].fill_diagonal_(0.5*timediff**2)
		F[I_p,I_ab] = -0.5*R*timediff**2
		F[I_p,I_dz] = -0.5*A*timediff**2
		F[I_v,I_g].fill_diagonal_(timediff)
		F[I_v,I_ab] = F[I_dz,I_wb] = -R*timediff
		F[I_v,I_dz] = -A*timediff

		Q = torch.zeros((SD_noiselen,SD_noiselen), dtype=torch.float32, device='cuda')
		Q[W_v,W_v].fill_diagonal_(0.1**2) # 0.05
		Q[W_z,W_z].fill_diagonal_(0.005**2) # 0.01
		Q[W_a,W_a] = accel_cov
		Q[W_w,W_w] = omega_cov

		N = torch.zeros((SD_len,SD_noiselen), dtype=torch.float32, device='cuda')
		N[I_p,W_a] = 0.5*R*timediff**2
		N[I_v,W_v].fill_diagonal_(1)
		N[I_v,W_a] = N[I_dz,W_w] = R*timediff
		N[I_dz,W_z].fill_diagonal_(1)

		self.cov = F @ self.cov @ F.t() + N @ Q @ N.t()
		self._covchol = None

	def transform(self, particles:torch.Tensor) -> torch.Tensor:
		raise NotImplementedError()

	def transform_as_pose(self, particles:torch.Tensor) -> RobotPose3D:
		raise NotImplementedError()

	# ICloudTransformer
	def transform_cloud(self, cloud:torch.Tensor, particles:torch.Tensor=None) -> torch.Tensor:
		raise NotImplementedError()

	def _update_common(self, y:torch.Tensor, y_cov:torch.Tensor, H:torch.Tensor) -> None:
		K = self.cov @ H.t() @ torch.linalg.inv(H @ self.cov @ H.t() + y_cov)
		L = self.I - K @ H

		dtheta = torch.zeros((3,), dtype=torch.float32, device='cuda')
		state = torch.concat((self.state[I_ekf], dtheta), dim=0)
		state += K @ (y - H @ state)
		self.state[I_ekf] = state[I_ekf]
		self.cov = L @ self.cov @ L.t() + K @ y_cov @ K.t()
		self._covchol = None

		# ESKF reset
		self.state[I_q] = quat_mult(qerr:=pure_quat_exp(0.5*state[I_dz]), self.state[I_q])
		self.state[I_q] /= torch.linalg.norm(self.state[I_q]) # renormalize to avoid rounding errors
		self._rotframe = None
		resetmat = torch.eye(SD_len, dtype=torch.float32, device='cuda')
		resetmat[I_dz,I_dz] = quat_to_mat(qerr)
		self.cov = resetmat @ self.cov @ resetmat.t()

	def update_egovel(self, egovel:torch.Tensor, egovel_cov:torch.Tensor) -> None:
		R = self.rotframe
		v = self.state[I_v]
		egovel = torch.as_tensor(egovel, dtype=torch.float32, device='cuda')
		egovel_cov = torch.as_tensor(egovel_cov, dtype=torch.float32, device='cuda')

		H = torch.zeros((3,SD_len), dtype=torch.float32, device='cuda')
		H[:,I_v] = R.t()

		self._update_common(egovel, egovel_cov, H)

	def update_dtheta(self, dtheta:torch.Tensor, dtheta_cov:torch.Tensor) -> None:
		dtheta = torch.as_tensor(dtheta, dtype=torch.float32, device='cuda')
		dtheta_cov = torch.as_tensor(dtheta_cov, dtype=torch.float32, device='cuda')

		H = torch.zeros((3,SD_len), dtype=torch.float32, device='cuda')
		H[:,I_dz].fill_diagonal_(1)

		self._update_common(dtheta, dtheta_cov, H)

	def update_pose(self, pose:torch.Tensor, pose_cov:torch.Tensor) -> None:
		raise NotImplementedError()