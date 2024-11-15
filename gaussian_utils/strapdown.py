import numpy as np

from .robot3d import RobotPose3D
from .utils import mat_to_quat, quat_to_mat, quat_mult, pure_quat_exp, skewsym

import itertools
from typing import List, Tuple, NamedTuple
try:
	from typing import Self
except:
	from typing_extensions import Self

TAU = float(2*np.pi)

I_p  = slice(0,3)   # Position vector (world-space)
I_v  = slice(3,6)   # Velocity vector (world-space)
I_ab = slice(6,9)  # Accelerometer bias
I_wb = slice(9,12) # Gyroscope bias
I_q  = slice(12,16) # Attitude quaternion
I_dz = slice(12,15) # Attitude error vector (world-space)

I_ekf = slice(I_p.start,I_q.start)

W_v = slice(0,3)  # Velocity process noise (world-space)
W_z = slice(3,6)  # Attitude error process noise (world-space)
W_a = slice(6,9)  # Accelerometer noise (body-space)
W_w = slice(9,12) # Gyroscope noise (body-space)

SD_statelen = I_q.stop
SD_len = I_dz.stop
SD_noiselen = W_w.stop

class Strapdown:
	def __init__(self,
		gravity=9.80511,
		accel_init_bias_std=0.02, #1.0,
		gyro_init_bias_std=0.00035*TAU/360,
		accel_bias_std=0.0,
		gyro_bias_std=0.0,
		rp_init_std=5*TAU/360, # 10.0
		want_yaw_gyro_bias=False
	):
		self.state = np.zeros((SD_statelen,), dtype=np.float64)
		self.gravity = np.asarray([0.0, 0.0, -gravity], dtype=np.float64)
		self.state[I_q.start] = 1
		self._rotframe = None

		self.cov = np.zeros((SD_len,SD_len), dtype=np.float64)

		self.accel_bias_var = accel_bias_std**2
		self.gyro_bias_var = gyro_bias_std**2

		np.fill_diagonal(self.cov[I_ab,I_ab], accel_init_bias_std**2)
		np.fill_diagonal(self.cov[I_wb,I_wb], gyro_init_bias_std**2)
		if not want_yaw_gyro_bias: # Disable gyroscope yaw bias estimation when there isn't a reliable source of yaw
			self.cov[I_wb.start+2,I_wb.start+2] = 0

		self.cov[I_dz.start+0,I_dz.start+0] = self.cov[I_dz.start+1,I_dz.start+1] = rp_init_std

		self.I = np.eye(SD_len, dtype=np.float64)
		self.zero_dtheta = np.zeros((3,), dtype=np.float64)


		Strapdown.keyframe(self)

	def keyframe(self) -> None:
		self.kf_pose = self.pose
		self.kf_cov = self.cov
		self.kf_frame = self.rotframe

	@property
	def rotframe(self) -> np.ndarray:
		if self._rotframe is None:
			self._rotframe = quat_to_mat(self.quat)
		return self._rotframe

	@property
	def pose(self) -> RobotPose3D:
		return RobotPose3D.from_tran_rot(self.pos, self.rotframe)

	@property
	def pos(self) -> np.ndarray:
		return self.state[I_p].copy()

	@property
	def vel(self) -> np.ndarray:
		return self.state[I_v].copy()

	@property
	def egovel(self) -> np.ndarray:
		return self.rotframe.T @ self.vel

	@property
	def quat(self) -> np.ndarray:
		return self.state[I_q].copy()

	@property
	def antigravity(self) -> np.ndarray:
		return self.rotframe.T @ (-self.gravity) + self.accel_bias

	@property
	def accel_bias(self) -> np.ndarray:
		return self.state[I_ab].copy()

	@property
	def gyro_bias(self) -> np.ndarray:
		return self.state[I_wb].copy()

	@property
	def dtheta_cov(self) -> np.ndarray:
		return self.cov[I_dz,I_dz].copy()

	def init_vel(self, vel:np.ndarray, vel_cov:np.ndarray) -> None:
		vel = np.asarray(vel, dtype=np.float64)
		vel_cov = np.asarray(vel_cov, dtype=np.float64)

		self.state[I_v] = vel
		self.cov[I_v,I_v] = vel_cov

	def advance(self, timediff:float, accel:np.ndarray, accel_cov:np.ndarray, omega:np.ndarray, omega_cov:np.ndarray):
		accel = np.asarray(accel, dtype=np.float64) - self.state[I_ab]
		accel_cov = np.asarray(accel_cov, dtype=np.float64)
		omega = np.asarray(omega, dtype=np.float64) - self.state[I_wb]
		omega_cov = np.asarray(omega_cov, dtype=np.float64)

		R = self.rotframe
		A = skewsym(accel_world := R @ accel)
		g = self.gravity

		self.state[I_p] += timediff*self.state[I_v] + 0.5*(accel_world + g)*timediff**2
		self.state[I_v] += timediff*(accel_world + g)
		self.state[I_q] = quat_mult(self.state[I_q], pure_quat_exp(0.5*omega*timediff))
		self._rotframe = None

		F = np.eye(SD_len, dtype=np.float64)
		np.fill_diagonal(F[I_p,I_v], timediff)
		F[I_p,I_ab] = -0.5*R*timediff**2
		F[I_p,I_dz] = -0.5*A*timediff**2
		F[I_v,I_ab] = F[I_dz,I_wb] = -R*timediff
		F[I_v,I_dz] = -A*timediff

		Q = np.zeros((SD_noiselen,SD_noiselen), dtype=np.float64)
		np.fill_diagonal(Q[W_v,W_v], 0.1**2) # 0.05
		np.fill_diagonal(Q[W_z,W_z], 0.005**2) # 0.01
		Q[W_a,W_a] = accel_cov/timediff
		Q[W_w,W_w] = omega_cov/timediff

		N = np.zeros((SD_len,SD_noiselen), dtype=np.float64)
		N[I_p,W_a] = 0.5*R*timediff**2
		np.fill_diagonal(N[I_v,W_v], 1)
		N[I_v,W_a] = N[I_dz,W_w] = R*timediff
		np.fill_diagonal(N[I_dz,W_z], 1)

		self.cov = F @ self.cov @ F.T + N @ Q @ N.T

		I3 = self.I[0:3,0:3]
		self.cov[I_ab,I_ab] += I3 * (self.accel_bias_var*timediff)
		self.cov[I_wb,I_wb] += I3 * ( self.gyro_bias_var*timediff)

	def _update_common(self, name:str, y:np.ndarray, y_cov:np.ndarray, H:np.ndarray, is_residual:bool=False, gamma_threshold:float=None) -> None:
		state = np.concatenate((self.state[I_ekf], self.zero_dtheta), axis=0)
		residual = y if is_residual else (y - H @ state)
		Sinv = np.linalg.inv(H @ self.cov @ H.T + y_cov)
		gamma = float((residual[None] @ Sinv @ residual[:,None]).flatten())

		if gamma_threshold is not None and gamma > gamma_threshold:
			print(f'  {{WARN}} KF {name} update rejected, gamma = {gamma}')
			return

		K = self.cov @ H.T @ Sinv
		L = self.I - K @ H

		state += K @ residual
		self.state[I_ekf] = state[I_ekf]
		self.cov = L @ self.cov @ L.T + K @ y_cov @ K.T

		# ESKF reset
		self.state[I_q] = quat_mult(qerr:=pure_quat_exp(0.5*state[I_dz]), self.state[I_q])
		self.state[I_q] /= np.linalg.norm(self.state[I_q]) # renormalize to avoid rounding errors
		self._rotframe = None
		resetmat = np.eye(SD_len, dtype=np.float64)
		resetmat[I_dz,I_dz] = quat_to_mat(qerr)
		self.cov = resetmat @ self.cov @ resetmat.T

	def update_antigravity(self, ag:np.ndarray, ag_cov:np.ndarray) -> None:
		R = self.rotframe
		g = self.gravity
		ag = np.asarray(ag, dtype=np.float64)
		ag_cov = np.asarray(ag_cov, dtype=np.float64)
		agnorm = np.linalg.norm(ag)

		H = np.zeros((3,SD_len), dtype=np.float64)
		np.fill_diagonal(H[:,I_ab], 1)
		H[:,I_dz] = -R.T @ skewsym(g)

		self._update_common('ag', ag/agnorm, ag_cov/(agnorm**2), H/np.linalg.norm(self.antigravity)) #, gamma_threshold=0.266)

	def update_egovel(self, egovel:np.ndarray, egovel_cov:np.ndarray, radar_arm:np.ndarray=None) -> None:
		R = self.rotframe
		v = self.state[I_v]
		residual = np.asarray(egovel, dtype=np.float64) - self.egovel
		egovel_cov = np.asarray(egovel_cov, dtype=np.float64)

		H = np.zeros((3,SD_len), dtype=np.float64)
		H[:,I_v] = R.T
		H[:,I_dz] = R.T @ skewsym(v)
		if radar_arm is not None:
			H[:,I_wb] = skewsym(radar_arm)

		self._update_common('egovel', residual, egovel_cov, H, True)

	def update_scanmatch(self, pose:RobotPose3D, pose_cov:np.ndarray, dof:List[int]=None) -> None:
		predpose = self.pose - self.kf_pose
		diff_xyz = pose.xyz_tran[0] - predpose.xyz_tran[0]
		diff_rot = predpose.mat_rot[0].t() @ pose.mat_rot[0]

		diff_xyz = np.asarray(diff_xyz.cpu(), dtype=np.float64)
		diff_rot = np.asarray(diff_rot.cpu(), dtype=np.float64)

		diff_q = mat_to_quat(diff_rot)
		diff_q /= diff_q[0] # ensure w=1 (and also fix signs)
		diff_dz = 2*diff_q[1:4]

		residual = np.concatenate((diff_xyz, diff_dz), axis=0)

		H = np.zeros((6,SD_len), dtype=np.float64)
		H[0:3,I_p] = H[3:6,I_dz] = self.kf_frame.T

		pose_cov = np.asarray(pose_cov, dtype=np.float64) + H @ self.kf_cov @ H.T

		if dof is not None:
			residual = residual[dof]
			pose_cov = pose_cov[dof][:,dof]
			H = H[dof]

		self._update_common('scanmatch', residual, pose_cov, H, True, 6.251)
