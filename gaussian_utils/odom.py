import torch
import numpy as np
import small_gicp

from .radar import ImuData, RadarData, calc_radar_egovel
from .utils import quat_to_rpy, rpy_to_quat, quat_conj, quat_mult, quat_to_mat, mat_to_quat
from .robot3d import RobotPose3D
from .strapdown import Strapdown

TAU = float(2*np.pi)

class ImuRadarOdometry(Strapdown):
	def __init__(self, seed=3135134162, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.ref_time = None
		self.imu_time = None
		self.imu_rp = (0.0,0.0)
		self.egovel = None
		self.egovel_frame = np.eye(5, dtype=np.float32)

	@property
	def is_initial(self) -> bool:
		return self.ref_time is None

	@property
	def time(self) -> float:
		return self.imu_time - self.ref_time

	def initialize(self, ref_time:float, vel:np.ndarray, vel_cov:np.ndarray) -> None:
		assert self.is_initial
		self.ref_time = ref_time
		self.imu_time = ref_time
		self.init_vel(vel, vel_cov)

	def process_imu(self, bundle:RadarData) -> None:
		if self.is_initial or len(bundle.imu) == 0:
			return

		saved_quat = self.quat.cpu().numpy()
		#saved_vel = self.vel.cpu().numpy()
		#saved_time = self.imu_time

		for imu in bundle.imu:
			tdiff = imu.t - self.imu_time
			assert tdiff >= 0.0
			self.imu_time = imu.t

			self.advance(tdiff, imu.accel, imu.accel_cov, imu.omega, imu.omega_cov)
			imu.accel -= self.accel_bias.cpu().numpy()

		#est_acc = (self.vel.cpu().numpy() - saved_vel) / (self.imu_time - saved_time)
		#for imu in bundle.imu:
		#	imu.accel -= est_acc

		imu_roll,imu_pitch,_ = bundle.roll_pitch_g
		cur_quat = self.quat.cpu().numpy()
		_,_,cur_yaw = quat_to_rpy(cur_quat)

		orient = rpy_to_quat(np.stack((imu_roll,imu_pitch,cur_yaw), axis=-1))
		qerror = quat_mult(orient, quat_conj(cur_quat))
		qerrormat = quat_to_mat(qerror)
		qerror /= qerror[0] # ensure w=1 (and also fix signs)

		dtheta = 2*qerror[1:4]
		dtheta_cov = np.zeros((3,3), dtype=np.float32)
		dtheta_cov[0,0] = dtheta_cov[1,1] = (2*0.05)**2 # 2 sigma
		dtheta_cov[2,2] = dtheta_cov[0,0] #(2*0.002)**2
		dtheta_cov += qerrormat @ self.dtheta_cov.cpu().numpy() @ qerrormat.T
		self.update_dtheta(dtheta, dtheta_cov)

		self.imu_rp = (imu_roll,imu_pitch)
		self.egovel_frame[0:3,0:3] = quat_to_mat(quat_mult(self.quat.cpu().numpy(), quat_conj(saved_quat)))

	def process_radar(self, cl:np.ndarray, t:float, **kwargs) -> np.ndarray:
		try:
			egovel, inliers = calc_radar_egovel(cl @ self.egovel_frame, force_forward=True, **kwargs)
		except:
			print('**WARNING**: egovel extraction fail')
			return cl

		R = self.egovel_frame[0:3,0:3]
		vel = R @ egovel.vel
		vel_cov = R @ egovel.cov @ R.T

		self.egovel = vel

		if self.is_initial:
			self.initialize(t, vel, vel_cov)
		else:
			self.update_egovel(vel, vel_cov)

		if inliers is not None:
			cl = cl[inliers]

		if self.has_keyframe:
			self.scan_matching(cl)

		return cl

	@property
	def has_keyframe(self) -> bool:
		return False

	def scan_matching(self, cl:np.ndarray) -> None:
		raise NotImplementedError()

class ImuRadarGicpOdometry(ImuRadarOdometry):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, want_yaw_gyro_bias=True, **kwargs)

		self.kf_pose : RobotPose3D = None
		self.downsampling = 1.0

	@property
	def has_keyframe(self) -> bool:
		return self.kf_pose is not None

	def keyframe(self, cl:np.ndarray, *args, **kwargs) -> None:
		super().keyframe()
		self.kf_pose = self.pose

		self.kf_cl, self.kf_tree = small_gicp.preprocess_points(cl[:,0:3].astype(np.float64), *args, downsampling_resolution=self.downsampling, **kwargs)

	def scan_matching(self, cl:np.ndarray, *args, **kwargs) -> None:
		relpose = (self.pose - self.kf_pose).xfrm_4x4[0].cpu().numpy()

		cur_cl, cur_tree = small_gicp.preprocess_points(cl[:,0:3].astype(np.float64), *args, downsampling_resolution=self.downsampling, **kwargs)

		result = small_gicp.align(self.kf_cl, cur_cl, self.kf_tree, relpose)
		result : small_gicp.RegistrationResult
		if not result.converged:
			print('  {WARN} Scan matching fail')
			return

		relpose = RobotPose3D.from_xfrm(result.T_target_source[None])

		newpose = self.kf_pose + relpose

		curquat = self.quat.cpu().numpy()
		newquat = mat_to_quat(newpose.mat_rot[0].cpu().numpy())
		qerror = quat_mult(newquat, quat_conj(curquat))
		qerror /= qerror[0] # ensure w=1 (and also fix signs)

		xyz = newpose.xyz_tran[0].cpu().numpy()
		dtheta = 2*qerror[1:4]

		basecov = self.particle_basecov
		noisecov = torch.zeros_like(basecov)
		noisecov[0:3,0:3].fill_diagonal_(1.0)
		noisecov[3:6,3:6].fill_diagonal_(0.01)

		kfpose = np.concatenate((xyz, dtheta))
		self.update_pose(kfpose, basecov + noisecov)