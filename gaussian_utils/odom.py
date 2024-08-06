import torch
import numpy as np
import small_gicp

from .radar import ImuData, RadarData, crop_radar_cloud, calc_radar_egovel
from .utils import ICloudTransformer, quat_to_rpy, rpy_to_quat, quat_conj, quat_mult, quat_to_mat, mat_to_quat, downsample_cloud
from .robot3d import RobotPose3D
from .ops import rot_scale_3d
from .model import GaussianModel
from .strapdown import Strapdown, pure_quat_exp, skewsym

TAU = float(2*np.pi)

class ImuRadarOdometry(Strapdown):
	def __init__(self, seed=3135134162, radar_to_imu=None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.ref_time = None
		self.imu_time = None
		self.imu_rp = (0.0,0.0)
		self.omega = torch.zeros(3, dtype=torch.float32, device='cuda')
		self.saved_quat = self.quat.cpu().numpy()

		if radar_to_imu is None:
			self.radar_to_imu = torch.eye(4, dtype=torch.float32, device='cuda')[0:3]
		else:
			self.radar_to_imu = torch.as_tensor(radar_to_imu, dtype=torch.float32, device='cuda')[0:3,0:4]

	@property
	def radar_rot(self) -> torch.Tensor:
		return self.radar_to_imu[:,0:3]

	@property
	def radar_arm(self) -> torch.Tensor:
		return self.radar_to_imu[:,3]

	@property
	def egovel(self) -> torch.Tensor:
		return super().egovel + skewsym(self.omega) @ self.radar_arm

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

	def process(self, bundle:RadarData) -> np.ndarray:
		self._process_imu(bundle)

		cl = crop_radar_cloud(bundle.scan)

		cl = np.copy(cl)
		cl[:,0:3] = (self.radar_rot.cpu().numpy() @ cl[:,0:3,None])[...,0]

		cl = self._process_egovel(cl, bundle.t)

		return cl

	def _process_imu(self, bundle:RadarData) -> None:
		if self.is_initial or len(bundle.imu) == 0:
			return

		self.saved_quat = self.quat.cpu().numpy()

		ag_cov = np.zeros((3,3), dtype=np.float32)
		ag_cov[0,0] = ag_cov[1,1] = ag_cov[2,2] = 1.0**2

		for imu in bundle.imu:
			tdiff = imu.t - self.imu_time
			assert tdiff >= 0.0
			self.imu_time = imu.t

			self.advance(tdiff, imu.accel, imu.accel_cov, imu.omega, imu.omega_cov)

		imu_roll,imu_pitch,_ = bundle.roll_pitch_g
		self.imu_rp = (imu_roll,imu_pitch)
		self.update_antigravity(bundle.mean_accel, ag_cov)

		self.omega = torch.as_tensor(bundle.mean_omega, dtype=torch.float32, device='cuda') - self.gyro_bias

	def _calc_egoframe(self) -> np.ndarray:
		return quat_to_mat(quat_mult(self.quat.cpu().numpy(), quat_conj(self.saved_quat)))

	def _process_egovel(self, cl:np.ndarray, t:float, force_forward=True) -> np.ndarray:
		if force_forward:
			R = self._calc_egoframe()
			egoframe = np.eye(cl.shape[1], dtype=np.float32)
			egoframe[0:3,0:3] = R

		try:
			egovel, inliers = calc_radar_egovel(cl @ egoframe if force_forward else cl, force_forward=force_forward)
		except:
			print('  {WARN} Egovel extraction fail')
			return cl

		if force_forward:
			egovel.vel = R @ egovel.vel
			egovel.cov = R @ egovel.cov @ R.T

		if self.is_initial:
			self.initialize(t, egovel.vel, egovel.cov)
		else:
			self.update_egovel(egovel.vel, egovel.cov, self.radar_arm)

		if inliers is not None:
			cl = cl[inliers]

		return cl

	def update_pose_wrapper(self, pose:RobotPose3D, xyzstd:float=1.0, rotstd:float=6.0, restrict_3dof=True) -> None:
		cov = torch.zeros((6,6), dtype=torch.float32, device='cuda')
		cov[0:3,0:3].fill_diagonal_(xyzstd**2)
		cov[3:6,3:6].fill_diagonal_((rotstd*TAU/360)**2)
		self.update_scanmatch(pose, cov, [0,1,5] if restrict_3dof else None)

class ImuRadarGicpOdometry(ImuRadarOdometry):
	def __init__(self, voxel_size=0.25, *args, **kwargs):
		super().__init__(*args, want_yaw_gyro_bias=True, **kwargs)

		self.voxel_size = voxel_size
		self.kf_cl    :small_gicp.PointCloud = None
		self.kf_tree  :small_gicp.KdTree     = None
		self.last_cl  :small_gicp.PointCloud = None
		self.last_tree:small_gicp.KdTree     = None

	@property
	def has_empty_keyframe(self) -> bool:
		return self.kf_cl is None

	@property
	def match_time(self) -> float:
		return self.imu_time - self.mtime

	def keyframe(self) -> None:
		super().keyframe()
		self.kf_cl   = self.last_cl
		self.kf_tree = self.kf_tree
		self.mtime   = self.imu_time

	def process(self, bundle:RadarData) -> np.ndarray:
		cl = super().process(bundle)
		self._scan_matching(cl)
		return cl

	def _scan_matching(self, cl:np.ndarray) -> None:
		cur_cl, cur_tree = small_gicp.preprocess_points(cl[:,0:3].astype(np.float64), downsampling_resolution=self.voxel_size)
		self.last_cl   = cur_cl
		self.last_tree = cur_tree

		if self.has_empty_keyframe:
			return

		relpose = (self.pose - self.kf_pose).xfrm_4x4[0].cpu().numpy()
		result = small_gicp.align(self.kf_cl, cur_cl, self.kf_tree, relpose)
		result : small_gicp.RegistrationResult
		if not result.converged:
			print('  {WARN} Scan matching fail')
			return

		relpose = RobotPose3D.from_xfrm(result.T_target_source[None])
		self.update_pose_wrapper(relpose)

		self.mtime = self.imu_time

class ImuRadarGaussianOdometry(ImuRadarOdometry, ICloudTransformer):
	def __init__(self,
		voxel_size   :float=0.25,
		gaussian_size:int=20,
		fit_points   :int=10000,
		fit_thresh   :float=50.0,
		reg_points   :int=5000,
		num_particles:int=8,
		*args, **kwargs
	):
		super().__init__(*args, want_yaw_gyro_bias=True, **kwargs)

		self.voxel_size    = voxel_size
		self.gaussian_size = gaussian_size
		self.fit_points    = fit_points
		self.fit_thresh    = fit_thresh
		self.reg_points    = reg_points
		self.num_particles = num_particles

		self.rng = np.random.default_rng(seed=3135134162)

		self.kf_model :GaussianModel = None
		self.last_cl  :np.ndarray    = None

	@property
	def has_empty_keyframe(self) -> bool:
		return self.kf_model is None

	@property
	def match_time(self) -> float:
		return self.imu_time - self.mtime

	def keyframe(self) -> None:
		super().keyframe()

		cl = downsample_cloud(self.last_cl, self.fit_points, self.rng)
		num_gaussians = int(0.5 + len(cl)/self.gaussian_size)

		self.mtime    = self.imu_time
		self.kf_model = GaussianModel(max_clusters=num_gaussians)
		self.kf_model.add_cloud(cl)

	def process(self, bundle:RadarData) -> np.ndarray:
		cl = super().process(bundle)
		self._scan_matching(cl)
		return cl

	def _scan_matching(self, cl:np.ndarray) -> None:
		# Perform voxel grid sampling on the point cloud
		cl = small_gicp.voxelgrid_sampling(cl[:,0:3], self.voxel_size).points()
		cl = cl[:,0:3].astype(np.float32).copy()
		self.last_cl = cl

		if self.has_empty_keyframe:
			return

		self.particlemean = self.pose - self.kf_pose

		basecov = self.particle_keyframe_cov
		particlecov = torch.zeros_like(basecov)
		particlecov[0:3,0:3].fill_diagonal_(0.25**2)
		particlecov[3:6,3:6].fill_diagonal_((10*TAU/360)**2)
		self.set_particle_space(particlecov)

		cl = downsample_cloud(cl, self.reg_points, self.rng)

		swarm = self.rng.normal(size=(self.num_particles, 6))
		swarm = torch.as_tensor(swarm, dtype=torch.float32, device='cuda')

		best_particles, best_L, best_pid = self.kf_model.register(cl, swarm, self)
		L = float(best_L[best_pid])
		if L >= self.fit_thresh:
			print('  {WARN} Scan matching fail')
			return

		new_relpose = self.transform_as_pose(best_particles[None,best_pid])
		new_pose = self.kf_pose + new_relpose

		#print('  RELPOSE', self.particlemean.xyz_tran[0].cpu().numpy())
		#print('  OUTPOSE', new_relpose.xyz_tran[0].cpu().numpy())
		#print('  LOSS', L)
		self.update_pose_wrapper(new_pose)

		self.mtime = self.imu_time

	def transform(self, particles:torch.Tensor) -> torch.Tensor:
		return (self.covchol @ particles[...,0:6,None])[...,0]

	def transform_as_pose(self, particles:torch.Tensor) -> RobotPose3D:
		p = self.transform(particles)
		p_xyz = p[:,0:3]
		p_quat = pure_quat_exp(0.5*p[:,3:6])
		p_mat = rot_scale_3d(torch.ones(p_quat.shape[:-1] + (3,), dtype=torch.float32, device='cuda'), p_quat)
		return self.particlemean + RobotPose3D(xyz_tran=p_xyz, mat_rot=p_mat)

	# ICloudTransformer
	def transform_cloud(self, cloud:torch.Tensor, particles:torch.Tensor=None) -> torch.Tensor:
		poses = self.transform_as_pose(particles) if particles is not None else self.pose
		return poses.xyz_tran[:,None,:] + (poses.mat_rot[:,None,:] @ cloud[None,:,:,None])[...,0]
