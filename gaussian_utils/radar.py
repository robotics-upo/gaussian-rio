import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
try:
	from typing import Self
except:
	from typing_extensions import Self

TAU = float(2*np.pi)

@dataclass
class ImuData:
	t        :float
	accel    :np.ndarray
	accel_cov:np.ndarray
	omega    :np.ndarray
	omega_cov:np.ndarray

@dataclass
class RadarData:
	t    : float
	scan : np.ndarray
	imu  : List[ImuData]
	img  : np.ndarray

	@property
	def roll_pitch_g(self) -> Tuple[float,float,float]:
		acc = np.stack([ x.accel for x in self.imu ], axis=0)
		roll = np.arctan2(acc[:,1], acc[:,2])
		pitch = np.arctan2(-acc[:,0], np.sqrt(acc[:,1]**2 + acc[:,2]**2))
		g = np.linalg.norm(acc, axis=1)

		return np.mean(roll), np.mean(pitch), np.mean(g)

@dataclass
class EgoVelocity:
	vel : np.ndarray
	cov : np.ndarray

	@staticmethod
	def dummy(std:float=0.02) -> Self:
		vel = np.zeros((3,),  dtype=np.float32)
		cov = np.zeros((3,3), dtype=np.float32)
		np.fill_diagonal(cov, std**2)
		return EgoVelocity(vel=vel, cov=cov)

def _solve_lsq(dirs, dops):
	return np.linalg.pinv(dirs, rcond=1e-4) @ dops

def _solve_egovel_ransac(dirs, dops, n_points=5, n_iters=None, p_outlier=0.05, p_success=0.995, inlier_thresh=0.3, recalc_with_inliers=True, sigma_offset=0.0):
	if n_iters is None:
		n_iters = int(0.5 + np.log(1.0 - p_success) / np.log(1.0 - (1.0-p_outlier)**n_points))

	pop_size = len(dirs)
	assert pop_size >= n_points

	rng = np.random.default_rng(seed=3135134162)

	all_idx = np.array(range(pop_size), np.int32)

	best_vel = None
	best_err = None
	best_inliers = None
	best_inlier_count = 0

	for it in range(n_iters):
		cursel = rng.choice(all_idx, n_points, replace=False, shuffle=False)
		try:
			curvel = _solve_lsq(dirs[cursel], dops[cursel])
		except:
			continue

		curerr = dirs @ curvel - dops

		curinliers = np.abs(curerr) < inlier_thresh
		curinliercount = int(np.sum(curinliers))

		#print('With',-curvel,curinliercount,'out of',pop_size,'are inliers')

		if curinliercount > best_inlier_count:
			best_vel = curvel
			best_err = curerr[curinliers]
			best_inliers = curinliers
			best_inlier_count = curinliercount

		if curinliercount == pop_size:
			#print('Already saturated')
			break

	if best_vel is None:
		raise Exception("What happened")

	#print(best_inlier_count,'out of',pop_size,'are inliers')

	if recalc_with_inliers:
		#print('Recalculating with all inliers')
		best_vel = _solve_lsq(dirs[best_inliers], dops[best_inliers])
		best_err = (dirs @ best_vel - dops)[best_inliers]

	# Calculate cov matrix
	H = dirs[best_inliers]
	cov = np.linalg.inv(H.T @ H)
	factor = (best_err.T @ best_err) / (len(best_err) - 3)

	if sigma_offset > 0.0:
		eigval,eigvec = np.linalg.eigh(cov)

		cov = np.zeros_like(cov)
		np.fill_diagonal(cov, factor*eigval + sigma_offset**2)

		cov = eigvec @ cov @ eigvec.T
	else:
		cov *= factor

	# Flip sign of velocity in order to refer to ourselves moving
	return -best_vel, cov, best_inliers

def crop_radar_cloud(cl:np.ndarray, min_power:float=10.0, max_azimuth:float=56.5, max_elevation:float=22.5) -> np.ndarray:
	max_azimuth = max_azimuth*TAU/360
	max_elevation = max_elevation*TAU/360
	azimuth = np.arctan2(cl[:,1], cl[:,0])
	elevation = np.arctan2(cl[:,2], np.linalg.norm(cl[:,0:2], axis=-1))
	filter = (cl[:,3] >= min_power) & (np.abs(azimuth) <= max_azimuth) & (np.abs(elevation) <= max_elevation)
	return cl[filter]

def calc_radar_egovel(cl:np.ndarray, median_thresh:float=0.05, **kwargs) -> Tuple[EgoVelocity, np.ndarray]:
	# Check median
	m_doppler = np.median(np.abs(cl[:,4]))
	if m_doppler < median_thresh:
		return EgoVelocity.dummy(), None

	direction = cl[:,0:3] / np.linalg.norm(cl[:,0:3], axis=-1, keepdims=True)
	vel, cov, inliers = _solve_egovel_ransac(direction, cl[:,4], **kwargs)

	return EgoVelocity(vel=vel, cov=cov), inliers
