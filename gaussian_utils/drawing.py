import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from collections import namedtuple
from functools import lru_cache

from .model import GaussianModel
from .utils import quat_to_rpy
from .robot3d import RobotPose3D

TAU = float(2*np.pi)

# Default backend (TkAgg) causes ridiculous memory leaks. Switch to non-interactive backend instead
matplotlib.use('Agg')

# https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
class Arrow3D(FancyArrowPatch):
	def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
		super().__init__((0, 0), (0, 0), *args, **kwargs)
		self._xyz = (x, y, z)
		self._dxdydz = (dx, dy, dz)

	def draw(self, renderer):
		x1, y1, z1 = self._xyz
		dx, dy, dz = self._dxdydz
		x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

		xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
		self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
		super().draw(renderer)

	def do_3d_projection(self, renderer=None):
		x1, y1, z1 = self._xyz
		dx, dy, dz = self._dxdydz
		x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

		xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
		self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

		return np.min(zs)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
	'''Add an 3d arrow to an `Axes3D` instance.'''

	arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
	ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

@lru_cache(maxsize=None)
def _calc_sphere_coords():
	u, v = np.mgrid[0:TAU:10j, 0:TAU/2:10j]
	x = np.cos(u) * np.sin(v)
	y = np.sin(u) * np.sin(v)
	z = np.cos(v)
	return np.stack([x,y,z], axis=-1)*2

def _spheroid(ax, pos, mat, *args, **kwargs):
	whatever = (mat @ _calc_sphere_coords()[...,None])[...,0] + pos
	x,y,z = whatever[...,0], whatever[...,1], whatever[...,2]
	ax.plot_surface(x, y, z, *args, **kwargs)

setattr(Axes3D, 'spheroid', _spheroid)

def visualize(
	g      :GaussianModel,
	pt     :np.ndarray,
	im     :np.ndarray,
	pose   :RobotPose3D=None,
	title  :str=None,
	outfile:str=None,
	oldpt  :np.ndarray=None,
	traj   :np.ndarray=None,
	gt_traj:np.ndarray=None,
	gt_idx :int=-1
):
	mat = g.matrices.cpu().numpy()

	if outfile is None:
		matplotlib.use('GTK3Agg')

	fig = plt.figure(figsize=(14,9))
	ax = fig.add_axes((0., 2/9., 0.5, 7./9.), projection='3d')
	ax2 = fig.add_axes((0., 0.1/9., 1., 2./9.))
	ax2.set_xticks([])
	ax2.set_yticks([])
	ax2.imshow(im, zorder=1.0)

	if gt_traj is not None or traj is not None:
		ax3 = fig.add_axes((0.5+0.07, 2/9., 0.5-0.07*2, 7./9.))
		ax3.set_aspect('equal')
		ax3.grid()
		if gt_traj is not None:
			ax3.plot(gt_traj[:,0], gt_traj[:,1], color='green')
			if gt_idx >= 0:
				ax3.plot(gt_traj[gt_idx,0], gt_traj[gt_idx,1], marker='*', markersize=7.5, color='green')
		if traj is not None:
			ax3.plot(traj[:,0], traj[:,1], color='red')

	if oldpt is not None:
		ax.scatter(oldpt[:,0], oldpt[:,1], oldpt[:,2], c='gray', s=0.25)
	ax.scatter(pt[:,0], pt[:,1], pt[:,2], c='black', s=0.25)

	for i in range(len(g.centers)):
		ax.spheroid(g.centers[i].cpu().numpy(), mat[i], cmap=plt.cm.YlGnBu_r)

	if pose is not None:
		tx,ty,tz = pose.xyz_tran[0].cpu().numpy()
		dx,dy,dz = pose.mat_rot[0].cpu().numpy() @ [1.0,0.0,0.0]
		ax.arrow3D(tx, ty, tz, 5*dx, 5*dy, 5*dz, mutation_scale=20, ec='green', fc='red')
		ax3.plot(tx, ty, marker='*', markersize=7.5, color='red')

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	ptmin = np.min(pt, axis=0)
	ptmax = np.max(pt, axis=0)
	ptscale = 0.5*(ptmax-ptmin)
	ptcenter = ptmin+ptscale
	ptscale = np.max(ptscale)

	ax.set_xlim3d(ptcenter[0]-ptscale, ptcenter[0]+ptscale)
	ax.set_ylim3d(ptcenter[1]-ptscale, ptcenter[1]+ptscale)
	ax.set_zlim3d(ptcenter[2]-ptscale, ptcenter[2]+ptscale)

	if title is not None: fig.suptitle(title)
	if outfile is None:
		plt.show()
	else:
		plt.savefig(outfile)

	plt.close('all')

def visualize_odom(
	pred_t   :np.ndarray,
	gt_t     :np.ndarray,
	pred_pos :np.ndarray,
	gt_pos   :np.ndarray,
	pred_rot :np.ndarray,
	gt_rot   :np.ndarray = None,
	imu_rp   :np.ndarray = None,
	imu_t    :np.ndarray = None,
	egovel   :np.ndarray = None,
	aclr_bias:np.ndarray = None,
	gyro_bias:np.ndarray = None,
	title    :str=None,
	outfile  :str=None
):
	pred_rot = quat_to_rpy(pred_rot) * 360/TAU
	if gt_rot is not None:
		gt_rot = quat_to_rpy(gt_rot) * 360/TAU
	if imu_rp is not None:
		imu_rp = imu_rp * 360/TAU
	if gyro_bias is not None:
		gyro_bias = gyro_bias * 360/TAU
	if imu_t is None:
		imu_t = pred_t

	min_t = min(np.min(pred_t), np.min(gt_t))
	max_t = max(np.max(pred_t), np.max(gt_t))

	min_x = min(np.min(pred_pos[:,0]), np.min(gt_pos[:,0]))
	max_x = max(np.max(pred_pos[:,0]), np.max(gt_pos[:,0]))
	min_y = min(np.min(pred_pos[:,1]), np.min(gt_pos[:,1]))
	max_y = max(np.max(pred_pos[:,1]), np.max(gt_pos[:,1]))
	min_z = min(np.min(pred_pos[:,2]), np.min(gt_pos[:,2]))
	max_z = max(np.max(pred_pos[:,2]), np.max(gt_pos[:,2]))

	min_roll  = np.min(pred_rot[:,0])
	max_roll  = np.max(pred_rot[:,0])
	min_pitch = np.min(pred_rot[:,1])
	max_pitch = np.max(pred_rot[:,1])
	min_yaw   = np.min(pred_rot[:,2])
	max_yaw   = np.max(pred_rot[:,2])

	if gt_rot is not None:
		min_roll  = min(min_roll,  np.min(gt_rot[:,0]))
		max_roll  = max(max_roll,  np.max(gt_rot[:,0]))
		min_pitch = min(min_pitch, np.min(gt_rot[:,1]))
		max_pitch = max(max_pitch, np.max(gt_rot[:,1]))
		min_yaw   = min(min_yaw,   np.min(gt_rot[:,2]))
		max_yaw   = max(max_yaw,   np.max(gt_rot[:,2]))

	if imu_rp is not None:
		min_roll  = min(min_roll,  np.min(imu_rp[:,0]))
		max_roll  = max(max_roll,  np.max(imu_rp[:,0]))
		min_pitch = min(min_pitch, np.min(imu_rp[:,1]))
		max_pitch = max(max_pitch, np.max(imu_rp[:,1]))

	#--------------------------------------------------------------------------

	if outfile is None:
		matplotlib.use('GTK3Agg')

	fig = plt.figure(figsize=(15.0, 10.0))

	axes = fig.subplot_mosaic([
		[ 'rot_roll',  'egovel',    'pos_xy' ],
		[ 'rot_pitch', 'aclr_bias', 'pos_xy' ],
		[ 'rot_yaw',   'gyro_bias', 'pos_z'  ],
	])

	ax = axes['pos_z']
	ax.grid()
	ax.plot(pred_t, pred_pos[:,2], label='pred')
	ax.plot(gt_t,   gt_pos[:,2],   label='gt')
	ax.legend()
	ax.set_xlim(min_t, max_t)
	ax.set_xlabel('Time (s)', loc='right', labelpad=0)
	ax.set_ylim(min_z, max_z)
	ax.set_ylabel('Z position (m)', loc='top', labelpad=0)

	ax = axes['rot_roll']
	ax.grid()
	ax.plot(pred_t, pred_rot[:,0], label='pred')
	if gt_rot is not None:
		ax.plot(gt_t, gt_rot[:,0], label='gt')
	if imu_rp is not None:
		ax.plot(imu_t, imu_rp[:,0], label='imu', zorder=0)
	ax.legend()
	ax.set_xlim(min_t, max_t)
	ax.set_xlabel('Time (s)', loc='right', labelpad=0)
	ax.set_ylim(min_roll, max_roll)
	ax.set_ylabel('Roll rotation (º)', loc='top', labelpad=0)

	ax = axes['rot_pitch']
	ax.grid()
	ax.plot(pred_t, pred_rot[:,1], label='pred')
	if gt_rot is not None:
		ax.plot(gt_t, gt_rot[:,1], label='gt')
	if imu_rp is not None:
		ax.plot(imu_t, imu_rp[:,1], label='imu', zorder=0)
	ax.legend()
	ax.set_xlim(min_t, max_t)
	ax.set_xlabel('Time (s)', loc='right', labelpad=0)
	ax.set_ylim(min_pitch, max_pitch)
	ax.set_ylabel('Pitch rotation (º)', loc='top', labelpad=0)

	ax = axes['rot_yaw']
	ax.grid()
	ax.plot(pred_t, pred_rot[:,2], label='pred')
	if gt_rot is not None:
		ax.plot(gt_t, gt_rot[:,2], label='gt')
	ax.legend()
	ax.set_xlim(min_t, max_t)
	ax.set_xlabel('Time (s)', loc='right', labelpad=0)
	ax.set_ylim(min_yaw, max_yaw)
	ax.set_ylabel('Yaw rotation (º)', loc='top', labelpad=0)

	if egovel is not None:
		ax = axes['egovel']
		ax.grid()
		ax.plot(pred_t, egovel[:,0], label='x')
		ax.plot(pred_t, egovel[:,1], label='y')
		ax.plot(pred_t, egovel[:,2], label='z')
		ax.legend()
		ax.set_xlim(min_t, max_t)
		ax.set_xlabel('Time (s)', loc='right', labelpad=0)
		ax.set_ylim(np.min(egovel), np.max(egovel))
		ax.set_ylabel('Egovelocity (m/s)', loc='top', labelpad=0)

	if aclr_bias is not None:
		ax = axes['aclr_bias']
		ax.grid()
		ax.plot(pred_t, aclr_bias[:,0], label='x')
		ax.plot(pred_t, aclr_bias[:,1], label='y')
		ax.plot(pred_t, aclr_bias[:,2], label='z')
		ax.legend()
		ax.set_xlim(min_t, max_t)
		ax.set_xlabel('Time (s)', loc='right', labelpad=0)
		ax.set_ylim(np.min(aclr_bias), np.max(aclr_bias))
		ax.set_ylabel('Accelerometer bias (m/s²)', loc='top', labelpad=0)

	if gyro_bias is not None:
		ax = axes['gyro_bias']
		ax.grid()
		ax.plot(pred_t, gyro_bias[:,0], label='roll')
		ax.plot(pred_t, gyro_bias[:,1], label='pitch')
		ax.plot(pred_t, gyro_bias[:,2], label='yaw')
		ax.legend()
		ax.set_xlim(min_t, max_t)
		ax.set_xlabel('Time (s)', loc='right', labelpad=0)
		ax.set_ylim(np.min(gyro_bias), np.max(gyro_bias))
		ax.set_ylabel('Gyroscope bias (º/s)', loc='top', labelpad=0)

	ax = axes['pos_xy']
	ax.grid()
	ax.set_aspect('equal')
	ax.plot(gt_pos[:,1], gt_pos[:,0], label='gt')
	ax.plot(pred_pos[:,1], pred_pos[:,0], label='pred')
	ax.legend()
	ax.set_xlim(min_y-5, max_y+5)
	ax.set_xlabel('Y position (m)', loc='right', labelpad=0)
	ax.invert_xaxis()
	ax.set_ylim(min_x-5, max_x+5)
	ax.set_ylabel('X position (m)', loc='top', labelpad=0)

	fig.tight_layout(pad=1.)

	if title is not None: fig.suptitle(title)
	if outfile is None:
		plt.show()
	else:
		plt.savefig(outfile)

	plt.close('all')

def visualize_error(
	error  :np.ndarray,
	title  :str=None,
	outfile:str=None
):
	if outfile is None:
		matplotlib.use('GTK3Agg')

	fig = plt.figure(figsize=(15.0, 10.0))

	axes = fig.subplot_mosaic([
		lbl_pos := [ 'X position error', 'Y position error', 'Z position error' ],
		lbl_rot := [ 'X rotation error', 'Y rotation error', 'Z rotation error' ],
	])

	for i,which in enumerate(lbl_pos):
		ax = axes[which]
		ax.grid()
		ax.scatter(error[:,0], error[:,1+i], s=2.0)
		ax.set_xlabel('Distance travelled (m)', loc='right', labelpad=0)
		ax.set_ylabel(which + ' (m)', loc='top', labelpad=0)

	for i,which in enumerate(lbl_rot):
		ax = axes[which]
		ax.grid()
		ax.scatter(error[:,0], error[:,4+i]*360/TAU, s=2.0)
		ax.set_xlabel('Distance travelled (m)', loc='right', labelpad=0)
		ax.set_ylabel(which + ' (º)', loc='top', labelpad=0)

	fig.tight_layout(pad=1.)

	if title is not None: fig.suptitle(title)
	if outfile is None:
		plt.show()
	else:
		plt.savefig(outfile)

	plt.close('all')
