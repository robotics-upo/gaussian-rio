import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from collections import namedtuple
from functools import lru_cache

from .model import GaussianModel
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
