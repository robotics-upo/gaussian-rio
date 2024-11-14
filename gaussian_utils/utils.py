import datetime

import torch
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

TAU = float(2*np.pi)

@dataclass
class GradientDescentParams:
	lr             :float
	eps            :float
	max_epochs     :int
	min_improvement:float
	patience       :int

@dataclass
class RobotModelParams:
	init_vel_std :float
	accel_std    :float
	angvel_std   :float

@dataclass
class RobotModelParams3D:
	init_vel_std :float
	xy_accel_std :float
	z_accel_std  :float
	n_vel_std    :float
	angvel_std   :float

class RequiresGrad:
	def __init__(self, *tensors):
		self.tensors = tensors

	def __enter__(self):
		for t in self.tensors:
			t.requires_grad_(True)
		return self

	def __exit__(self, *_):
		for t in self.tensors:
			t.requires_grad_(False)

class ICloudTransformer(ABC):
	@abstractmethod
	def transform_cloud(self, cloud:torch.Tensor, particles:torch.Tensor) -> torch.Tensor:
		raise NotImplementedError()

def quat_conj(q: torch.Tensor):
	assert q.shape[-1] == 4
	ret = -q
	ret[...,0] = -ret[...,0]
	return ret

def mat_ortho(mat: torch.Tensor):
	# https://raw.githubusercontent.com/martius-lab/hitchhiking-rotations/main/hitchhiking_rotations/utils/conversions.py
	# Explanation: SVD sometimes gives out rotation matrices that have det=-1.
	# Flip the last vector if needed in order to fix it (left hand->right hand)
	U,S,Vt = torch.linalg.svd(mat)
	sign,_ = torch.linalg.slogdet(U @ Vt)
	return U @ torch.cat((Vt[..., :2, :], sign[...,None,None]*Vt[..., -1:, :]), dim=-2)

def mat_to_quat(mat: np.ndarray) -> np.ndarray:
	tx,ty,tz = mat[0,0],mat[1,1],mat[2,2]
	if tz <= 0:
		if tx >= ty:
			x = 1 + tx - ty - tz
			y = mat[1,0] + mat[0,1]
			z = mat[2,0] + mat[0,2]
			w = mat[2,1] - mat[1,2]
		else:
			x = mat[1,0] + mat[0,1]
			y = 1 - tx + ty - tz
			z = mat[2,1] + mat[1,2]
			w = mat[0,2] - mat[2,0]
	else:
		if -tx >= ty:
			x = mat[2,0] + mat[0,2]
			y = mat[2,1] + mat[1,2]
			z = 1 - tx - ty + tz
			w = mat[1,0] - mat[0,1]
		else:
			x = mat[2,1] - mat[1,2]
			y = mat[0,2] - mat[2,0]
			z = mat[1,0] - mat[0,1]
			w = 1 + tx + ty + tz
	q = np.stack([w,x,y,z], axis=-1)
	return q / np.linalg.norm(q)

def quat_to_mat(q: np.ndarray) -> np.ndarray:
	qxx = q[...,1] * q[...,1]
	qyy = q[...,2] * q[...,2]
	qzz = q[...,3] * q[...,3]
	qxz = q[...,1] * q[...,3]
	qxy = q[...,1] * q[...,2]
	qyz = q[...,2] * q[...,3]
	qwx = q[...,0] * q[...,1]
	qwy = q[...,0] * q[...,2]
	qwz = q[...,0] * q[...,3]

	mat = (
		[ 1 - 2*(qyy + qzz), 2*(qxy - qwz), 2*(qxz + qwy) ],
		[ 2*(qxy + qwz), 1 - 2*(qxx + qzz), 2*(qyz - qwx) ],
		[ 2*(qxz - qwy), 2*(qyz + qwx), 1 - 2*(qxx + qyy) ],
	)

	return np.stack(tuple(np.stack(row, axis=-1) for row in mat), axis=-2)

def mat_to_zrot_normal(mat: np.ndarray):
	xlate = mat[...,0:3,3]
	normals = mat[...,0:3,2]
	normals = normals[...,0:2] / normals[...,2,None]
	zrot = np.arctan2(mat[...,1,0], mat[...,0,0])
	return xlate, zrot, normals

def nzrot_to_mat(nzrot: torch.Tensor):
	z_axis = torch.nn.functional.pad(nzrot[...,0:2], (0, 1), value=1.0)[...,None,:]
	z_axis = z_axis / torch.linalg.norm(z_axis, dim=-1)[...,None]

	r_angle = nzrot[...,2]
	r_cos = torch.cos(r_angle)
	r_sin = -torch.sin(r_angle) # Transpose
	r_zero = torch.zeros_like(r_cos)
	x_axis = torch.stack([r_cos, r_sin, r_zero], dim=-1)[...,None,:]

	y_axis = torch.linalg.cross(z_axis, x_axis)
	x_axis = torch.linalg.cross(y_axis, z_axis)

	return torch.concat([x_axis, y_axis, z_axis], dim=-2)

def quat_to_z(quat: np.ndarray):
	qw,qx,qy,qz = quat[...,0], quat[...,1], quat[...,2], quat[...,3]
	siny_cosp = 2*(qw*qz + qx*qy)
	cosy_cosp = 1 - 2*(qy*qy + qz*qz)
	return np.arctan2(siny_cosp, cosy_cosp)

def quat_to_rpy(quat: np.ndarray):
	qw,qx,qy,qz = quat[...,0], quat[...,1], quat[...,2], quat[...,3]

	# roll (x-axis rotation)
	sinr_cosp = 2*(qw*qx + qy*qz)
	cosr_cosp = 1 - 2*(qx*qx + qy*qy)
	roll = np.arctan2(sinr_cosp, cosr_cosp)

	# pitch (y-axis rotation)
	sinp = np.sqrt(1 + 2*(qw*qy - qx*qz))
	cosp = np.sqrt(1 - 2*(qw*qy - qx*qz))
	pitch = 2*np.arctan2(sinp, cosp) - TAU/4

	# yaw (z-axis rotation)
	siny_cosp = 2*(qw*qz + qx*qy)
	cosy_cosp = 1 - 2*(qy*qy + qz*qz)
	yaw = np.arctan2(siny_cosp, cosy_cosp)

	return np.stack((roll,pitch,yaw), axis=-1)

def rpy_to_quat(rpy: np.ndarray):
	r,p,y = rpy[...,0]/2, rpy[...,1]/2, rpy[...,2]/2

	cosr,sinr = np.cos(r),np.sin(r)
	cosp,sinp = np.cos(p),np.sin(p)
	cosy,siny = np.cos(y),np.sin(y)

	return np.stack([
		cosr*cosp*cosy + sinr*sinp*siny,
		sinr*cosp*cosy - cosr*sinp*siny,
		cosr*sinp*cosy + sinr*cosp*siny,
		cosr*cosp*siny - sinr*sinp*cosy,
	], axis=-1)

def quat_mult(q0:np.ndarray, q1:np.ndarray) -> np.ndarray:
	w0,x0,y0,z0 = q0[...,0], q0[...,1], q0[...,2], q0[...,3]
	w1,x1,y1,z1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
	return np.stack([
		w0*w1 - x0*x1 - y0*y1 - z0*z1,
		w0*x1 + x0*w1 + y0*z1 - z0*y1,
		w0*y1 + y0*w1 + z0*x1 - x0*z1,
		w0*z1 + z0*w1 + x0*y1 - y0*x1,
	], axis=-1)

def quat_vec_mult(q:np.ndarray, v:np.ndarray):
	qv = q[...,1:4]
	qw = q[...,0,None]
	uv = np.cross(qv, v)
	uuv = np.cross(qv, uv)
	return v + 2*(qw*uv + uuv)

def pure_quat_exp(w: np.ndarray) -> np.ndarray:
	norm = np.linalg.norm(w, axis=-1)
	qw,sinc = np.cos(norm), np.sinc(norm / (0.5*TAU))
	qv = w*sinc[...,None]
	return np.stack([ qw, qv[...,0], qv[...,1], qv[...,2] ], axis=-1)

def skewsym(omega:np.ndarray) -> np.ndarray:
	x,y,z = omega[0],omega[1],omega[2]
	return np.asarray([
		[ 0.0, -z, +y ],
		[ +z, 0.0, -x ],
		[ -y, +x, 0.0 ],
	], dtype=omega.dtype)

def crop_cloud(cl: np.ndarray, mindist:float=1.5, maxdist:float=30.0):
	cldist = np.linalg.norm(cl[:,0:2], axis=1)
	return cl[(mindist <= cldist) & (cldist <= maxdist), :]

def downsample_cloud(cl: np.ndarray, num_points:int, rng:Union[np.random.Generator,int]=None):
	if len(cl) <= num_points: return cl
	if rng is None or type(rng) is int: rng = np.random.default_rng(seed=rng)
	return rng.choice(cl, num_points, replace=False)

def downsample_floor(cl: np.ndarray, z_plane:float = -1.25, floor_ratio = 0.5, rng:Union[np.random.Generator,int]=None) -> np.ndarray:
	is_floor = cl[:,2] <= z_plane
	num_floor_points = int(np.sum(is_floor))

	if num_floor_points == 0 or num_floor_points == len(cl):
		return cl

	if rng is None or type(rng) is int: rng = np.random.default_rng(seed=rng)

	desired_floor_points = int(0.5 + floor_ratio*(len(cl) - num_floor_points))

	floor_selection = rng.choice(cl[is_floor], desired_floor_points, replace=desired_floor_points > num_floor_points)

	return np.concatenate((cl[~is_floor], floor_selection), axis=0)
