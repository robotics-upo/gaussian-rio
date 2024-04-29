import datetime

import torch
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

@dataclass
class GradientDescentParams:
	lr             :float
	eps            :float
	max_epochs     :int
	min_improvement:float
	patience       :int

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

def mat_to_quat(mat: np.array):
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

def crop_cloud(cl: np.array, mindist:int=3, maxdist:int=15):
	cldist = np.linalg.norm(cl[:,0:2], axis=1)
	return cl[(mindist <= cldist) & (cldist <= maxdist), :]

def downsample_cloud(cl: np.array, num_points:int, rng:Union[np.random.Generator,int]=None):
	if len(cl) <= num_points: return cl
	if rng is None or type(rng) is int: rng = np.random.default_rng(seed=rng)
	return rng.choice(cl, num_points, replace=False)
