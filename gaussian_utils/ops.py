import os

import torch
import torch.utils.cpp_extension as cpp

MY_PATH = os.path.dirname(os.path.abspath(__file__))

_C = cpp.load('gaussian_utils_cpp',
	[
		f'{MY_PATH}/gaussian_utils.cpp',
		f'{MY_PATH}/rotscale3d.cu',
		f'{MY_PATH}/nearest3d.cu',
	],
	extra_cflags=[ '-std=gnu++17', '-O2', '-Wall', f'-I{MY_PATH}' ],
	extra_cuda_cflags=[ '-O2', f'-I{MY_PATH}' ],
)

#------------------------------------------------------------------------------

class _RotScale3d(torch.autograd.Function):
	@staticmethod
	def forward(scales, quats, scale_first):
		return _C.rot_scale_3d_forward(scales, quats, scale_first)

	@staticmethod
	def setup_context(ctx, inputs, outputs):
		scales, quats, scale_first = inputs
		ctx.scale_first = scale_first
		ctx.save_for_backward(scales, quats)

	@staticmethod
	def backward(ctx, grad_output):
		scales, quats = ctx.saved_tensors
		return _C.rot_scale_3d_backward(grad_output.contiguous(), scales, quats, ctx.scale_first) + (None,)

def rot_scale_3d(scales: torch.Tensor, quats: torch.Tensor, scale_first: bool = True) -> torch.Tensor:
	return _RotScale3d.apply(scales, quats, scale_first)

def rot_scale_3d_norm(scales: torch.Tensor, quats: torch.Tensor, scale_first: bool = True) -> torch.Tensor:
	return rot_scale_3d(torch.exp(scales), torch.nn.functional.normalize(quats), scale_first)

#------------------------------------------------------------------------------

class _NearestCenter3d(torch.autograd.Function):
	@staticmethod
	def forward(points, centers):
		return _C.nearest_center_3d_forward(points, centers)

	@staticmethod
	def setup_context(ctx:torch.autograd.function.FunctionCtx, inputs, outputs):
		points, centers = inputs
		vectors, indices = outputs
		ctx.N_centers = len(centers)
		ctx.save_for_backward(indices)
		ctx.mark_non_differentiable(indices)

	@staticmethod
	def backward(ctx, grad_vectors, grad_indices):
		(indices,) = ctx.saved_tensors
		return _C.nearest_center_3d_backward(ctx.N_centers, grad_vectors, indices)

def nearest_center_3d(points: torch.Tensor, centers: torch.Tensor):
	return _NearestCenter3d.apply(points, centers)

#------------------------------------------------------------------------------

class _NearestGaussian3d(torch.autograd.Function):
	@staticmethod
	def forward(points, centers, matrices):
		return _C.nearest_gaussian_3d_forward(points, centers, matrices)

	@staticmethod
	def setup_context(ctx:torch.autograd.function.FunctionCtx, inputs, outputs):
		points, centers, matrices = inputs
		sqmahal, indices = outputs
		ctx.save_for_backward(points, centers, matrices, indices)
		ctx.mark_non_differentiable(indices)

	@staticmethod
	def backward(ctx, grad_sqmahal, grad_indices):
		points, centers, matrices, indices = ctx.saved_tensors
		if centers.requires_grad or matrices.requires_grad:
			return _C.nearest_gaussian_3d_backward(grad_sqmahal, points, centers, matrices, indices)
		else:
			return _C.nearest_gaussian_3d_backward_lite(grad_sqmahal, points, centers, matrices, indices), None, None

def nearest_gaussian_3d(points: torch.Tensor, centers: torch.Tensor, matrices: torch.Tensor):
	return _NearestGaussian3d.apply(points, centers, matrices)

#------------------------------------------------------------------------------

class _IndexedTransform3d(torch.autograd.Function):
	@staticmethod
	def forward(points, matrices, indices):
		return _C.indexed_transform_3d_forward(points, matrices, indices)

	@staticmethod
	def setup_context(ctx:torch.autograd.function.FunctionCtx, inputs, outputs):
		points,matrices,indices = inputs
		ctx.save_for_backward(points,matrices,indices)

	@staticmethod
	def backward(ctx, grad):
		points,matrices,indices = ctx.saved_tensors
		return _C.indexed_transform_3d_backward(grad,points,matrices,indices) + (None,)

def indexed_transform_3d(points: torch.Tensor, matrices: torch.Tensor, indices: torch.Tensor):
	return _IndexedTransform3d.apply(points, matrices, indices)
