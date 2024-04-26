import torch
import numpy as np

from sklearn.cluster import BisectingKMeans

from .ops import rot_scale_3d, nearest_center_3d, nearest_gaussian_3d, indexed_transform_3d
from .utils import quat_conj

from typing import Union

class GaussianModel:
	def __init__(self,
		max_clusters:int=100,
		disc_thickness:float=0.15,
		mahal_thresh:float=2.0,
		fit_lr:float=0.05,
		fit_eps:float=1e-15,
		fit_max_epochs:int=1000,
		fit_min_improvement:float=1e-4,
		fit_patience:int=50
	):
		self.max_clusters = max_clusters
		self.log_disc_thickness = float(np.log(disc_thickness))
		self.mahal2_thresh = mahal_thresh*mahal_thresh
		self.fit_lr = fit_lr
		self.fit_eps = fit_eps
		self.fit_max_epochs = fit_max_epochs
		self.fit_min_improvement = fit_min_improvement
		self.fit_patience = fit_patience
		self.clear()

	@property
	def is_empty(self) -> bool:
		return self.centers is None

	def clear(self) -> None:
		self.centers = None
		self.log_scales = None
		self.quats = None

	@property
	def scales(self) -> torch.Tensor:
		return torch.exp(self.log_scales)

	@property
	def inverse_scales(self) -> torch.Tensor:
		return torch.exp(-self.log_scales)

	@property
	def inverse_quats(self) -> torch.Tensor:
		return quat_conj(self.quats)

	@property
	def matrices(self) -> torch.Tensor:
		return rot_scale_3d(self.scales, self.quats, scale_first=True)

	@property
	def inverse_matrices(self) -> torch.Tensor:
		return rot_scale_3d(self.inverse_scales, self.inverse_quats, scale_first=False)

	def add_cloud(self, cloud:Union[np.array,torch.Tensor]) -> None:
		if type(cloud) is np.array:
			cl_to_bisect = cloud
			cloud = torch.as_tensor(cloud, dtype=torch.float32, device='cuda')
		elif type(cloud) is torch.Tensor:
			cl_to_bisect = cloud.cpu().numpy()
		else:
			raise TypeError('cloud must be np.array or torch.Tensor')

		new_centers = BisectingKMeans(n_clusters=self.max_clusters, random_state=3135134162).fit(cl_to_bisect).cluster_centers_
		new_centers = torch.as_tensor(new_centers, dtype=torch.float32, device='cuda')

		def create_log_scales(N):
			return torch.zeros((N, 3), dtype=torch.float32, device='cuda')

		def create_quats(N):
			q = torch.zeros((N, 4), dtype=torch.float32, device='cuda')
			q[:,0] = 1
			return q

		if not self.is_empty:
			mahal2, idx = nearest_gaussian_3d(new_centers, self.centers, self.inverse_matrices)
			subsumed = mahal2 <= self.mahal2_thresh

			gaussians_to_retrain = torch.unique(idx[subsumed])
			num_new_gaussians = int(torch.sum(~subsumed))
			print('Adding', num_new_gaussians, 'new Gaussians')

			if num_new_gaussians < self.max_clusters:
				gaussians_to_keep = torch.ones((len(self.centers),), dtype=torch.bool, device='cuda')
				gaussians_to_keep[gaussians_to_retrain] = 0
			else:
				self.clear()

		if self.is_empty:
			train_centers    = new_centers
			train_log_scales = create_log_scales(self.max_clusters)
			train_quats      = create_quats(self.max_clusters)
		else:
			train_centers    = torch.cat([ self.centers[gaussians_to_retrain],    new_centers[~subsumed]               ], dim=0)
			train_log_scales = torch.cat([ self.log_scales[gaussians_to_retrain], create_log_scales(num_new_gaussians) ], dim=0)
			train_quats      = torch.cat([ self.quats[gaussians_to_retrain],      create_quats(num_new_gaussians)      ], dim=0)

		self._fit_gaussians(cloud, train_centers, train_log_scales, train_quats)

		if self.is_empty:
			self.centers    = train_centers
			self.log_scales = train_log_scales
			self.quats      = train_quats
		else:
			self.centers    = torch.cat([ self.centers[gaussians_to_keep],    train_centers    ], dim=0)
			self.log_scales = torch.cat([ self.log_scales[gaussians_to_keep], train_log_scales ], dim=0)
			self.quats      = torch.cat([ self.quats[gaussians_to_keep],      train_quats      ], dim=0)

	def _fit_gaussians(self, cloud:torch.Tensor, g_centers:torch.Tensor, g_log_scales:torch.Tensor, g_quats:torch.Tensor):
		g_quats *= -1
		g_quats[:,0] *= -1

		best_weights = None
		best_loss = None
		best_epoch = None

		with RequiresGrad(g_centers, g_log_scales, g_quats) as rg:
			opt = torch.optim.Adam(rg.tensors, lr=self.fit_lr, eps=self.fit_eps)

			for epoch in range(self.fit_max_epochs):
				# Parametrize
				g_invmats = rot_scale_3d(torch.exp(-g_log_scales), torch.nn.functional.normalize(g_quats), scale_first=False)

				# Calculate vectors from every gaussian center to every point
				cl_tran, cl_gidx = nearest_center_3d(cloud, g_centers)
				cl_xfrm = indexed_transform_3d(cl_tran, g_invmats, cl_gidx)
				cl_mahal2 = (cl_xfrm*cl_xfrm).sum(dim=-1)
				#cl_mahal2, cl_gidx = nearest_gaussian_3d(cloud, g_centers, g_invmats)

				g_mahal2 = torch.zeros((len(g_centers),), dtype=torch.float32, device='cuda')
				g_mahal2.scatter_reduce_(0, cl_gidx, cl_mahal2, reduce='mean')

				L1 = 0.5*g_mahal2 + torch.sum(g_log_scales, dim=1)
				L2 = torch.nn.functional.relu(torch.min(g_log_scales, axis=1)[0] - self.log_disc_thickness)
				L = torch.mean(L1 + L2)

				#print(f'Epoch {1+epoch} L={float(L):.4f}')

				opt.zero_grad()
				L.backward()
				opt.step()

				if best_epoch is None or best_loss > float(L) + self.fit_min_improvement:
					best_epoch = epoch
					best_loss = float(L)
					best_weights = (g_centers.detach().clone(), g_log_scales.detach().clone(), g_quats.detach().clone())
				elif (epoch - best_epoch) > self.fit_patience:
					#print('Early stopping')
					break

		print(f'Gaussian fitting ended, loss = {best_loss:.4f}')

		# Restore best weights
		g_centers[:] = best_weights[0]
		g_log_scales[:] = best_weights[1]
		g_quats[:] = -torch.nn.functional.normalize(best_weights[2])
		g_quats[:,0] *= -1

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
