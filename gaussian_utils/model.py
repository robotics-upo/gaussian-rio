import torch
import numpy as np

from sklearn.cluster import BisectingKMeans

from .ops import rot_scale_3d, nearest_center_3d, nearest_gaussian_3d, indexed_transform_3d
from .utils import GradientDescentParams, RequiresGrad, ICloudTransformer, quat_conj

from typing import Union, Tuple

class GaussianModel:
	def __init__(self,
		max_clusters:int=150,
		disc_thickness:float=0.15,
		mahal_thresh:float=2.0,
		fit_params:GradientDescentParams=GradientDescentParams(
			lr=0.05,
			eps=1e-15,
			max_epochs=1000,
			min_improvement=1e-4,
			patience=50
		),
		register_params:GradientDescentParams=GradientDescentParams(
			lr=0.1,
			eps=1e-15,
			max_epochs=200,
			min_improvement=0.0,
			patience=10
		)
	):
		self.max_clusters = max_clusters
		self.log_disc_thickness = float(np.log(disc_thickness))
		self.mahal2_thresh = mahal_thresh*mahal_thresh
		self._fit = fit_params
		self._reg = register_params
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

	def add_cloud(self, cloud:Union[np.ndarray,torch.Tensor]) -> None:
		if type(cloud) is np.ndarray:
			cl_to_bisect = cloud
			cloud = torch.as_tensor(cloud, dtype=torch.float32, device='cuda')
		elif type(cloud) is torch.Tensor:
			cl_to_bisect = cloud.cpu().numpy()
		else:
			raise TypeError('cloud must be np.ndarray or torch.Tensor')

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

		if True or self.is_empty:
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
			opt = torch.optim.Adam(rg.tensors, lr=self._fit.lr, eps=self._fit.eps)

			for epoch in range(self._fit.max_epochs):
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

				if best_epoch is None or best_loss > float(L) + self._fit.min_improvement:
					best_epoch = epoch
					best_loss = float(L)
					best_weights = (g_centers.detach().clone(), g_log_scales.detach().clone(), g_quats.detach().clone())
				elif (epoch - best_epoch) > self._fit.patience:
					#print('Early stopping')
					break

				opt.zero_grad()
				L.backward()
				opt.step()

		#print(f'Gaussian fitting ended, loss = {best_loss:.4f}')

		# Restore best weights
		g_centers[:] = best_weights[0]
		g_log_scales[:] = best_weights[1]
		g_quats[:] = -torch.nn.functional.normalize(best_weights[2])
		g_quats[:,0] *= -1

	def register(self, cloud:Union[np.ndarray,torch.Tensor], swarm:Union[np.ndarray,torch.Tensor], xfrm:ICloudTransformer) -> Tuple[torch.Tensor, torch.Tensor, int]:
		cloud = torch.as_tensor(cloud, dtype=torch.float32, device='cuda')
		swarm = torch.as_tensor(swarm, dtype=torch.float32, device='cuda')

		g_invmat = self.inverse_matrices

		best_epoch, best_swarm, best_L, best_overall_L, best_overall_pid = (None,)*5

		with RequiresGrad(swarm) as rg:
			opt = torch.optim.Adam(rg.tensors, lr=self._reg.lr, eps=self._reg.eps)

			for epoch in range(self._reg.max_epochs):
				cl_reg = xfrm.transform_cloud(cloud, swarm)

				cl_sqmahal, _ = nearest_gaussian_3d(cl_reg.reshape(-1,3), self.centers, g_invmat)
				cl_sqmahal = cl_sqmahal.reshape(swarm.shape[0], -1)

				L_batch = torch.mean(cl_sqmahal, dim=-1)
				L_batch_nograd = L_batch.detach()

				cur_L, cur_pid = torch.min(L_batch_nograd, dim=0)
				#print(f'  Epoch {1+epoch} loss={float(cur_L):.4f} particle={int(cur_pid)}')

				if best_epoch is None:
					best_epoch = epoch
					best_swarm = swarm.detach().clone()
					best_L = L_batch_nograd
					best_overall_L, best_overall_pid = torch.min(best_L, dim=0)
				elif torch.any(improvement := best_L > L_batch_nograd + self._reg.min_improvement):
					best_swarm = torch.where(improvement[:,None], swarm.detach().clone(), best_swarm)
					best_L = torch.where(improvement, L_batch_nograd, best_L)
					overall_L, overall_pid = torch.min(best_L, dim=0)
					if best_overall_L > overall_L + self._reg.min_improvement:
						best_epoch = epoch
						best_overall_L = overall_L
						best_overall_pid = overall_pid

				if (epoch - best_epoch) > self._reg.patience:
					#print('Early stopping')
					break

				opt.zero_grad()
				torch.sum(L_batch).backward()
				opt.step()

		return best_swarm, best_L, best_overall_pid
