import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import struct
import random
import numpy as np
import torch

from gaussian_utils.model import GaussianModel

PIPE_PATH = '/tmp/gaussian_server'

u32 = struct.Struct('=I')
GaussianFitParams = struct.Struct('@Nddd')

try:
	os.mkfifo(PIPE_PATH+'_in')
	os.mkfifo(PIPE_PATH+'_out')
except FileExistsError:
	pass

FIXED_RANDOM_SEED = 3135134162
torch.use_deterministic_algorithms(True)

print('Ready to accept commands')

try:
	while True:
		with open(PIPE_PATH+'_in', "rb") as fin:
			with open(PIPE_PATH+'_out', "wb") as fout:
				while len(cmdid := fin.read(u32.size)) == u32.size:
					cmdid = u32.unpack(cmdid)[0]
					print('Command',cmdid)
					num_gaussians, initial_scale, min_scale, disc_thickness = GaussianFitParams.unpack_from(fin.read(40))
					num_points = u32.unpack(fin.read(u32.size))[0]
					cl = np.frombuffer(fin.read(num_points*3*4), dtype=np.float32).reshape((-1, 3)).copy()
					torch.manual_seed(FIXED_RANDOM_SEED)
					np.random.seed(FIXED_RANDOM_SEED)
					random.seed(FIXED_RANDOM_SEED)

					m = GaussianModel(
						max_clusters=num_gaussians,
						disc_thickness=disc_thickness,
						min_std=min_scale,
					)
					m.add_cloud(cl)
					centers = np.array(m.centers.cpu().numpy(), dtype=np.float64)
					log_scales = np.array(m.log_scales.cpu().numpy(), dtype=np.float64) + m.scale_baseline
					quats = np.array(wtf:=m.quats[:,(1,2,3,0)].cpu().numpy(), dtype=np.float64)

					fout.write(centers.tobytes())
					fout.write(log_scales.tobytes())
					fout.write(quats.tobytes())
					m = None
except KeyboardInterrupt:
	pass
finally:
	os.unlink(PIPE_PATH+'_in')
	os.unlink(PIPE_PATH+'_out')
