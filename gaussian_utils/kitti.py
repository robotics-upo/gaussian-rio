import datetime
import numpy as np

def load_kitti_file(basedir, id, remove_floor=True):
	cl = np.fromfile(f'{basedir}/data/{id:010d}.bin', dtype=np.float32)
	cl = cl.reshape((-1,4))[:,0:3]
	if remove_floor:
		cl = cl[~(cl[:,2] <= -1.5),:]
	return cl

def load_kitti_timestamps(basedir, which='timestamps'):
	ts = []
	with open(f'{basedir}/{which}.txt', 'r', encoding='utf-8') as f:
		for line in f.readlines():
			ts.append(datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f').timestamp())
	return ts
