import datetime
import numpy as np
import cv2

from .utils import mat_to_quat

def load_kitti_cam(basedir, id, which=2):
	return cv2.imread(f'{basedir}/../image_{which:02d}/data/{id:010d}.png')[...,::-1]

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

def load_kitti_odom(basedir, seqid, id, remove_floor=False):
	cl = np.fromfile(f'{basedir}/sequences/{seqid:02d}/velodyne/{id:06d}.bin', dtype=np.float32)
	cl = cl.reshape((-1,4))[:,0:3]
	if remove_floor:
		cl = cl[~(cl[:,2] <= -1.25),:]
	return cl

def load_kitti_odom_cam(basedir, seqid, id, which=2):
	return cv2.imread(f'{basedir}/sequences/{seqid:02d}/image_{which:d}/{id:06d}.png')[...,::-1]

def load_kitti_odom_ts(basedir, seqid):
	ts = []
	with open(f'{basedir}/sequences/{seqid:02d}/times.txt', 'r', encoding='utf-8') as f:
		for line in f.readlines():
			ts.append(float(line))
	return ts

def load_kitti_odom_calib(basedir, seqid):
	calib = {}
	with open(f'{basedir}/sequences/{seqid:02d}/calib.txt', 'r', encoding='utf-8') as f:
		for line in f.readlines():
			key,value = line.split(':',1)
			value = np.array([ float(x) for x in value.split() ] + [0,0,0,1], dtype=np.float32).reshape(4,4)
			calib[key] = value
	return calib

def load_kitti_odom_gt(basedir, seqid, T_cam0_velodyne:np.ndarray=None, basescan:int=None, raw_matrices:bool=False):
	# Load poses and expand 3x4 to 4x4
	mat = np.loadtxt(f'{basedir}/poses/{seqid:02d}.txt', dtype=np.float32).reshape((-1, 3, 4))
	lastrow = np.array([0,0,0,1], dtype=np.float32)[None,None,:].repeat(len(mat), axis=0)
	mat = np.concatenate([mat, lastrow], axis=1)

	# KITTI ground truth poses convert from cam0 (left) to world.
	# We may want to convert from velodyne to world; so we append the velodyne-to-cam0 matrix if needed.
	if T_cam0_velodyne is not None:
		mat = mat @ T_cam0_velodyne[None,...]

	# Use given base scan as origin
	if basescan is not None:
		mat = np.linalg.inv(mat[None,basescan]) @ mat

	# Return raw matrices if requested
	if raw_matrices:
		return mat

	# Extract translation vector and 3x3 rotation submatrix
	xlate = mat[:,0:3,3]
	rotmat = mat[:,0:3,0:3]

	# Convert rotation matrices to quaternions
	rotq = np.empty((len(rotmat),4), dtype=np.float32)
	for j in range(len(rotmat)):
		rotq[j] = mat_to_quat(rotmat[j])

	return xlate, rotq
