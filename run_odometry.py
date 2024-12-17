import torch
import numpy as np
from datetime import datetime

from gaussian_utils.config import CONFIG
from gaussian_utils.utils import quat_to_rpy
from gaussian_utils.robot3d import RobotPose3D
from gaussian_utils.odom import ImuRadarGicpOdometry, ImuRadarGaussianOdometry
from gaussian_utils.drawing import visualize_odom

TAU = float(2*np.pi)

DATASET_TYPE = CONFIG.get('config', 'dataset', fallback='NTU4DRadLM')
DATASET_DIR = CONFIG.get(DATASET_TYPE, 'path')

DATASET_SEQ = CONFIG.get('odometry', 'sequence')
ABLATION_GICP = CONFIG.getboolean('odometry', 'ablation_gicp', fallback=False)
NUM_PARTICLES = CONFIG.getint('odometry', 'num_particles', fallback=4)

OUT_NAME = CONFIG.get('odometry', 'out_name', fallback=None)
if OUT_NAME is None:
	OUT_NAME = 'odom_' + datetime.now().strftime('%Y%m%d%H%M%S')
OUT_NAME = f'{OUT_NAME}-{DATASET_TYPE}-{DATASET_SEQ}'

if DATASET_TYPE == 'NTU4DRadLM':
	from gaussian_utils.ntu4dradlm import load_ntu4dradlm_gt, load_ntu4dradlm_seq, RADAR_TO_IMU, ACCEL_RWALK_STD, GYRO_RWALK_STD
	gt_ts, gt_pos, gt_quat = load_ntu4dradlm_gt(DATASET_DIR, DATASET_SEQ)
	dataset = load_ntu4dradlm_seq(DATASET_DIR, DATASET_SEQ)
else:
	raise Exception('Unknown dataset type')

#------------------------------------------------------------------------------

np.set_printoptions(linewidth=1000)

if not ABLATION_GICP:
	odom = ImuRadarGaussianOdometry(radar_to_imu=RADAR_TO_IMU, accel_bias_std=ACCEL_RWALK_STD, gyro_bias_std=GYRO_RWALK_STD, num_particles=NUM_PARTICLES)
else:
	odom = ImuRadarGicpOdometry(radar_to_imu=RADAR_TO_IMU, accel_bias_std=ACCEL_RWALK_STD, gyro_bias_std=GYRO_RWALK_STD)
qid = -1

pred_t = []
pred_pos = []
pred_rot = []

imu_rp = []
egovel = []
aclr_bias = []
gyro_bias = []

def is_keyframe(oldpose:RobotPose3D, newpose:RobotPose3D):
	xlate = float(torch.linalg.norm(newpose.xyz_tran[0] - oldpose.xyz_tran[0]))
	rot = (newpose.mat_rot[0] @ oldpose.mat_rot[0].T).cpu().numpy()
	tr = rot[0,0] + rot[1,1] + rot[2,2]
	angle = np.arccos(min(1.0, np.abs(0.5*(tr - 1))))*360/TAU
	print('  translated',xlate,'m, rotated',angle,'º')
	return xlate >= 15.0 or angle >= 15.0

txtout = []

for bundle in dataset:
	scan = odom.process(bundle)

	if odom.is_initial:
		continue # Shouldn't happen, but just in case

	gt_idx = np.argmin(np.abs(gt_ts - bundle.t))
	cur_gt_pos = gt_pos[gt_idx]
	cur_gt_rot = gt_quat[gt_idx] if gt_quat is not None else odom.quat

	qid += 1
	print(f'---- Scan {qid}, t = {odom.time:.3f} s')
	print('  PR POS', odom.pos, 'ROT', quat_to_rpy(odom.quat) * 360/TAU)
	print('  GT POS', cur_gt_pos, 'ROT', quat_to_rpy(cur_gt_rot) * 360/TAU)

	txtout.append(f'{bundle.t} {float(odom.pos[0])} {float(odom.pos[1])} {float(odom.pos[2])} {float(odom.quat[1])} {float(odom.quat[2])} {float(odom.quat[3])} {float(odom.quat[0])}\n')

	if odom.time >= 2.0 if odom.has_empty_keyframe else odom.match_time >= 0.5 or is_keyframe(odom.kf_pose, odom.pose):
		print('  Keyframe')
		odom.keyframe()

	pred_t.append(odom.time)
	pred_pos.append(odom.pos)
	pred_rot.append(odom.quat)
	imu_rp.append(odom.imu_rp)
	egovel.append(odom.egovel)
	aclr_bias.append(odom.accel_bias)
	gyro_bias.append(odom.gyro_bias)

with open(f'{OUT_NAME}-traj.txt', 'w', encoding='utf-8') as f:
	f.write(''.join(txtout))

visualize_odom(
	np.array(pred_t),
	gt_ts - odom.ref_time,
	np.array(pred_pos),
	gt_pos,
	np.array(pred_rot),
	gt_quat,
	np.array(imu_rp),
	None,
	np.array(egovel),
	np.array(aclr_bias),
	np.array(gyro_bias),
	outfile=f'{OUT_NAME}-stats.png'
)
