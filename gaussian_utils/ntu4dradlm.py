import glob
import numpy as np
from dataclasses import dataclass
from typing import Union, List, Tuple, Iterator

from .utils import quat_vec_mult, quat_mult, rpy_to_quat
from .radar import ImuData, RadarData

import rosbag
from rospy import Time, Duration
from sensor_msgs.msg import PointCloud, Imu, CompressedImage
from cv_bridge import CvBridge

TAU = float(2*np.pi)

TOPIC_LIST = [
	RADAR_TOPIC := '/radar_enhanced_pcl',
	IMU_TOPIC   := '/vectornav/imu',
	CAM_TOPIC   := '/rgb_cam/image_raw/compressed',
]

RADAR_TO_IMU = np.array([
	[ 0.999735807578,   -0.0215215701795, -0.0081643477385,  -0.3176955976234  ],
	[ 0.02148120581797,  0.9997581134183, -0.00502853428037,  0.13761019052125 ],
	[ 0.00826995351904,  0.0048509797951,  0.99995400578406, -0.05898352725152 ],
], dtype=np.float32)

GT_ROT_CORRECTIONS = {
	'cp':    ( 0.0,  -0.30, 0.0 ),
	'nyl':   ( 1.5,   0.65, 0.0 ),
	'loop1': (-0.9,   2.0,  0.0 ),
	'loop2': ( 1.75,  3.4,  0.0 ),
	'loop3': (-0.45,  3.77, 0.0 ),
}

ACCEL_NOISE_STD = 0.0022281160035059417
ACCEL_RWALK_STD = 0.00011782392708033614

GYRO_NOISE_STD = 0.00011667951042710442
GYRO_RWALK_STD = 2.616129872371749e-06

def load_ntu4dradlm_gt(basedir, seqid):
	mat = np.loadtxt(f'{basedir}/{seqid}/gt_odom.txt', dtype=np.float64)

	ts   = mat[:,0]
	pos  = mat[:,1:4].astype(np.float32)
	pos -= pos[0]
	quat = mat[:,4:8].astype(np.float32)
	quat = quat[:,[3,0,1,2]]

	if correction := GT_ROT_CORRECTIONS.get(seqid):
		cquat = rpy_to_quat(np.array(correction, dtype=np.float32) * TAU/360)
		quat = quat_mult(cquat, quat)
		pos = quat_vec_mult(cquat[None], pos)

	return ts, pos, quat

def _parse_imu(msg: Imu):
	t = msg.header.stamp.to_sec()
	accel = np.array([msg.linear_acceleration.x, -msg.linear_acceleration.y, -msg.linear_acceleration.z], dtype=np.float32)
	omega = np.array([msg.angular_velocity.x,    -msg.angular_velocity.y,    -msg.angular_velocity.z],    dtype=np.float32)

	#accel_cov = np.array(msg.linear_acceleration_covariance, dtype=np.float32).reshape((3,3))
	#omega_cov = np.array(msg.angular_velocity_covariance,    dtype=np.float32).reshape((3,3))

	accel_cov = (ACCEL_NOISE_STD**2) * np.eye(3, dtype=np.float32)
	omega_cov = ( GYRO_NOISE_STD**2) * np.eye(3, dtype=np.float32)

	return ImuData(t=t, accel=accel, accel_cov=accel_cov, omega=omega, omega_cov=omega_cov)

def _parse_radar(msg: PointCloud):
	chan = {}
	for x in msg.channels:
		chan[x.name] = x.values

	# Notes:
	# Alpha = -azimuth         (calculated as atan2(-y,x) instead of atan2(y,x))
	# Beta = elevation - tau/4 (calculated as atan2(-z,r) instead of atan2(r,z))

	# Note that standard azimuth is [-tau/2, tau/2] and standard elevation is [0, tau/2]
	# This radar is NED, so the signs of Y/Z are flipped, and offsets elevation by tau/4
	# so that [0, tau/2] becomes [-tau/4, tau/4] and 0 means Z=0

	# We decide to use the following convention:
	# Positive azimuth: turning left (with zero meaning Y=0)
	# Positive elevation: turning up (with zero meaning Z=0)

	cl = np.empty((len(msg.points), 5), dtype=np.float32)
	for i,p in enumerate(msg.points):
		cl[i,0:3] = (p.x, p.y, p.z)

	cl[:,3] = np.array(chan['Power'], dtype=np.float32)
	cl[:,4] = np.array(chan['Doppler'], dtype=np.float32)

	return msg.header.stamp.to_sec(), cl

def load_ntu4dradlm_seq(basedir, seqid) -> Iterator[RadarData]:
	baglist = sorted(glob.glob(f'{basedir}/{seqid}/{seqid}_*.bag'))

	imu_accum = []
	last_image:CompressedImage = None

	for bagfilename in baglist:
		with rosbag.Bag(bagfilename) as bag:
			for topic, msg, t in bag.read_messages(topics=TOPIC_LIST):
				if topic == IMU_TOPIC:
					imu_accum.append(_parse_imu(msg))
				elif topic == RADAR_TOPIC:
					t,cl = _parse_radar(msg)
					img = CvBridge().compressed_imgmsg_to_cv2(last_image, desired_encoding='rgb8') if last_image is not None else None
					yield RadarData(t=t, scan=cl, imu=imu_accum, img=img)
					imu_accum = []
				elif topic == CAM_TOPIC:
					last_image = msg
