import glob
import numpy as np
from dataclasses import dataclass
from typing import Union, List, Tuple, Iterator

from utm import from_latlon, latitude_to_zone_letter, latlon_to_zone_number

from .utils import quat_vec_mult
from .radar import ImuData, RadarData

import rosbag
from rospy import Time, Duration
from sensor_msgs.msg import PointCloud2, Imu, Image, NavSatFix
from sensor_msgs.point_cloud2 import read_points
from cv_bridge import CvBridge

GT_TOPIC = '/fix'

TOPIC_LIST = [
	RADAR_TOPIC := '/oculii_radar/point_cloud',
	IMU_TOPIC   := '/imu/data',
	CAM_TOPIC   := '/camera_array/left/image_raw',
]

def _parse_imu(msg: Imu):
	t = msg.header.stamp.to_sec()
	accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=np.float32)
	omega = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float32)

	# XX: Incoming covariance matrices are zero
	#accel_cov = np.array(msg.linear_acceleration_covariance, dtype=np.float32).reshape((3,3))
	#omega_cov = np.array(msg.angular_velocity_covariance,    dtype=np.float32).reshape((3,3))

	accel_cov = 0.01 * np.eye(3, dtype=np.float32)
	omega_cov = accel_cov

	return ImuData(t=t, accel=accel, accel_cov=accel_cov, omega=omega, omega_cov=omega_cov)

def _parse_radar(msg: PointCloud2):
	cl = []
	for p in read_points(msg, field_names=('x','y','z','power','doppler'), skip_nans=True):
		cl.append((p[2], -p[0], -p[1], p[4], p[3])) # XX: power and doppler fields are swapped

	cl = np.array(cl, dtype=np.float32)
	return msg.header.stamp.to_sec(), cl

def load_mscrad4r_gt(basedir, seqid) -> np.ndarray:
	ret = []

	with rosbag.Bag(f'{basedir}/{seqid}.bag') as bag:
		for _, msg, _ in bag.read_messages(topics=[GT_TOPIC]):
			msg:NavSatFix
			ret.append(( msg.header.stamp.to_sec(), msg.latitude, msg.longitude, msg.altitude ))

	ret = np.array(ret, dtype=np.float64)

	zl = latitude_to_zone_letter(ret[:,1])
	zn = latlon_to_zone_number(ret[:,1], ret[:,2])

	east, north, _, _ = from_latlon(ret[:,1], ret[:,2], force_zone_number=zn, force_zone_letter=zl)

	east -= east[0]
	north -= north[0]
	up = ret[:,3] - ret[0,3]

	return ret[:,0], np.stack([ east, north, up ], axis=-1).astype(np.float32)

def load_mscrad4r_seq(basedir, seqid) -> Iterator[RadarData]:
	imu_accum = []
	last_image:Image = None

	with rosbag.Bag(f'{basedir}/{seqid}.bag') as bag:
		for topic, msg, t in bag.read_messages(topics=TOPIC_LIST):
			if topic == IMU_TOPIC:
				imu_accum.append(_parse_imu(msg))
			elif topic == RADAR_TOPIC:
				t,cl = _parse_radar(msg)
				img = CvBridge().imgmsg_to_cv2(last_image, desired_encoding='rgb8') if last_image is not None else None
				yield RadarData(t=t, scan=cl, imu=imu_accum, img=img)
				imu_accum = []
			elif topic == CAM_TOPIC:
				last_image = msg
