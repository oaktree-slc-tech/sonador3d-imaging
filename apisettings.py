import numpy as np


# Direction matrix that can be used to align imaging volumes in a neutral coordinate
# system without "pose" data or to calculate coordinate transforms.

NUETRAL_3D_POSE = np.array([
	[0, 0, -1.0],
	[1.0, 0, 0],
	[0, -1.0, 0]
])


LABELMAP_DTYPE_DEFAULT = np.uint8

IDENTIY_3D = np.identity(3)