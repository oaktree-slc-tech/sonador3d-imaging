import numpy as np
import SimpleITK as sitk

from sonador.imaging.orthanc.base import ImageCoord

from .apisettings import DCM_VOLUME_LEFT_POSTERIOR_SUPERIOR


def imagevolume_center(img: sitk.Image):
	'''	return the physical center point of the provided image
	'''
	width, height, depth = img.GetSize()
	return ImageCoord(*img.TransformIndexToPhysicalPoint((
		int(np.ceil(width/2)),
        int(np.ceil(height/2)),
		int(np.ceil(depth/2)))))


def init_empty_image(img: sitk.Image, direction=None, origin=None, spacing=None, dtype=None):
	'''	Create an empty SimpleITK.Image instance with the same parameters as the provided input.
	'''
	# Retrieve size, direction, and origin
	isize = img.GetSize()
	direction = direction or img.GetDirection()
	origin = origin or img.GetOrigin()
	spacing = spacing or img.GetSpacing()

	# Initialize image
	empty_img = sitk.Image(isize[0], isize[1], isize[2], dtype or img.GetPixelIDValue())
	empty_img.SetDirection(direction)
	empty_img.SetOrigin(origin)
	empty_img.SetSpacing(spacing)

	return empty_img
