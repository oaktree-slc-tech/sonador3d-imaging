import os, tempfile, logging, functools, abc
from abc import ABC
from collections import OrderedDict, namedtuple

import numpy as np
import SimpleITK as sitk

from sonador.imaging.orthanc import ImagingSeries
from sonador.imaging.orthanc.base import ImageCoord
from sonador.tasks.images import download_imaging_filearchive
from sonador.imaging.orthanc.seg import ImageSegmentData

from .apisettings import NUETRAL_3D_POSE, LABELMAP_DTYPE_DEFAULT
from .seg import dcmseg2labelmap
from .helpers import imagevolume_center, init_empty_image

logger = logging.getLogger(__name__)


class SonadorImagingVolume(object):
	'''	Helper object used for working with 3D medical imaging data retrieved from Sonador.
		"volume" property is a SimpleITK image.
	'''
	def __init__(self, series, *args, **kwargs):
		self.series = series
		self._volume = kwargs.get('volume')
		self._reader = kwargs.get('reader')

		# Ensure that an imaging series was provided
		if not isinstance(self.series, ImagingSeries):
			raise ValueError('Unable to initialize imaging volume, invalid DICOM resource. '
				+ 'Volumes may only be initialized using ImagingSeries resources.')

		# Check volume instance and ensure it is a SimpleITK Image
		if self._volume and not isinstance(self._volume, sitk.Image):
			raise ValueError('Invalid volume type. SonadorImagingVolume instances '
				+ 'may only be initilized from SimpleITK.Image instances.')

		# Labelmap options
		self._labelmap_opts = kwargs.get('labelmap_opts', {})

	def fectch_volume(self, *args, **kwargs):
		'''	Retrieve the voxel data associated with the image and return a point cloud

			@returns SimpleITK.Image
		'''
		if not len(self.series.slices_collection):
			raise ValueError(('Unable to retrieve imaging data for series %s, series does not contain any '
				+ 'DICOM slices.') % self.series.pk)

		with tempfile.TemporaryDirectory() as tmp:

			# Download and unpack imaging pixel data from Orthanc
			download_imaging_filearchive(self.series, tmp, extract=True)
			ipath = os.path.join(tmp, self.series.pk, 'IMAGES')

			# Initialize image series reader
			self._reader = sitk.ImageSeriesReader()
			if kwargs.get('load_meta', True):
				self._reader.MetaDataDictionaryArrayUpdateOn()
				self._reader.LoadPrivateTagsOn()

			# Set file names
			self._reader.SetFileNames(self._reader.GetGDCMSeriesFileNames(ipath))

			# Set output options
			self._reader.SetImageIO(kwargs.get('image_type', ''))
			self._reader.SetOutputPixelType(kwargs.get('pixel_type', sitk.sitkUnknown))

			# Read image stack
			return self._reader.Execute()

	@property
	def center(self):
		'''	Return the physical center point of the volume

			@returns ImageCoord
		'''
		return imagevolume_center(self.volume)

	@property
	def origin(self):
		'''	Retrieve the origin point of the volume coordinate system.

			@returns ImageCoord
		'''
		return ImageCoord(*self.volume.GetOrigin())

	@property
	def volume(self):
		'''	Voxel data for the imaging series

			@returns sitk.Image
		'''
		if getattr(self, '_volume', None) is None:
			setattr(self, '_volume', self.fectch_volume())

		return self._volume

	def fetch_dcmsegments(self, *args, segments=None, segfilter=None, **kwargs):
		'''	Retrieve DICOM segments from imaging series. All segments from all segmentations
			are included.

			@returns iterable of DICOM segments
		'''
		# Retrieve segments from DICOM segmentations associated with the series
		segments = segments or []
		for dcmseg in self.series.segmentations:
			segments.extend([s for s in dcmseg.segments])

		# Filter segments
		if segfilter and callable(segfilter):
			segments = list(filter(segfilter, segments))

		return segments

	def labelmap_arrays(self, segments=None, dtype=LABELMAP_DTYPE_DEFAULT, **kwargs):
		'''	Create labelmap arrays from image segments for the image

			@input segments: iterable of segments (ImageSegmentData) to be included in the labelmap
				(default=segmentations associated with the imaging series)

			returns OrderedDict of labelmaps (numpy.ndarray) indexed to the segment metadata
		'''
		# Retrieve segments from Sonador segmentations (DICOM-SEG)
		segments = segments or self.fetch_dcmsegments(**kwargs)

		# Package segments as NumPy arrays
		return OrderedDict(
			(seg.meta, dcmseg2labelmap(self, seg, dtype=dtype, **kwargs)) for seg in segments)

	def fetch_labelmaps(self, segments=None, dtype=LABELMAP_DTYPE_DEFAULT, **kwargs):
		'''	Retrieve segmentations and create labelmap images from segments for the volume. 
			(Calls labelmap_arrays to package the labelmap volumes.)
			
			@returns OrderedDict of labelmaps (sitk.Image) indexed to the segment metadata
		'''
		labelmaps = OrderedDict()

		for smeta, sdata in self.labelmap_arrays(segments=segments, dtype=dtype, **kwargs).items():

			# Initialize labelmap instance with volume direction, origin, spacing
			seg_labelmap = labelmaps[smeta] = sitk.GetImageFromArray(sdata)
			seg_labelmap.SetDirection(self.volume.GetDirection())
			seg_labelmap.SetOrigin(self.volume.GetOrigin())
			seg_labelmap.SetSpacing(self.volume.GetSpacing())

		return labelmaps

	def _init_empty_image(self, img=None, direction=None, origin=None, spacing=None):
		''' Initialize an SimpleITK.Image with the same properties as the provided image.

			@input img (SimpleITK.Image, default=volume): Image to be used for creating the
				placeholder.
		'''
		return init_empty_image(img or self.volume, direction=direction, origin=origin, spacing=spacing,
			dtype=self._labelmap_opts.get('dtype'))

	@property
	def labelmaps(self, *args, **kwargs):
		'''	Labelmap images of segmentations associated with the image.

			@returns OrderedDict of labelmap images (SimpleITK.Image) keyed to the segment metadata
		'''
		if getattr(self, '_labelmap_dict', None) is None:
			setattr(self, '_labelmap_dict', self.fetch_labelmaps(**self._labelmap_opts))

		return self._labelmap_dict

	@property
	def labelmaps_image(self):
		'''	SimpleITK.Image with labelmap values in a single image instance for visualization and multi-label analysis.
			Each segmentation is encoded with a different integer value depending on their order in the DICOM-SEG file.			
		'''
		# Pack labelmaps into SimpleITK image instance
		seg_labelmap = self._init_empty_image()
		for i, cmask in enumerate(self.labelmaps.values()):
			seg_labelmap += (i+1)*sitk.Cast(cmask, seg_labelmap.GetPixelIDValue())

		return seg_labelmap	

	@property
	def volume_array(self):
		'''	numpy.ndarray representation of the imaging voxel data
		'''
		return sitk.GetArrayFromImage(self.volume)

	@property
	@functools.lru_cache()
	def rotation_matrix(self):
		'''	Rotation matrix for the image series calculated from the ImageOrientationPatient
			headers of the series.

			@returns numpy.Array
		'''
		o = self.series.image_orientation_patient

		# Package as x, y, z components
		x = np.array([o.row.x, o.row.y, o.row.z])
		y = np.array([o.col.x, o.col.y, o.col.z])
		z = np.cross(x, y)

		return np.array([x, y, z]).T

	@property
	@functools.lru_cache()
	def sitk_rotation_matrix(self):
		return np.array(self.volume.GetDirection()).reshape((3,3))

	def resample(self, transform, output_volume=None, default_value=0, interpolator=sitk.sitkLinear, inplace=False):
		'''	Apply an image transform transform and resample the volume.

			@input transform (SimpleITK.Transform): SimpleITK transform to be performed.
			@input output_volume (SimpleITK.Image, default=self.volume): Volume which should be used
				to bound the data after the transform has been applied.
			@input inplace (bool, default=False): Toggles whether the transform should return a 
				new SimpleITK.Image instance and new SonadorImagingVolume or should
				be applied to the current instance.

			@returns SonadorImagingVolume with the transform applied
		'''
		# Apply the transform and resample to a new output volume
		if output_volume:
			volume = sitk.Resample(self.volume, output_volume, transform, interpolator, default_value)
		
		# If an output size is not provided, use the current volume for the bounding shape
		else:
			volume = sitk.Resample(self.volume, self.volume, transform, interpolator, default_value)

		# Perform the transform inplace and update the internal volume
		if inplace:
			self._volume = volume
			return self

		# Return a new imaging volume with the changes applied
		return SonadorImagingVolume(self.series, volume=volume, reader=self._reader)

	def rotate(self, x_deg, y_deg, z_deg, center=None, inplace=False):
		'''	Rotate the volume the specified number of degrees around the x, y, z axis

			@returns SonadorImagingVolume with a copy of the data
		'''
		# Set the center of rotation
		if center is None:
			center = self.center

		# Convert the degrees provided to radians
		theta_x = np.deg2rad(x_deg)
		theta_y = np.deg2rad(y_deg)
		theta_z = np.deg2rad(z_deg)

		# Initialize and set properties of the transform
		transform = sitk.Euler3DTransform()
		transform.SetCenter(center)
		transform.SetRotation(theta_x, theta_y, theta_z)

		return self.resample(transform, inplace=inplace)

	def transform(self, transform_matrix, translation_matrix=None, center=None, inplace=False):
		'''	EXPERIMENTAL: Perform an affine transform defined by the transformation matrix

			@returns SonadorImagingVolume with a copy of the data
		'''
		# Set the center of rotation
		if center is None:
			center = self.center

		# Initialize and set properties of the transform
		transform = sitk.AffineTransform(self.volume.GetDimension())
		transform.SetCenter(center)

		# Set rotation matrix and translation components of the transform
		transform.SetMatrix(transform_matrix)
		if translation_matrix is not None:
			transform.SetTranslation(translation_matrix)

		return self.resample(transform, inplace=inplace)

	def translate(self, offset, inplace=False):
		'''	Translate the volume based on the provided offset

			@returns SonadorImagingVolume with a copy of the data
		'''
		transform = sitk.TranslationTransform(self.volume.GetDimension(), offset)
		return self.resample(transform, inplace=inplace)
