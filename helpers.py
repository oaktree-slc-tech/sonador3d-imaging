'''	Helpers for SimpleITK imaging volumes: the geometry of a volume (center, empty copies, slice positions
	and orientation), display copies of a volume on a coarser grid (`resample_display_image`) and the
	encoding of a display copy as a derived DICOM series uploaded to Sonador (`dcm_encode_display_volume`).
'''
import copy, datetime, logging
from collections import OrderedDict, namedtuple
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import SimpleITK as sitk

from pydicom.dataset import Dataset
from pydicom.encaps import encapsulate
from pydicom.pixels import get_encoder
from pydicom.sequence import Sequence
from pydicom.uid import UID, RE_VALID_UID, generate_uid, ExplicitVRLittleEndian, \
	JPEGBaseline8Bit, JPEGLossless, JPEGLosslessSV1, JPEGLSNearLossless, JPEG2000Lossless, JPEG2000, HTJ2KLossless

from client.utils.object import pick

from sonador.apisettings import DicomMetaKey, DicomMeta, IMAGING_SERVER_RESOURCE_SERIES, IMAGING_SERVER_RESOURCE_IMAGE, \
	DCM_DATE_STRFORMAT, DCM_TIME_STRFORMAT, DCM_MODALITY_CT, \
	DCMHEADER_SOP_CLASS_UID, DCMHEADER_SOP_INSTANCE_UID, DCMHEADER_SERIES_INSTANCE_UID, DCMHEADER_SERIES_NUMBER, \
		DCMHEADER_SERIES_DESCRIPTION, DCMHEADER_SERIES_DATE, DCMHEADER_SERIES_TIME, DCMHEADER_MODALITY, \
		DCMHEADER_INSTANCE_NUMBER, DCMHEADER_IMAGE_TYPE, DCMHEADER_FRAME_OF_REFERENCE_UID, \
	DCMHEADER_IMAGE_POSITION_PATIENT, DCMHEADER_IMAGE_ORIENTATION_PATIENT, DCMHEADER_PIXEL_SPACING, \
		DCMHEADER_SLICE_THICKNESS, DCMHEADER_SLICE_LOCATION, \
	DCMHEADER_PIXEL_REPRESENTATION, DCMHEADER_BITS_STORED, DCMHEADER_HIGH_BIT, DCMHEADER_MEDIA_STORAGE_SOP_INSTANCE_UID, \
		DCMHEADER_IMG_PIXELDATA, \
	DCMHEADER_MANUFACTURER, DCMHEADER_SOFTWARE_VERSIONS, \
	DCMHEADER_SR_REF_SERIES_SEQ, DCMHEADER_SR_SOURCE_IMAGE_SEQ, DCMHEADER_SR_SOP_CLASS_UID, DCMHEADER_SR_REF_INSTANCE_UID
from sonador.helpers import srcode2dataset
from sonador.imaging.orthanc import ImagingSeries
from sonador.imaging.orthanc.base import ImageCoord
from sonador.imaging.helpers.conversion import array2dcmimg, dcm_set_original_encoding

from .apisettings import DCM_VOLUME_LEFT_POSTERIOR_SUPERIOR, \
	DCMHEADER_RESCALE_INTERCEPT, DCMHEADER_RESCALE_SLOPE, DCMHEADER_RESCALE_TYPE, DCM_RESCALE_TYPE_HU, \
	DCMHEADER_SPACING_BETWEEN_SLICES, DCMHEADER_DERIVATION_DESCRIPTION, DCMHEADER_DERIVATION_CODE_SEQUENCE, \
	DCMHEADER_PURPOSE_OF_REFERENCE_CODE_SEQUENCE, \
	DCMHEADER_LOSSY_IMAGE_COMPRESSION, DCMHEADER_LOSSY_IMAGE_COMPRESSION_RATIO, DCMHEADER_LOSSY_IMAGE_COMPRESSION_METHOD, \
		DCM_LOSSY_IMAGE_COMPRESSION_NONE, DCM_LOSSY_IMAGE_COMPRESSION_LOSSY, \
	DCM_MEDIA_STORAGE_CT_IMAGE, DCM_IMAGE_TYPE_DISPLAY_VOLUME, \
	DCMCODE_DERIVATION_SPATIAL_RESAMPLING, DCMCODE_PURPOSE_SOURCE_IMAGE_PROCESSING, \
	SITK_META_SOP_CLASS_UID, SITK_META_SOP_INSTANCE_UID, SITK_META_IMAGE_POSITION_PATIENT, SITK_INTERPOLATOR_LABELS, \
	DISPLAY_VOLUME_SIZES, DISPLAY_VOLUME_SERIES_NUMBER_BASE, DISPLAY_VOLUME_SERIES_DESCRIPTION_TEMPLATE, \
	DISPLAY_VOLUME_SERIES_DESCRIPTION_DEFAULT, DISPLAY_VOLUME_MANUFACTURER, DISPLAY_VOLUME_SOFTWARE_VERSIONS_TEMPLATE, \
	DISPLAY_VOLUME_SOURCE_HEADERS, DISPLAY_VOLUME_REQUIRED_HEADERS, DISPLAY_VOLUME_ANTIALIAS_KERNEL_WIDTH, \
	DISPLAY_VOLUME_COMPRESSION_DEFAULT, DISPLAY_VOLUME_TRANSFER_SYNTAXES, DISPLAY_VOLUME_COMPRESSION_ALIASES, \
		DCM_TRANSFER_SYNTAX_LOSSY_METHODS, DCM_TRANSFER_SYNTAX_MAX_BITS, DISPLAY_VOLUME_SOP_CLASSES_16_BIT, \
		DISPLAY_VOLUME_SOP_CLASSES_16_BIT_MIN_BITS_STORED, DISPLAY_VOLUME_MIN_BITS_STORED, \
		DISPLAY_VOLUME_SOP_CLASS_HEADERS, \
	DISPLAY_VOLUME_ENCODING_BACKEND_NATIVE, DISPLAY_VOLUME_ENCODING_BACKEND_PYDICOM, DISPLAY_VOLUME_ENCODING_BACKEND_IMAGECODECS, \
		DISPLAY_VOLUME_PYDICOM_PLUGINS, DISPLAY_VOLUME_PYDICOM_ENCODER_OPTIONS, DISPLAY_VOLUME_IMAGECODECS_CODECS, \
		DISPLAY_VOLUME_JLS_ERROR_DEFAULT, DISPLAY_VOLUME_J2K_PSNR_DEFAULT, DISPLAY_VOLUME_JPEG_QUALITY_DEFAULT, \
	DISPLAY_VOLUME_RESCALE_OVERFLOW_ADAPT, DISPLAY_VOLUME_RESCALE_OVERFLOW_RAISE

logger = logging.getLogger(__name__)


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


# Grid of a display copy of a volume: voxel counts, spacing (mm), origin and direction (row-major 3x3)
DisplayImageGrid = namedtuple('DisplayImageGrid', ('size', 'spacing', 'origin', 'direction'))

# Source instance of an imaging volume: SOP class and instance UIDs, ImagePositionPatient and the
# position projected on the slice normal
SourceSlice = namedtuple('SourceSlice', ('sop_class_uid', 'sop_instance_uid', 'position', 'z'))


def sitk_image(img):
	'''	SimpleITK image of the input: a SimpleITK.Image, or an object whose `volume` attribute holds one
		(SonadorImagingVolume).

		@returns SimpleITK.Image
	'''
	if isinstance(img, sitk.Image):
		return img

	volume = getattr(img, 'volume', None)
	if isinstance(volume, sitk.Image):
		return volume

	raise TypeError('Invalid image: %s. A SimpleITK.Image or an imaging volume holding a SimpleITK.Image is required.'
		% type(img).__name__)


def _vector3(value, dtype, label):
	'''	Three component vector from a scalar (repeated) or a sequence of three positive values
	'''
	if isinstance(value, (int, float, np.integer, np.floating)):
		value = (value, value, value)

	value = np.array(tuple(value), dtype=dtype)
	if value.shape != (3,):
		raise ValueError('Invalid %s: %s. A scalar or three values (x, y, z) are required.' % (label, str(value)))
	if np.any(value <= 0):
		raise ValueError('Invalid %s: %s. Values must be positive.' % (label, str(value)))

	return value


def _tag_float(value):
	'''	float of a DICOM tag value (Orthanc string, number or None); None when it cannot be parsed
	'''
	try:
		return float(str(value).strip()) if value is not None and str(value).strip() != '' else None
	except (TypeError, ValueError):
		return None


def _tag_int(value):
	'''	int of a DICOM tag value (Orthanc string, number or None); None when it cannot be parsed
	'''
	fval = _tag_float(value)
	return int(fval) if fval is not None else None


def _point_array(point):
	'''	numpy array of a point given as an ImageCoord (iterable, no length) or a sequence of three numbers
	'''
	return np.array(tuple(point), dtype=float)


def _tag_point(value):
	'''	ImageCoord from a DICOM point: a multi-value string ('x\\y\\z'), a sequence of three numbers or an
		ImageCoord. None when the value cannot be parsed.
	'''
	if value is None:
		return None

	if isinstance(value, str):
		value = value.replace(',', '\\').split('\\')

	try:
		coords = tuple(float(v) for v in value)
	except (TypeError, ValueError):
		return None

	return ImageCoord(*coords) if len(coords) == 3 else None


def display_image_grid(img, size=None, spacing=None):
	'''	Grid of a display copy of a volume: the voxel count, spacing, origin and direction of a copy that
		covers the extent of the source on a different grid (exactly for `size`, rounded up to whole voxels
		for `spacing`). Exactly one of `size` and `spacing` must be provided.

		* `size`: voxel count per axis, a cube of equal counts when an int is given. The spacing of each
		  axis is its extent divided by the count (anisotropic): 512 for a 512 x 512 x 1299 scan at
		  0.68 x 0.68 x 0.5 mm gives 512 x 512 x 512 voxels of 0.68 x 0.68 x 1.27 mm.
		* `spacing`: voxel size per axis, isotropic when a float is given. The count is the number of
		  voxels needed to cover the extent (rounded up), so the copy extends past the source by less than
		  a voxel along each axis whose extent is not a multiple of the spacing.

		The origin moves by half the change in spacing along each axis so that the outer edge of the first
		voxel of both grids coincides (DICOM positions are voxel centers). The direction is kept.

		@input img (SimpleITK.Image or SonadorImagingVolume): source volume
		@input size (int or three ints, default=None): voxel count per axis (x, y, z)
		@input spacing (float or three floats, default=None): voxel size per axis in mm (x, y, z)

		@returns DisplayImageGrid(size, spacing, origin, direction)
	'''
	img = sitk_image(img)
	if img.GetDimension() != 3:
		raise ValueError('Invalid image: a display grid can only be created for a 3D volume (%dD provided).'
			% img.GetDimension())
	if (size is None) == (spacing is None):
		raise ValueError('Provide either the size or the spacing of the display grid, not both.')

	src_spacing = np.array(img.GetSpacing(), dtype=float)
	extent = np.array(img.GetSize(), dtype=float) * src_spacing

	if size is not None:
		size = _vector3(size, int, 'display size')
		spacing = extent / size
	else:
		spacing = _vector3(spacing, float, 'display spacing')
		size = np.maximum(1, np.ceil(extent / spacing - 1e-6)).astype(int)

	direction = np.array(img.GetDirection(), dtype=float).reshape((3, 3))
	origin = np.array(img.GetOrigin(), dtype=float) + direction.dot((spacing - src_spacing) / 2.0)

	return DisplayImageGrid(
		tuple(int(v) for v in size), tuple(float(v) for v in spacing),
		tuple(float(v) for v in origin), tuple(float(v) for v in direction.ravel()))


def image_grid(img):
	'''	Grid (size, spacing, origin, direction) of an existing volume

		@returns DisplayImageGrid
	'''
	img = sitk_image(img)
	return DisplayImageGrid(tuple(img.GetSize()), tuple(img.GetSpacing()), tuple(img.GetOrigin()), tuple(img.GetDirection()))


def antialias_sigma(src_spacing, dst_spacing):
	'''	Gaussian sigma (mm) per axis that limits aliasing when a volume is resampled to a coarser grid: half
		the reduction factor less one, in source voxels ((f - 1) / 2 x source spacing, the scikit-image
		anti-aliasing rule), and 0 for axes that are not coarsened.

		@input src_spacing (three floats): spacing of the source (mm)
		@input dst_spacing (three floats): spacing of the display grid (mm)
		@returns tuple of three floats
	'''
	src = np.array(src_spacing, dtype=float)
	dst = np.array(dst_spacing, dtype=float)

	return tuple(float(max(0.0, (f - 1.0) / 2.0) * s) for f, s in zip(dst / src, src))


def resample_display_image(img, size=None, spacing=None, grid=None, interpolator=sitk.sitkLinear, antialias=True,
		default_value=None, output_pixel_type=None, antialias_kernel_width=DISPLAY_VOLUME_ANTIALIAS_KERNEL_WIDTH):
	'''	Display copy of a volume: the volume resampled to the grid of `display_image_grid(img, size, spacing)`
		(or to `grid`): the extent of the source on fewer (or more) voxels (exactly the extent for a
		size-based grid, up to one voxel more for a spacing-based grid, whatever `grid` describes otherwise).
		The copy keeps the direction, frame of reference and intensity units of the source; the samples
		are filtered and interpolated, so individual values differ from the source.

		@input img (SimpleITK.Image or SonadorImagingVolume): source volume
		@input size (int or three ints, default=None): voxel count of the copy, see `display_image_grid`
		@input spacing (float or three floats, default=None): voxel size of the copy, see `display_image_grid`
		@input grid (DisplayImageGrid, default=None): grid of the copy; replaces `size` and `spacing`
		@input interpolator (SimpleITK interpolator, default=sitkLinear): interpolation of the samples
		@input antialias (bool, default=True): smooth the source with a Gaussian (`antialias_sigma`) before
			it is sampled on a coarser grid; axes that are not coarsened are left alone. The smoothing keeps
			the pixel type of the source.
		@input default_value (number, default=minimum of the source): value of the voxels outside the source
		@input output_pixel_type (SimpleITK pixel id, default=pixel type of the source)
		@input antialias_kernel_width (int): largest kernel (voxels) of the anti-aliasing Gaussian

		@returns SimpleITK.Image
	'''
	img = sitk_image(img)
	grid = grid or display_image_grid(img, size=size, spacing=spacing)

	# Voxels outside of the source take the lowest value of the source (air for CT)
	if default_value is None:
		mmfilter = sitk.MinimumMaximumImageFilter()
		mmfilter.Execute(img)
		default_value = mmfilter.GetMinimum()

	# Limit aliasing along the axes that are coarsened
	source = img
	if antialias:
		sigma = antialias_sigma(img.GetSpacing(), grid.spacing)
		if any(s > 0 for s in sigma):
			source = sitk.DiscreteGaussian(img, variance=[s*s for s in sigma],
				maximumKernelWidth=antialias_kernel_width, useImageSpacing=True)

	# Reference image which defines the output grid
	reference = sitk.Image(list(grid.size), img.GetPixelIDValue())
	reference.SetSpacing(grid.spacing)
	reference.SetOrigin(grid.origin)
	reference.SetDirection(grid.direction)

	return sitk.Resample(source, reference, sitk.Transform(), interpolator, float(default_value),
		output_pixel_type if output_pixel_type is not None else img.GetPixelIDValue())


def image_orientation_patient(img):
	'''	ImageOrientationPatient of the slices of a volume: the direction cosines of the first row (x axis of
		the volume) followed by those of the first column (y axis), read from the direction matrix.

		@returns list of six floats
	'''
	direction = np.array(sitk_image(img).GetDirection(), dtype=float).reshape((3, 3))
	return [float(v) for v in np.concatenate((direction[:, 0], direction[:, 1]))]


def image_slice_normal(img):
	'''	Unit normal of the slices of a volume (direction of the z axis)

		@returns numpy.ndarray of three floats
	'''
	return np.array(sitk_image(img).GetDirection(), dtype=float).reshape((3, 3))[:, 2]


def image_slice_positions(img):
	'''	ImagePositionPatient of every slice of a volume: the physical point of the voxel (0, 0, k), the center
		of the first voxel transmitted of slice k.

		@returns list of ImageCoord
	'''
	img = sitk_image(img)
	return [ImageCoord(*img.TransformIndexToPhysicalPoint((0, 0, k))) for k in range(img.GetDepth())]


def imaging_volume_source_slices(ivolume, normal=None, sop_class_uid=None):
	'''	Source instances of an imaging volume, one entry per slice of its series, ordered along the slice
		normal: SOP class and instance UIDs, ImagePositionPatient and the position projected on the normal.

		The entries are read from the metadata of the series reader when the volume was loaded by
		`SonadorImagingVolume.fectch_volume` (no request), else from the Orthanc main tags of the
		instances of the series (ImagePositionPatient, SOPInstanceUID; one request for the instance list,
		plus one request per instance whose main tags hold no position).

		@input ivolume (SonadorImagingVolume): imaging volume
		@input normal (three floats, default=slice normal of the volume): axis the positions are projected on
		@input sop_class_uid (str, default=None): SOP class of the instances whose class is not recorded in
			the metadata or the main tags

		@returns list of SourceSlice
	'''
	series = getattr(ivolume, 'series', None)
	img = sitk_image(ivolume)
	normal = np.array(normal if normal is not None else image_slice_normal(img), dtype=float)

	entries = []

	# Series reader metadata: one dictionary per file read
	reader = getattr(ivolume, '_reader', None)
	if isinstance(reader, sitk.ImageSeriesReader):
		for k in range(len(reader.GetFileNames())):
			if not (reader.HasMetaDataKey(k, SITK_META_SOP_INSTANCE_UID) and reader.HasMetaDataKey(k, SITK_META_IMAGE_POSITION_PATIENT)):
				entries = []
				break

			position = _tag_point(reader.GetMetaData(k, SITK_META_IMAGE_POSITION_PATIENT).strip())
			if position is None:
				entries = []
				break

			entries.append((
				reader.GetMetaData(k, SITK_META_SOP_CLASS_UID).strip() if reader.HasMetaDataKey(k, SITK_META_SOP_CLASS_UID) else None,
				reader.GetMetaData(k, SITK_META_SOP_INSTANCE_UID).strip(), position))

	# Orthanc instance main tags
	if not entries and series is not None:
		for inst in series.slices_collection:
			position = _tag_point(inst.dicomdata.get(DCMHEADER_IMAGE_POSITION_PATIENT))
			if position is None:
				position = _tag_point(inst.image_position_patient)
			if position is None:
				logger.warning('series=%s instance=%s: no ImagePositionPatient, the instance is left out of the source references'
					% (getattr(series, 'pk', None), getattr(inst, 'pk', None)))
				continue

			entries.append((inst.sop_class_uid, inst.sop_instance_uid, position))

	slices = [
		SourceSlice(cls or sop_class_uid, uid, pos, float(np.dot(_point_array(pos), normal)))
		for cls, uid, pos in entries if uid]

	return sorted(slices, key=lambda s: s.z)


def source_slice_pitch(source_slices, default=None):
	'''	Distance between consecutive source slices along the normal: the median spacing of the sorted
		positions, `default` when fewer than two slices are available.

		@returns float or None
	'''
	zs = np.array([s.z for s in source_slices], dtype=float)
	if len(zs) < 2:
		return default

	return float(np.median(np.abs(np.diff(np.sort(zs)))))


def source_slice_references(source_slices, z, thickness, pitch, purpose=DCMCODE_PURPOSE_SOURCE_IMAGE_PROCESSING):
	'''	SourceImageSequence of a display slice: the source slices whose extent along the normal
		([z - pitch/2, z + pitch/2]) overlaps the extent of the display slice ([z - thickness/2, z + thickness/2]),
		each with the purpose of reference "Source image for image processing operation".

		@input source_slices (iterable of SourceSlice): source slices sorted along the normal
		@input z (float): position of the display slice on the normal
		@input thickness (float): thickness of the display slice (its spacing along the normal)
		@input pitch (float): distance between source slices along the normal
		@input purpose (highdicom Code, default=Source image for image processing operation)

		@returns pydicom.sequence.Sequence
	'''
	zs = np.array([s.z for s in source_slices], dtype=float)
	within = np.abs(zs - float(z)) <= (float(thickness) + float(pitch or 0)) / 2.0 + 1e-6

	items = []
	for idx in np.nonzero(within)[0]:
		src = source_slices[int(idx)]

		item = Dataset()
		if src.sop_class_uid:
			setattr(item, DCMHEADER_SR_SOP_CLASS_UID, src.sop_class_uid)
		setattr(item, DCMHEADER_SR_REF_INSTANCE_UID, src.sop_instance_uid)
		setattr(item, DCMHEADER_PURPOSE_OF_REFERENCE_CODE_SEQUENCE, Sequence([srcode2dataset(purpose, omit_empty=True)]))
		items.append(item)

	return Sequence(items)


def copy_private_blocks(src_ds, dst_ds, creators):
	'''	Copy the private blocks of the given private creators from one dataset to another. The elements
		are added to the block the creator is given in the destination, which may be a different block
		slot than in the source.

		@input src_ds (pydicom.dataset.Dataset): dataset the blocks are read from
		@input dst_ds (pydicom.dataset.Dataset): dataset the blocks are added to
		@input creators (iterable of str): private creator identifiers (e.g. "ACME PLANNING 1.0")

		@returns int: number of elements copied
	'''
	creators = set(str(c).strip() for c in (creators or ()))
	copied = 0

	for creator_el in [el for el in src_ds if el.tag.is_private and el.tag.element < 0x0100]:
		creator = str(creator_el.value).strip()
		if creator not in creators:
			continue

		group, slot = creator_el.tag.group, creator_el.tag.element
		block = dst_ds.private_block(group, creator, create=True)

		for el in src_ds:
			if el.tag.group == group and (el.tag.element >> 8) == slot and el.tag.element >= 0x0100:
				block.add_new(el.tag.element & 0xff, el.VR, copy.deepcopy(el.value))
				copied += 1

	return copied


def display_volume_size_label(size):
	'''	Label of a display grid size: the voxel count of a cube ("512"), else the counts per axis
		("512x512x300")
	'''
	size = tuple(int(v) for v in size)
	return '%d' % size[0] if len(set(size)) == 1 else 'x'.join(str(v) for v in size)


def display_volume_series_number(size, base=DISPLAY_VOLUME_SERIES_NUMBER_BASE):
	'''	Default series number of a display series: `base` plus the largest voxel count (512 -> 1512)
	'''
	return int(base) + max(int(v) for v in size)


def display_volume_series_description(source_description, size, template=DISPLAY_VOLUME_SERIES_DESCRIPTION_TEMPLATE,
		default=DISPLAY_VOLUME_SERIES_DESCRIPTION_DEFAULT):
	'''	Default description of a display series: the description of the source followed by the size label
		("PELVIS _FEM Display 512"), `default` when the source has no description
	'''
	label = display_volume_size_label(size)
	description = str(source_description or '').strip()

	if description:
		return (template % {'description': description, 'size': label}).strip()

	return (default % {'size': label}).strip()


def display_volume_derivation_description(source_uid, source_grid, grid, interpolator=sitk.sitkLinear, antialias=True):
	'''	DerivationDescription of a display series: what was resampled from where and how. The physical
		extent of both grids is recorded (equal for a size-based grid, up to one voxel larger for a
		spacing-based grid, arbitrary for a caller-supplied grid or display image) rather than claimed.
	'''
	fmt_size = lambda s: 'x'.join(str(int(v)) for v in s)
	fmt_spacing = lambda s: ' x '.join('%.4g' % float(v) for v in s)
	fmt_extent = lambda g: ' x '.join('%.4g' % (float(n) * float(s)) for n, s in zip(g.size, g.spacing))

	return ('Display volume %s voxels (%s mm, extent %s mm) resampled from series %s (%s voxels, %s mm, extent %s mm) '
		+ 'with SimpleITK %s: %s interpolation%s; same frame of reference') % (
		fmt_size(grid.size), fmt_spacing(grid.spacing), fmt_extent(grid), source_uid,
		fmt_size(source_grid.size), fmt_spacing(source_grid.spacing), fmt_extent(source_grid),
		sitk.Version_VersionString(), SITK_INTERPOLATOR_LABELS.get(interpolator, 'SimpleITK interpolator %s' % interpolator),
		', Gaussian anti-aliasing' if antialias else '')


# Stored pixel representation of a display volume: rescale, signedness, BitsStored, numpy dtype, the range
# of the stored values and whether the rescale of the source was preserved
DisplayPixelRepresentation = namedtuple('DisplayPixelRepresentation',
	('intercept', 'slope', 'unsigned', 'bits_stored', 'dtype', 'stored_min', 'stored_max', 'preserved'))


def _stored_range(bits_stored, unsigned):
	'''	Smallest and largest stored value of a representation
	'''
	if unsigned:
		return 0, 2**bits_stored - 1
	return -(2**(bits_stored - 1)), 2**(bits_stored - 1) - 1


def _bits_needed(smin, smax, unsigned):
	'''	BitsStored needed to hold the stored values smin..smax
	'''
	if unsigned:
		return max(1, int(np.ceil(np.log2(smax + 1))) if smax > 0 else 1)
	bits_neg = int(np.ceil(np.log2(-smin))) + 1 if smin < 0 else 1
	bits_pos = int(np.ceil(np.log2(smax + 1))) + 1 if smax > 0 else 1
	return max(1, bits_neg, bits_pos)


def image_intensity_range(img):
	'''	Smallest and largest intensity of a volume

		@returns tuple (min, max) of floats
	'''
	mmfilter = sitk.MinimumMaximumImageFilter()
	mmfilter.Execute(sitk_image(img))
	return float(mmfilter.GetMinimum()), float(mmfilter.GetMaximum())


def display_pixel_representation(display, src_tags=None, preserve_rescale=True, rescale_overflow=DISPLAY_VOLUME_RESCALE_OVERFLOW_ADAPT,
		max_bits=16):
	'''	Stored pixel representation of a display volume: RescaleIntercept, RescaleSlope, signedness and
		BitsStored chosen so that every intensity of the volume is representable. Values are never
		saturated to the range of the source: the representation is widened, re-derived or rejected.

		The representations are tried in this order:

		1. With `preserve_rescale` the intercept and slope of the source are kept (a negative slope
		   included) and the signedness and BitsStored are widened as needed: a source stored as unsigned
		   12 bit whose resampled intensities need 13 bits is stored as unsigned 13 bit (RescaleIntercept
		   and RescaleSlope unchanged).
		2. Signed `max_bits` bit intensities with RescaleIntercept 0 and RescaleSlope 1, when the source
		   has no rescale, `preserve_rescale` is False, or the stored values of step 1 exceed `max_bits`.
		3. RescaleIntercept at the minimum and a RescaleSlope covering the range as unsigned `max_bits`
		   bit (quantized, logged as a warning).

		`rescale_overflow` "adapt" falls through the steps; "raise" raises a ValueError at the first
		step whose stored values overflow: with `preserve_rescale` as soon as the rescale of the source
		cannot be kept (before step 2 is considered), else when the identity rescale overflows.

		@input display (SimpleITK.Image or SonadorImagingVolume): display volume
		@input src_tags (dict, default=None): headers of the first instance of the source series
			(RescaleIntercept, RescaleSlope, PixelRepresentation, BitsStored)
		@input preserve_rescale (bool, default=True): keep the rescale of the source when possible
		@input rescale_overflow (str, default="adapt"): "adapt" (fall through to the next representation)
			or "raise" (ValueError at the first representation that overflows), see above
		@input max_bits (int, default=16): largest BitsStored of the stored values

		@returns DisplayPixelRepresentation; its dtype (BitsAllocated) is 8 bit for at most 8 stored bits, else
			16 bit. `stored_pixel_container` adjusts it to the requirements of the SOP class and the decoders.
	'''
	src_tags = src_tags or {}
	if rescale_overflow not in (DISPLAY_VOLUME_RESCALE_OVERFLOW_ADAPT, DISPLAY_VOLUME_RESCALE_OVERFLOW_RAISE):
		raise ValueError('Invalid rescale_overflow: %s. "%s" or "%s" expected.' % (
			rescale_overflow, DISPLAY_VOLUME_RESCALE_OVERFLOW_ADAPT, DISPLAY_VOLUME_RESCALE_OVERFLOW_RAISE))

	vmin, vmax = image_intensity_range(display)

	intercept = _tag_float(src_tags.get(DCMHEADER_RESCALE_INTERCEPT))
	slope = _tag_float(src_tags.get(DCMHEADER_RESCALE_SLOPE))
	src_representation = _tag_int(src_tags.get(DCMHEADER_PIXEL_REPRESENTATION))
	src_unsigned = src_representation == 0 if src_representation is not None else None
	src_bits = min(max(_tag_int(src_tags.get(DCMHEADER_BITS_STORED)) or 0, 0), max_bits)

	def representation(intercept, slope, unsigned, bits_stored, preserved):
		dtype = (np.uint8 if unsigned else np.int8) if bits_stored <= 8 else (np.uint16 if unsigned else np.int16)
		smin, smax = _stored_range(bits_stored, unsigned)
		return DisplayPixelRepresentation(float(intercept), float(slope), bool(unsigned), int(bits_stored), dtype, smin, smax, bool(preserved))

	# Rescale of the source, widened as needed. A negative slope reverses the order of the stored endpoints.
	if preserve_rescale and intercept is not None and slope:
		smin, smax = sorted(int(np.rint((v - intercept) / slope)) for v in (vmin, vmax))

		if src_unsigned is not False and smin >= 0 and smax <= 2**max_bits - 1:
			return representation(intercept, slope, True, min(max(src_bits, _bits_needed(smin, smax, True)), max_bits), True)

		if smin >= -(2**(max_bits - 1)) and smax <= 2**(max_bits - 1) - 1:
			return representation(intercept, slope, False,
				min(max(src_bits if src_unsigned is False else 0, _bits_needed(smin, smax, False)), max_bits), True)

		message = ('intensities %g..%g need stored values %d..%d which do not fit %d bit with the rescale of the source '
			+ '(intercept %g, slope %g)') % (vmin, vmax, smin, smax, max_bits, intercept, slope)
		if rescale_overflow == DISPLAY_VOLUME_RESCALE_OVERFLOW_RAISE:
			raise ValueError('Unable to preserve the pixel representation of the source: %s.' % message)
		logger.warning('Pixel representation of the source not preserved: %s; the intensities are stored with a derived rescale' % message)

	# Identity rescale: signed intensities
	smin, smax = int(np.rint(vmin)), int(np.rint(vmax))
	if smin >= -(2**(max_bits - 1)) and smax <= 2**(max_bits - 1) - 1:
		return representation(0.0, 1.0, False, max_bits, False)

	# Derived rescale covering the range (quantized when the range exceeds the stored values)
	message = 'intensities %g..%g exceed the %d bit stored values' % (vmin, vmax, max_bits)
	if rescale_overflow == DISPLAY_VOLUME_RESCALE_OVERFLOW_RAISE:
		raise ValueError('Unable to store the display volume: %s.' % message)

	slope = max(1.0, (vmax - vmin) / float(2**max_bits - 1))
	logger.warning('%s; stored with RescaleIntercept %g and RescaleSlope %g (quantization step %g)' % (message, vmin, slope, slope))
	return representation(vmin, slope, True, max_bits, False)


def stored_pixel_container(representation, sop_class_uid=None):
	'''	Pixel representation adjusted to the pixel container the SOP class and the decoders allow. BitsAllocated
		follows BitsStored (8 bit samples for at most 8 stored bits, 16 bit otherwise) so that the pixel attributes,
		the native or encoded samples and the decoders agree. A representation of fewer than 8 bits is widened to
		8 (DISPLAY_VOLUME_MIN_BITS_STORED): GDCM decodes an 8 bit container only when BitsStored equals
		BitsAllocated. The CT Image module requires BitsAllocated 16 and BitsStored 12 to 16 (PS3.3 C.8.2.1.1.4,
		C.8.2.1.1.5; DISPLAY_VOLUME_SOP_CLASSES_16_BIT): a narrower representation is widened to 12 bits in a
		16 bit container for it. Widening never saturates.

		@input representation (DisplayPixelRepresentation): see `display_pixel_representation`
		@input sop_class_uid (str, default=None): SOP class of the instances written

		@returns DisplayPixelRepresentation
	'''
	bits = max(representation.bits_stored, DISPLAY_VOLUME_MIN_BITS_STORED)
	if sop_class_uid is not None and str(sop_class_uid) in DISPLAY_VOLUME_SOP_CLASSES_16_BIT:
		bits, wide = max(bits, DISPLAY_VOLUME_SOP_CLASSES_16_BIT_MIN_BITS_STORED), True
	else:
		wide = bits > 8

	unsigned = representation.unsigned
	dtype = (np.uint16 if unsigned else np.int16) if wide else (np.uint8 if unsigned else np.int8)
	smin, smax = _stored_range(bits, unsigned)
	return representation._replace(bits_stored=bits, dtype=dtype, stored_min=smin, stored_max=smax)


def stored_pixel_values(arr, representation):
	'''	Stored values of an array of intensities for a pixel representation: (value - intercept) / slope
		rounded to the nearest integer. The representation covers the intensities by construction
		(`display_pixel_representation`); values outside of it raise a ValueError rather than saturate.

		@returns numpy.ndarray of representation.dtype
	'''
	stored = np.rint((np.asarray(arr).astype(np.float64) - representation.intercept) / representation.slope)
	if stored.size and (stored.min() < representation.stored_min or stored.max() > representation.stored_max):
		raise ValueError('Stored values %d..%d exceed the pixel representation %d..%d (%d bit %s).' % (
			stored.min(), stored.max(), representation.stored_min, representation.stored_max,
			representation.bits_stored, 'unsigned' if representation.unsigned else 'signed'))

	return stored.astype(representation.dtype)


def _tag_list(value):
	'''	List of the values of a (possibly multi-valued) DICOM tag: an Orthanc string ('a\\b'), a sequence
		or a scalar; empty for None
	'''
	if value is None:
		return []
	if isinstance(value, str):
		return [v.strip() for v in value.split('\\') if v.strip() != '']
	if isinstance(value, (list, tuple)):
		return [str(v) for v in value]
	return [str(value)]


def lossy_compression_history(src_tags=None, transfer_syntax=None, ratio=None):
	'''	LossyImageCompression, LossyImageCompressionRatio and LossyImageCompressionMethod of a display
		slice: the history of the source (a source that was lossy compressed stays "01", its ratios and
		methods are carried forward; PS3.3 C.7.6.1.1.5) followed by the step of the display encoding when
		`transfer_syntax` is a lossy transfer syntax.

		@input src_tags (dict, default=None): headers of the first instance of the source series
		@input transfer_syntax (pydicom UID, default=None): transfer syntax of the display slice
		@input ratio (float, default=None): compression ratio of the display slice (uncompressed bytes /
			compressed bytes), recorded for a lossy transfer syntax

		@returns dict of headers
	'''
	src_tags = src_tags or {}
	lossy = str(src_tags.get(DCMHEADER_LOSSY_IMAGE_COMPRESSION) or '').strip() == DCM_LOSSY_IMAGE_COMPRESSION_LOSSY
	ratios = _tag_list(src_tags.get(DCMHEADER_LOSSY_IMAGE_COMPRESSION_RATIO))
	methods = _tag_list(src_tags.get(DCMHEADER_LOSSY_IMAGE_COMPRESSION_METHOD))

	method = DCM_TRANSFER_SYNTAX_LOSSY_METHODS.get(transfer_syntax) if transfer_syntax else None
	if method:
		lossy = True
		methods.append(method)
		if ratio:
			ratios.append('%.2f' % float(ratio))

	hdata = {DCMHEADER_LOSSY_IMAGE_COMPRESSION: DCM_LOSSY_IMAGE_COMPRESSION_LOSSY if lossy else DCM_LOSSY_IMAGE_COMPRESSION_NONE}
	if lossy and ratios:
		hdata[DCMHEADER_LOSSY_IMAGE_COMPRESSION_RATIO] = [float(v) for v in ratios]
	if lossy and methods:
		hdata[DCMHEADER_LOSSY_IMAGE_COMPRESSION_METHOD] = methods

	return hdata


def display_compression_transfer_syntax(dcm_compression):
	'''	Transfer syntax of a `dcm_compression` option: None or Explicit VR Little Endian for uncompressed
		pixel data, a pydicom transfer syntax UID of DISPLAY_VOLUME_TRANSFER_SYNTAXES, or one of the
		aliases of DISPLAY_VOLUME_COMPRESSION_ALIASES ("rle", "deflate"/"zlib", "jpeg-lossless"/"huffman",
		"jpeg-ls", "jpeg-ls-near", "jpeg2000", "jpeg2000-lossy", "htj2k", "jpeg"/"dct", "jpeg-baseline", "none").

		@returns pydicom.uid.UID or None (uncompressed)
	'''
	if dcm_compression is None:
		return None

	if isinstance(dcm_compression, str):
		alias = dcm_compression.strip().lower()
		if alias in DISPLAY_VOLUME_COMPRESSION_ALIASES:
			return DISPLAY_VOLUME_COMPRESSION_ALIASES[alias]

	transfer_syntax = UID(str(dcm_compression).strip()) if RE_VALID_UID.match(str(dcm_compression).strip()) else None
	if transfer_syntax == ExplicitVRLittleEndian:
		return None
	if transfer_syntax not in DISPLAY_VOLUME_TRANSFER_SYNTAXES:
		raise ValueError('Unsupported display volume compression: %s. Use None, one of %s or a transfer syntax among %s.' % (
			dcm_compression, ', '.join('"%s"' % a for a in sorted(k for k in DISPLAY_VOLUME_COMPRESSION_ALIASES if k)),
			', '.join(ts.name for ts in DISPLAY_VOLUME_TRANSFER_SYNTAXES)))

	return transfer_syntax


def _imagecodecs():
	'''	The imagecodecs module when installed, else None
	'''
	try:
		import imagecodecs
	except ImportError:
		return None
	return imagecodecs


def _imagecodecs_available(codec):
	'''	True when the imagecodecs codec (e.g. "jpeg2k") is installed with its encoder
	'''
	imagecodecs = _imagecodecs()
	if imagecodecs is None:
		return False
	codec_cls = getattr(imagecodecs, codec.upper(), None)
	return bool(codec_cls is not None and getattr(codec_cls, 'available', False) and hasattr(imagecodecs, '%s_encode' % codec))


def display_compression_backends(transfer_syntax):
	'''	Encoders installed for a transfer syntax, in order of preference: "pydicom:<plugin>" for each
		available pydicom encoding plugin (gdcm, pydicom, pyjpegls, pylibjpeg), "imagecodecs" when the
		imagecodecs codec of the syntax is installed, "native" for the syntaxes pydicom writes itself
		(uncompressed, deflated).

		@input transfer_syntax (pydicom UID or None)
		@returns list of str, empty when the transfer syntax cannot be produced
	'''
	if transfer_syntax is None or not UID(transfer_syntax).is_encapsulated:
		return [DISPLAY_VOLUME_ENCODING_BACKEND_NATIVE]

	backends = []
	try:
		encoder = get_encoder(transfer_syntax)
		backends.extend('%s:%s' % (DISPLAY_VOLUME_ENCODING_BACKEND_PYDICOM, plugin) for plugin in encoder.available_plugins)
	except NotImplementedError:
		pass

	codec = DISPLAY_VOLUME_IMAGECODECS_CODECS.get(transfer_syntax)
	if codec and _imagecodecs_available(codec):
		backends.append(DISPLAY_VOLUME_ENCODING_BACKEND_IMAGECODECS)

	return backends


def supported_display_compressions():
	'''	Transfer syntaxes a display volume can be encoded with and the encoders installed for each
		(`display_compression_backends`); a syntax with no encoder is listed with an empty tuple.

		@returns OrderedDict {pydicom UID or None (uncompressed): tuple of backend names}
	'''
	support = OrderedDict()
	support[None] = tuple(display_compression_backends(None))
	for transfer_syntax in DISPLAY_VOLUME_TRANSFER_SYNTAXES:
		support[transfer_syntax] = tuple(display_compression_backends(transfer_syntax))
	return support


def _select_compression_backend(transfer_syntax, encoding_plugin=None):
	'''	Backend used for a transfer syntax: the requested pydicom plugin or "imagecodecs", else the first
		installed backend. Raises a ValueError when nothing is installed.
	'''
	backends = display_compression_backends(transfer_syntax)
	if not backends:
		raise ValueError(('No encoder installed for %s. Install a pydicom encoding plugin (%s) or imagecodecs, '
			+ 'or choose another compression (`supported_display_compressions`).') % (
			UID(transfer_syntax).name, ', '.join(DISPLAY_VOLUME_PYDICOM_PLUGINS)))

	if not encoding_plugin:
		return backends[0]

	backend = encoding_plugin if encoding_plugin == DISPLAY_VOLUME_ENCODING_BACKEND_IMAGECODECS \
		else '%s:%s' % (DISPLAY_VOLUME_ENCODING_BACKEND_PYDICOM, encoding_plugin)
	if backend not in backends:
		raise ValueError('Encoding plugin %s is not available for %s. Installed: %s.' % (
			encoding_plugin, UID(transfer_syntax).name, ', '.join(backends)))

	return backend


def check_compression_sop_class(transfer_syntax, sop_class_uid):
	'''	Ensure that a transfer syntax is allowed for the image module of a SOP class: the 8 bit JPEG baseline
		syntax stores BitsAllocated 8, which the CT Image module does not allow (BitsAllocated 16).

		@raises ValueError
	'''
	if transfer_syntax is None or sop_class_uid is None:
		return

	bits = DCM_TRANSFER_SYNTAX_MAX_BITS.get(transfer_syntax)
	if bits is not None and bits <= 8 and str(sop_class_uid) in DISPLAY_VOLUME_SOP_CLASSES_16_BIT:
		raise ValueError('%s stores 8 bit samples (BitsAllocated 8), which %s does not allow (BitsAllocated 16). Use "jpeg" '
			'(12 bit) or a 16 bit capable compression, or encode the display volume with a SOP class that allows 8 bit samples '
			'(sopclass_uid, e.g. Secondary Capture Image Storage).' % (UID(transfer_syntax).name, UID(str(sop_class_uid)).name))


def check_compression_representation(transfer_syntax, representation):
	'''	Pixel representation to store for a transfer syntax. The JPEG (DCT) syntaxes hold unsigned samples
		of a fixed precision (PS3.5 Table 8.2.1-1): 8 bits for baseline (BitsAllocated 8) and 12 bits for
		extended (BitsAllocated 16). A representation of fewer bits is widened to that precision so that the
		pixel attributes match the codestream; a larger or a signed representation is rejected. The other
		syntaxes store the representation as given.

		@returns DisplayPixelRepresentation
		@raises ValueError
	'''
	if transfer_syntax is None:
		return representation

	bits = DCM_TRANSFER_SYNTAX_MAX_BITS.get(transfer_syntax)
	if bits is None:
		return representation

	if representation.bits_stored > bits or not representation.unsigned:
		raise ValueError('%s requires unsigned stored values of at most %d bits; the display volume needs %d bit %s values. '
			'Use a 16 bit capable compression (RLE, JPEG-LS, JPEG 2000, HTJ2K, JPEG Lossless, deflate).' % (
			UID(transfer_syntax).name, bits, representation.bits_stored, 'unsigned' if representation.unsigned else 'signed'))

	smin, smax = _stored_range(bits, True)
	return representation._replace(bits_stored=bits, dtype=np.uint8 if bits <= 8 else np.uint16, stored_min=smin, stored_max=smax)


def _imagecodecs_encode_frame(transfer_syntax, arr, bits_stored, options):
	'''	Encode one frame with imagecodecs for the transfer syntax (see DISPLAY_VOLUME_IMAGECODECS_CODECS)

		@returns bytes
	'''
	imagecodecs = _imagecodecs()
	codec = DISPLAY_VOLUME_IMAGECODECS_CODECS[transfer_syntax]
	unsigned = arr.dtype.kind == 'u'

	if codec in ('jpegls', 'jpeg8') and not unsigned:
		raise ValueError('%s via imagecodecs requires unsigned stored values (signed %s given). Use RLE, JPEG 2000, HTJ2K or '
			'deflate for signed values, the pyjpegls plugin for signed JPEG-LS, or preserve_rescale=True with an unsigned source.'
			% (UID(transfer_syntax).name, arr.dtype))

	if codec == 'jpeg8' and transfer_syntax in (JPEGLosslessSV1, JPEGLossless):
		# Lossless JPEG (ITU T.81 process 14, SOF3) with first-order prediction, as selection value 1 requires
		return imagecodecs.jpeg8_encode(arr, lossless=True, predictor=1, bitspersample=bits_stored)

	if codec == 'jpegls':
		near = int(options.get('jls_error', DISPLAY_VOLUME_JLS_ERROR_DEFAULT)) if transfer_syntax == JPEGLSNearLossless else 0
		return imagecodecs.jpegls_encode(arr, level=near, bitspersample=bits_stored)

	if codec == 'jpeg2k':
		lossless = transfer_syntax == JPEG2000Lossless
		if not lossless and 'j2k_cr' in options:
			raise ValueError('imagecodecs encodes JPEG 2000 by target PSNR (j2k_psnr), not by compression ratio (j2k_cr). '
				'Install pylibjpeg-openjpeg to use j2k_cr.')
		psnr = options.get('j2k_psnr', DISPLAY_VOLUME_J2K_PSNR_DEFAULT)
		psnr = float(psnr[0] if isinstance(psnr, (list, tuple)) else psnr)
		return imagecodecs.jpeg2k_encode(arr, level=0 if lossless else psnr, codecformat='j2k',
			bitspersample=bits_stored, reversible=lossless)

	if codec == 'htj2k':
		return imagecodecs.htj2k_encode(arr, level=0, bitspersample=bits_stored, reversible=True)

	if codec == 'jpeg8':
		quality = int(options.get('jpeg_quality', DISPLAY_VOLUME_JPEG_QUALITY_DEFAULT))
		# The representation was widened to the precision of the syntax (check_compression_representation)
		if transfer_syntax == JPEGBaseline8Bit:
			return imagecodecs.jpeg8_encode(arr.astype(np.uint8), level=quality)
		return imagecodecs.jpeg8_encode(arr, level=quality, bitspersample=bits_stored)

	raise ValueError('No imagecodecs encoder for %s' % UID(transfer_syntax).name)


def encode_display_pixel_data(dcm, arr, transfer_syntax=None, encoding_plugin=None, options=None):
	'''	Store the pixel data of a display slice in its dataset with a transfer syntax. The dataset holds
		the native pixel data of `arr` (array2dcmimg) and its pixel attributes (BitsStored, PixelRepresentation)
		when called. The SOP instance UID of the dataset is kept.

		* None: native pixel data, Explicit VR Little Endian (nothing to do)
		* Deflated Explicit VR Little Endian: native pixel data, the whole dataset is deflated when it is saved
		* encapsulated transfer syntaxes: the frame is compressed with a pydicom encoding plugin or imagecodecs
		  (`display_compression_backends`) and encapsulated

		@input dcm (pydicom.dataset.Dataset): dataset of the slice
		@input arr (numpy.ndarray): stored values of the slice (rows x columns)
		@input transfer_syntax (pydicom UID or None): see `display_compression_transfer_syntax`
		@input encoding_plugin (str, default=None): pydicom encoding plugin (gdcm, pydicom, pyjpegls,
			pylibjpeg) or "imagecodecs"; None uses the first installed backend
		@input options (dict, default=None): codec parameters: `jls_error` (JPEG-LS near-lossless maximum
			error), `j2k_psnr` (JPEG 2000 lossy target PSNR in dB, a number or a list of quality layers) or
			`j2k_cr` (compression ratio(s), pylibjpeg only), `jpeg_quality` (JPEG baseline/extended quality
			1-100). Scalars are accepted for every backend and converted to the shape the encoder requires.

		@returns int: number of bytes of the encoded pixel data (native size when not encapsulated)
	'''
	options = dict(options or {})

	if transfer_syntax is None:
		return arr.nbytes

	transfer_syntax = UID(transfer_syntax)
	if not transfer_syntax.is_encapsulated:
		dcm.file_meta.TransferSyntaxUID = transfer_syntax
		dcm_set_original_encoding(dcm)
		return arr.nbytes

	backend = _select_compression_backend(transfer_syntax, encoding_plugin)
	if backend == DISPLAY_VOLUME_ENCODING_BACKEND_IMAGECODECS:
		frame = _imagecodecs_encode_frame(transfer_syntax, arr, int(getattr(dcm, DCMHEADER_BITS_STORED)), options)
		dcm.file_meta.TransferSyntaxUID = transfer_syntax
		setattr(dcm, DCMHEADER_IMG_PIXELDATA, encapsulate([frame]))
		dcm[DCMHEADER_IMG_PIXELDATA].is_undefined_length = True
	else:
		# pydicom encoders: the lossy syntaxes need their quality parameter, else pyjpegls/pylibjpeg encode
		# losslessly under a lossy transfer syntax
		kwargs = pick(options, DISPLAY_VOLUME_PYDICOM_ENCODER_OPTIONS)
		if transfer_syntax == JPEGLSNearLossless:
			kwargs.setdefault('jls_error', DISPLAY_VOLUME_JLS_ERROR_DEFAULT)
		if transfer_syntax == JPEG2000 and 'j2k_cr' not in kwargs and 'j2k_psnr' not in kwargs:
			kwargs['j2k_psnr'] = DISPLAY_VOLUME_J2K_PSNR_DEFAULT
		# pylibjpeg takes the JPEG 2000 quality layers as lists; scalars are accepted here for every backend
		for option in ('j2k_psnr', 'j2k_cr'):
			if option in kwargs and not isinstance(kwargs[option], (list, tuple)):
				kwargs[option] = [kwargs[option]]
		plugin = backend.split(':', 1)[1]
		dcm.compress(transfer_syntax, arr, encoding_plugin=plugin, generate_instance_uid=False, **kwargs)

	dcm_set_original_encoding(dcm)
	return len(getattr(dcm, DCMHEADER_IMG_PIXELDATA))


def dcm_encode_display_volume(ivolume, size=None, spacing=None, grid=None, display=None,
		series_uid=None, series_number=None, series_description=None, series_headers=None, instance_headers=None,
		source_headers=DISPLAY_VOLUME_SOURCE_HEADERS, required_headers=DISPLAY_VOLUME_REQUIRED_HEADERS,
		copy_private_creators=None, sopclass_uid=None,
		derivation_description=None, reference_series=True, source_image_references=True,
		interpolator=sitk.sitkLinear, antialias=True, preserve_rescale=True, rescale_overflow=DISPLAY_VOLUME_RESCALE_OVERFLOW_ADAPT,
		dcm_compression=DISPLAY_VOLUME_COMPRESSION_DEFAULT, dcm_encoding_plugin=None, dcm_compression_options=None,
		dcm_encode_options=None, presave_callback=None, dry_run=False, hcache=None, tpool=None, threads=4):
	'''	Create a display series from the provided imaging volume: the volume is resampled to a coarser grid
		(`resample_display_image`) and each slice of the copy is encoded as a DICOM image of the same SOP
		class as the source (CT Image Storage for a CT) and uploaded to the imaging server of the series.
		The display series is part of the same study and frame of reference as the source series.

		Grid. The copy is a cube of `size` voxels (anisotropic spacing over the extent of the source) by
		default (DISPLAY_VOLUME_SIZES[0]), a grid of the given `spacing`, a caller-supplied `grid`, or the
		grid of an already resampled `display` image. `size` and `spacing` are exclusive.

		Headers. Patient, study, clinical trial and acquisition headers (`source_headers`) are copied from
		the first instance of the source series; the Type 2 headers of the CT Image module that the source
		lacks (`required_headers`: AcquisitionNumber, KVP) are written empty. The series receives a new
		SeriesInstanceUID, a SeriesNumber (`display_volume_series_number`, 1000 + voxel count) and
		SeriesDescription (source description + "Display N"). Each slice carries its ImagePositionPatient,
		ImageOrientationPatient, PixelSpacing, SliceThickness, SpacingBetweenSlices and SliceLocation
		computed from the grid; ImageType DERIVED\\SECONDARY\\AXIAL\\RESAMPLED; DerivationDescription and
		DerivationCodeSequence (spatial resampling); the lossy compression history of the source extended
		with the display encoding (`lossy_compression_history`); a ReferencedSeriesSequence naming the
		source series (series UID only) and a SourceImageSequence listing the source instances whose extent
		overlaps the slice.

		Pixel values. The copy keeps the intensity units of the source (HU for a CT); the samples are
		filtered and interpolated by the resampling, so individual values differ from the acquisition.
		The stored representation (`display_pixel_representation`) covers the full intensity range of the
		copy: with `preserve_rescale` the RescaleIntercept/RescaleSlope of the source are kept and
		BitsStored/PixelRepresentation are widened as needed; otherwise (or when the source rescale cannot
		hold the range) the intensities are stored as signed 16 bit with RescaleIntercept 0 and
		RescaleSlope 1, or with a derived rescale when even that overflows (`rescale_overflow`). Values
		are never saturated. BitsAllocated follows BitsStored (8 bit samples for at most 8 stored bits, else
		16 bit); fewer than 8 stored bits are widened to 8 (GDCM decodes an 8 bit container only when BitsStored
		equals BitsAllocated); CT Image Storage requires BitsAllocated 16 and BitsStored 12 to 16, so a narrower
		representation is widened to 12 bits for it (`stored_pixel_container`).

		Encoding. `dcm_compression` selects the transfer syntax (`display_compression_transfer_syntax`):
		RLE Lossless by default, None for uncompressed Explicit VR Little Endian, "deflate" (zlib of the
		whole dataset), "jpeg-lossless" (Huffman coded lossless JPEG), "jpeg-ls", "jpeg2000", "htj2k"
		(lossless), or the lossy "jpeg-ls-near", "jpeg2000-lossy", "jpeg" (12 bit DCT) and "jpeg-baseline"
		(8 bit DCT, BitsAllocated 8: not allowed for CT Image Storage, pass another `sopclass_uid`, e.g.
		Secondary Capture Image Storage, whose required ConversionType is written as "WSD" unless the
		source or `series_headers` provide one; see DISPLAY_VOLUME_SOP_CLASS_HEADERS). The
		encoder installed for the syntax is used (`display_compression_backends`; pydicom plugins first,
		then imagecodecs) unless `dcm_encoding_plugin` names one; the imagecodecs JPEG-LS and lossless
		JPEG encoders take unsigned stored values only. The DCT syntaxes store exactly 8 or 12 bits
		(`check_compression_representation`). Lossy syntaxes record LossyImageCompression "01" with the
		method and ratio of each slice. An unsupported or unavailable compression, or one the SOP class
		does not allow, raises a ValueError before anything is resampled or uploaded; a representation the
		syntax cannot hold raises before anything is uploaded.

		@input ivolume (SonadorImagingVolume): imaging volume initialized from the source series. Its
			`volume` is the source of the copy and its `series` the source of the headers and the target
			of the upload.
		@input size (int or three ints, default=None): voxel count of the copy, see `display_image_grid`;
			DISPLAY_VOLUME_SIZES[0] when no grid selector (size, spacing, grid, display) is given
		@input spacing (float or three floats, default=None): voxel size of the copy, see `display_image_grid`
		@input grid (DisplayImageGrid, default=None): grid of the copy; replaces `size` and `spacing`
		@input display (SimpleITK.Image, default=None): copy already resampled (`resample_display_image`);
			when provided its grid is used and no resampling is performed
		@input series_uid (str, default=new UID): SeriesInstanceUID of the display series
		@input series_number (int, default=`display_volume_series_number`): SeriesNumber of the display series
		@input series_description (str, default=`display_volume_series_description`): SeriesDescription
		@input series_headers (dict, default=None): DICOM attributes applied to every instance of the series
			after the computed series headers
		@input instance_headers (dict, default=None): DICOM attributes applied to every instance after the
			per-slice headers
		@input source_headers (iterable of keywords, default=DISPLAY_VOLUME_SOURCE_HEADERS): headers copied
			from the first instance of the source series
		@input required_headers (iterable of keywords, default=DISPLAY_VOLUME_REQUIRED_HEADERS): headers
			written empty when the source does not provide them (Type 2 attributes)
		@input copy_private_creators (iterable of str, default=None): private creators whose blocks are
			copied from the first instance of the source series to every instance (e.g. "ACME PLANNING 1.0").
			Costs one file request for the source instance.
		@input sopclass_uid (str, default=SOP class of the source, else CT Image Storage). The Type 1 headers
			the class requires beyond the CT headers (DISPLAY_VOLUME_SOP_CLASS_HEADERS) are copied from the
			source or written with their default.
		@input derivation_description (str, default=`display_volume_derivation_description`)
		@input reference_series (bool, default=True): write the ReferencedSeriesSequence naming the source series
		@input source_image_references (bool, default=True): write the per-slice SourceImageSequence
			(`imaging_volume_source_slices`, `source_slice_references`)
		@input interpolator (SimpleITK interpolator, default=sitkLinear): see `resample_display_image`
		@input antialias (bool, default=True): see `resample_display_image`
		@input preserve_rescale (bool, default=True): keep the rescale of the source when the intensities
			fit, see `display_pixel_representation`
		@input rescale_overflow (str, default="adapt"): "adapt" (fall back to the next representation, finally
			a derived rescale with a warning) or "raise" (ValueError as soon as the preferred representation,
			the rescale of the source with `preserve_rescale`, cannot hold the intensities), see
			`display_pixel_representation`
		@input dcm_compression (pydicom UID, alias or None, default=RLELossless): transfer syntax of the
			pixel data, see `display_compression_transfer_syntax`
		@input dcm_encoding_plugin (str, default=None): encoder: a pydicom encoding plugin (gdcm, pydicom,
			pyjpegls, pylibjpeg) or "imagecodecs"; None uses the first installed backend
		@input dcm_compression_options (dict, default=None): codec parameters, see `encode_display_pixel_data`
			(jls_error, j2k_psnr, j2k_cr, jpeg_quality)
		@input dcm_encode_options (dict, default=None): extra options of `dcm_encode_filedata` (dcm_ts,
			dcm_preamble, ...)
		@input dry_run (bool, default=False): when True the slices are encoded but not uploaded
		@input hcache (OrderedDict, default=new): header cache the study and series metadata are added to
		@input tpool (ThreadPoolExecutor, default=new pool of `threads` workers): pool the uploads run on

		@callback presave_callback (callable, default=None): invoked before each slice is saved. Signature:
			- k (int): index of the slice in the display volume (InstanceNumber k + 1)
			- dcmfile (pydicom.dataset.Dataset): dataset of the slice
			- display (SimpleITK.Image): the display volume

		@returns hcache (OrderedDict): metadata of the uploaded data keyed by DicomMetaKey: the study and
			the display series, whose DicomMeta `attrs` hold the display image (`image`), its grid (`grid`),
			the source series key (`source`), the transfer syntax (`transfer_syntax`), the stored pixel
			representation (`representation`) and the instances (`instances`, DicomMeta per slice keyed by
			the SOPInstanceUID written to the file)
	'''
	hcache = hcache if isinstance(hcache, (OrderedDict, dict)) else OrderedDict()
	series_headers = series_headers or {}
	instance_headers = instance_headers or {}
	dcm_encode_options = dict(dcm_encode_options or {})
	dcm_encode_options.setdefault('dcm_implicit_vr', False)

	# Ensure that the imaging volume was initialized from an imaging series
	series = getattr(ivolume, 'series', None)
	if not isinstance(series, ImagingSeries):
		raise ValueError('Unable to encode display volume, invalid imaging volume: an imaging volume initialized '
			+ 'from an ImagingSeries resource is required.')

	# Transfer syntax and encoder, checked before any work is done
	transfer_syntax = display_compression_transfer_syntax(dcm_compression)
	_select_compression_backend(transfer_syntax, dcm_encoding_plugin)

	# Headers of the first source instance (one request, cached on the instance); the SOP class of the display
	# series must allow the transfer syntax
	src_tags = series.dcm0.tags or {}
	sop_class_uid = str(sopclass_uid or src_tags.get(DCMHEADER_SOP_CLASS_UID) or DCM_MEDIA_STORAGE_CT_IMAGE)
	check_compression_sop_class(transfer_syntax, sop_class_uid)

	img = sitk_image(ivolume)
	if img.GetDimension() != 3 or img.GetNumberOfComponentsPerPixel() != 1:
		raise ValueError('Unable to encode display volume for series %s: a 3D scalar volume is required (%dD, %d components).'
			% (series.pk, img.GetDimension(), img.GetNumberOfComponentsPerPixel()))

	src_grid = image_grid(img)

	# Resample the volume to the display grid: the default size applies when no grid selector was given;
	# an explicit size together with an explicit spacing is rejected by display_image_grid
	if display is not None:
		display = sitk_image(display)
		grid = image_grid(display)
	else:
		if grid is None and size is None and spacing is None:
			size = DISPLAY_VOLUME_SIZES[0]
		grid = grid or display_image_grid(img, size=size, spacing=spacing)
		logger.info('series=%s: resample %s voxels (%s mm) to display grid %s voxels (%s mm)' % (
			series.pk, 'x'.join(str(v) for v in src_grid.size), ' x '.join('%.4g' % v for v in src_grid.spacing),
			'x'.join(str(v) for v in grid.size), ' x '.join('%.4g' % v for v in grid.spacing)))
		display = resample_display_image(img, grid=grid, interpolator=interpolator, antialias=antialias)

	# Stored pixel representation covering the intensities of the copy, in the container the SOP class allows,
	# adjusted to the precision of the transfer syntax
	representation = display_pixel_representation(display, src_tags, preserve_rescale=preserve_rescale, rescale_overflow=rescale_overflow)
	representation = stored_pixel_container(representation, sop_class_uid)
	representation = check_compression_representation(transfer_syntax, representation)

	# Series headers: white listed headers of the source, identifiers of the display series
	sx_hdata = pick(src_tags, source_headers)
	for header in (required_headers or ()):
		sx_hdata.setdefault(header, None)

	# Type 1 headers of the SOP class that the (CT oriented) source headers do not carry, e.g. ConversionType of
	# Secondary Capture: the value of the source when it has one, else the default of the class
	for header, default in DISPLAY_VOLUME_SOP_CLASS_HEADERS.get(sop_class_uid, {}).items():
		sx_hdata[header] = src_tags.get(header) or default

	series_uid = series_uid or generate_uid()
	sx_hdata[DCMHEADER_SERIES_INSTANCE_UID] = series_uid
	sx_hdata[DCMHEADER_SERIES_NUMBER] = int(series_number) if series_number is not None else display_volume_series_number(grid.size)
	sx_hdata[DCMHEADER_SERIES_DESCRIPTION] = series_description or display_volume_series_description(
		src_tags.get(DCMHEADER_SERIES_DESCRIPTION) or series.description, grid.size)
	sx_hdata[DCMHEADER_MODALITY] = modality = src_tags.get(DCMHEADER_MODALITY) or series.modality or DCM_MODALITY_CT
	sx_hdata[DCMHEADER_SOP_CLASS_UID] = sop_class_uid

	# The display series shares the frame of reference of the source
	if not sx_hdata.get(DCMHEADER_FRAME_OF_REFERENCE_UID):
		sx_hdata[DCMHEADER_FRAME_OF_REFERENCE_UID] = generate_uid()
		logger.warning('series=%s: no FrameOfReferenceUID on the source series, a new frame of reference %s is used for the display series'
			% (series.pk, sx_hdata[DCMHEADER_FRAME_OF_REFERENCE_UID]))

	# Series timestamp
	sx_ts = datetime.datetime.now()
	sx_hdata[DCMHEADER_SERIES_DATE] = sx_ts.strftime(DCM_DATE_STRFORMAT)
	sx_hdata[DCMHEADER_SERIES_TIME] = sx_ts.strftime(DCM_TIME_STRFORMAT)

	# Geometry shared by the slices
	sx_hdata[DCMHEADER_IMAGE_ORIENTATION_PATIENT] = image_orientation_patient(display)
	sx_hdata[DCMHEADER_PIXEL_SPACING] = [float(grid.spacing[1]), float(grid.spacing[0])]
	sx_hdata[DCMHEADER_SLICE_THICKNESS] = float(grid.spacing[2])
	sx_hdata[DCMHEADER_SPACING_BETWEEN_SLICES] = float(grid.spacing[2])

	# Derived image: type, provenance and equipment
	sx_hdata[DCMHEADER_IMAGE_TYPE] = list(DCM_IMAGE_TYPE_DISPLAY_VOLUME)
	sx_hdata[DCMHEADER_DERIVATION_DESCRIPTION] = derivation_description or display_volume_derivation_description(
		series.series_uid, src_grid, grid, interpolator=interpolator, antialias=antialias)
	sx_hdata[DCMHEADER_DERIVATION_CODE_SEQUENCE] = Sequence([srcode2dataset(DCMCODE_DERIVATION_SPATIAL_RESAMPLING, omit_empty=True)])
	sx_hdata.update(lossy_compression_history(src_tags, transfer_syntax))
	sx_hdata[DCMHEADER_MANUFACTURER] = DISPLAY_VOLUME_MANUFACTURER
	sx_hdata[DCMHEADER_SOFTWARE_VERSIONS] = DISPLAY_VOLUME_SOFTWARE_VERSIONS_TEMPLATE % {'sitk': sitk.Version_VersionString()}

	if reference_series:
		sx_ref = Dataset()
		setattr(sx_ref, DCMHEADER_SERIES_INSTANCE_UID, series.series_uid)
		sx_hdata[DCMHEADER_SR_REF_SERIES_SEQ] = Sequence([sx_ref])

	# Pixel value rescale of the stored representation
	sx_hdata[DCMHEADER_RESCALE_INTERCEPT] = representation.intercept
	sx_hdata[DCMHEADER_RESCALE_SLOPE] = representation.slope
	rescale_type = src_tags.get(DCMHEADER_RESCALE_TYPE) or (DCM_RESCALE_TYPE_HU if modality == DCM_MODALITY_CT else None)
	if rescale_type:
		sx_hdata[DCMHEADER_RESCALE_TYPE] = rescale_type

	sx_hdata.update(series_headers)

	# Source instances referenced by the slices
	source_slices, pitch = [], None
	if source_image_references:
		source_slices = imaging_volume_source_slices(ivolume, normal=image_slice_normal(display), sop_class_uid=sx_hdata[DCMHEADER_SOP_CLASS_UID])
		pitch = source_slice_pitch(source_slices, default=float(src_grid.spacing[2]))
		if not source_slices:
			logger.warning('series=%s: no source instance positions available, the display slices carry no SourceImageSequence' % series.pk)

	# Private blocks copied from the first source instance
	src_ds = series.dcm0.dcmfile(cache=True) if copy_private_creators else None

	# Encode the slices and upload them. Encoding runs in this thread, uploads on the pool with a bounded
	# number of pending transfers.
	own_tpool = tpool is None
	tpool = tpool or ThreadPoolExecutor(max_workers=threads)
	pending = []
	hcache_instances = OrderedDict()
	normal = image_slice_normal(display)
	description = sx_hdata.get(DCMHEADER_SERIES_DESCRIPTION)

	def _wait(limit):
		while len(pending) > limit:
			pending.pop(0).result()

	try:
		for k, position in enumerate(image_slice_positions(display)):

			# Slice headers
			slice_hdata = copy.deepcopy(sx_hdata)
			slice_hdata[DCMHEADER_SOP_INSTANCE_UID] = generate_uid()
			slice_hdata[DCMHEADER_INSTANCE_NUMBER] = k + 1
			slice_hdata[DCMHEADER_IMAGE_POSITION_PATIENT] = [float(v) for v in position]
			slice_z = float(np.dot(_point_array(position), normal))
			slice_hdata[DCMHEADER_SLICE_LOCATION] = slice_z

			slice_sources = None
			if source_slices:
				slice_sources = source_slice_references(source_slices, slice_z, grid.spacing[2], pitch)
				slice_hdata[DCMHEADER_SR_SOURCE_IMAGE_SEQ] = slice_sources

			slice_hdata.update(instance_headers)

			# Pixel data: rows (y) by columns (x) of slice k, stored with the representation
			stored = stored_pixel_values(sitk.GetArrayFromImage(display[:, :, k]), representation)

			dcm_slice = array2dcmimg(stored, dcm_attrs=slice_hdata, dcm_compression=None, **dcm_encode_options)
			setattr(dcm_slice, DCMHEADER_BITS_STORED, representation.bits_stored)
			setattr(dcm_slice, DCMHEADER_HIGH_BIT, representation.bits_stored - 1)

			# Compression: the SOP instance UID is kept, lossy syntaxes record their ratio
			encoded_size = encode_display_pixel_data(dcm_slice, stored, transfer_syntax, dcm_encoding_plugin, dcm_compression_options)
			if transfer_syntax in DCM_TRANSFER_SYNTAX_LOSSY_METHODS:
				for header, value in lossy_compression_history(src_tags, transfer_syntax, stored.nbytes / float(encoded_size or 1)).items():
					setattr(dcm_slice, header, value)

			if src_ds is not None:
				copy_private_blocks(src_ds, dcm_slice, copy_private_creators)

			if callable(presave_callback):
				presave_callback(k, dcm_slice, display)

			# Identifiers as written: the dataset and its file meta must agree
			slice_uid = str(getattr(dcm_slice, DCMHEADER_SOP_INSTANCE_UID))
			if str(getattr(dcm_slice.file_meta, DCMHEADER_MEDIA_STORAGE_SOP_INSTANCE_UID, '')) != slice_uid:
				setattr(dcm_slice.file_meta, DCMHEADER_MEDIA_STORAGE_SOP_INSTANCE_UID, slice_uid)

			# Stream for upload
			dcm_stream = BytesIO()
			dcm_slice.save_as(dcm_stream)
			dcm_stream.seek(0)

			if not dry_run:
				_wait(2 * threads)
				pending.append(tpool.submit(series.pacs.upload_image, dcm_stream))

			logger.debug('series=%s display-series=%s/%d: slice encoded, position=%s sources=%d bytes=%d%s' % (
				series.pk, series_uid, k + 1, tuple(round(v, 3) for v in position),
				len(slice_sources) if slice_sources is not None else 0, encoded_size, ' (dry run)' if dry_run else ''))

			# Instance metadata, keyed by the SOP instance UID written to the file
			hmeta_idx = DicomMetaKey(IMAGING_SERVER_RESOURCE_IMAGE, DCMHEADER_SOP_INSTANCE_UID, slice_uid)
			hcache_instances[hmeta_idx] = DicomMeta('%s %d' % (description, k + 1), modality, meta=hmeta_idx, attrs={
				'number': k + 1, 'position': position, 'z': slice_z,
				'sources': len(slice_sources) if slice_sources is not None else None,
				'encoded_size': encoded_size,
			})

		_wait(0)
	finally:
		if own_tpool:
			tpool.shutdown(wait=True)

	if not dry_run:
		logger.info('patient=%s series=%s: DCM upload of display series %s "%s" (%d slices, %s voxels, %s) successful' % (
			series.model_patient.patientid, series.pk, series_uid, description, display.GetDepth(), 'x'.join(str(v) for v in grid.size),
			transfer_syntax.name if transfer_syntax else 'uncompressed'))
	else:
		logger.info('Dry run: patient=%s series=%s: display series %s "%s" (%d slices, %s voxels, %s) processed successfully' % (
			series.model_patient.patientid, series.pk, series_uid, description, display.GetDepth(), 'x'.join(str(v) for v in grid.size),
			transfer_syntax.name if transfer_syntax else 'uncompressed'))

	# Add study metadata to the hcache (if not already present)
	if not hcache.get(series.parent.hmeta_key):
		hcache[series.parent.hmeta_key] = series.parent.hmeta

	# Add series metadata to the hcache
	hmeta_sx = DicomMetaKey(IMAGING_SERVER_RESOURCE_SERIES, DCMHEADER_SERIES_INSTANCE_UID, series_uid)
	hcache[hmeta_sx] = DicomMeta(description, modality, meta=hmeta_sx, attrs={
		'image': display, 'grid': grid, 'source': series.hmeta_key, 'instances': hcache_instances, 'dry_run': dry_run,
		'transfer_syntax': transfer_syntax, 'representation': representation,
	})

	return hcache


def dcm_encode_display_volumes(ivolume, sizes=DISPLAY_VOLUME_SIZES, hcache=None, **kwargs):
	'''	Create one display series per size from the provided imaging volume (`dcm_encode_display_volume`
		for each of `sizes`). Options that identify a single series (size, series_uid, series_number,
		series_description, grid, display, spacing) cannot be shared and are rejected.

		@input ivolume (SonadorImagingVolume): imaging volume initialized from the source series
		@input sizes (iterable of int or three ints, default=DISPLAY_VOLUME_SIZES): voxel counts of the copies
		@input hcache (OrderedDict, default=new): header cache the study and series metadata are added to

		@returns hcache (OrderedDict)
	'''
	hcache = hcache if isinstance(hcache, (OrderedDict, dict)) else OrderedDict()

	exclusive = [k for k in ('size', 'series_uid', 'series_number', 'series_description', 'grid', 'display', 'spacing') if k in kwargs]
	if exclusive:
		raise ValueError('Unable to encode display volumes: %s identify a single series and cannot be shared across sizes. '
			'Call dcm_encode_display_volume for each series instead.' % ', '.join(exclusive))

	for size in sizes:
		dcm_encode_display_volume(ivolume, size=size, hcache=hcache, **kwargs)

	return hcache
