import itertools

import numpy as np
import SimpleITK as sitk
from pydicom.sr.codedict import codes as dcmcodes
from pydicom.uid import RLELossless, DeflatedExplicitVRLittleEndian, JPEGBaseline8Bit, JPEGExtended12Bit, \
	JPEGLossless, JPEGLosslessSV1, JPEGLSLossless, JPEGLSNearLossless, JPEG2000Lossless, JPEG2000, HTJ2KLossless

from sonador.apisettings import DCMHEADER_SPECIFIC_CHARSET, DCMHEADER_MODALITY, DCMHEADER_SOP_CLASS_UID, \
	DCMHEADER_FRAME_OF_REFERENCE_UID, DCMHEADER_POSITION_REFERENCE_INDICATOR, DCMHEADER_PATIENT_POSITION, \
	DCMHEADER_BODY_PART_EXAMINED, DCMHEADER_PROTOCOL_NAME, DCMHEADER_LATERALITY
from sonador.apisettings.media import DCM_MEDIA_PATIENT_TAGS, DCM_MEDIA_STUDY_TAGS, DCM_MEDIA_CLINICAL_TRIAL_TAGS


# Direction matrix that can be used to align imaging volumes in a neutral coordinate
# system without "pose" data or to calculate coordinate transforms.

NUETRAL_3D_POSE = np.array([
	[0, 0, -1.0],
	[1.0, 0, 0],
	[0, -1.0, 0]
])


LABELMAP_DTYPE_DEFAULT = np.uint8

IDENTIY_3D = np.identity(3)


IMAGING_VOLUME_TYPE_NUMPY = 'numpy'
IMAGING_VOLUME_TYPE_ITK = 'itk'


DCM_VOLUME_LEFT_POSTERIOR_SUPERIOR = 'LPS'
DCM_VOLUME_RIGHT_ANTERIOR_INFERIOR = 'RAI'
DCM_VOLUME_RIGHT_ANTERIOR_SUPERIOR = 'RAS'


# DICOM headers that the Sonador IO constants do not define: pixel value rescaling, display windows,
# slice spacing, acquisition parameters and the provenance of derived images.

DCMHEADER_RESCALE_INTERCEPT = 'RescaleIntercept'
DCMHEADER_RESCALE_SLOPE = 'RescaleSlope'
DCMHEADER_RESCALE_TYPE = 'RescaleType'
DCM_RESCALE_TYPE_HU = 'HU'

DCMHEADER_WINDOW_CENTER = 'WindowCenter'
DCMHEADER_WINDOW_WIDTH = 'WindowWidth'

DCMHEADER_SPACING_BETWEEN_SLICES = 'SpacingBetweenSlices'
DCMHEADER_KVP = 'KVP'
DCMHEADER_ACQUISITION_NUMBER = 'AcquisitionNumber'

DCMHEADER_DERIVATION_DESCRIPTION = 'DerivationDescription'
DCMHEADER_DERIVATION_CODE_SEQUENCE = 'DerivationCodeSequence'
DCMHEADER_PURPOSE_OF_REFERENCE_CODE_SEQUENCE = 'PurposeOfReferenceCodeSequence'

DCMHEADER_LOSSY_IMAGE_COMPRESSION = 'LossyImageCompression'
DCMHEADER_LOSSY_IMAGE_COMPRESSION_RATIO = 'LossyImageCompressionRatio'
DCMHEADER_LOSSY_IMAGE_COMPRESSION_METHOD = 'LossyImageCompressionMethod'
DCM_LOSSY_IMAGE_COMPRESSION_NONE = '00'
DCM_LOSSY_IMAGE_COMPRESSION_LOSSY = '01'

# CT Image Storage and Secondary Capture Image Storage SOP classes (PS3.4 B.5)
DCM_MEDIA_STORAGE_CT_IMAGE = '1.2.840.10008.5.1.4.1.1.2'
DCM_MEDIA_STORAGE_SECONDARY_CAPTURE = '1.2.840.10008.5.1.4.1.1.7'

# Secondary Capture equipment (PS3.3 C.8.6.1): ConversionType is Type 1; "WSD" (workstation) describes
# an image computed by software from other images
DCMHEADER_CONVERSION_TYPE = 'ConversionType'
DCM_CONVERSION_TYPE_WORKSTATION = 'WSD'

# Derivation of a resampled copy (PS3.16 CID 7203) and the purpose of its source image references (CID 7202)
DCMCODE_DERIVATION_SPATIAL_RESAMPLING = dcmcodes.DCM.SpatialResampling
DCMCODE_PURPOSE_SOURCE_IMAGE_PROCESSING = dcmcodes.DCM.SourceImageForImageProcessingOperation

# SimpleITK metadata dictionary keys (group|element) of the per-slice attributes the series reader keeps
SITK_META_SOP_CLASS_UID = '0008|0016'
SITK_META_SOP_INSTANCE_UID = '0008|0018'
SITK_META_IMAGE_POSITION_PATIENT = '0020|0032'


# Display volumes: copies of an imaging volume resampled to a coarser grid so that viewers and planning
# tools can load them faster than the acquisition. The default sizes are cubes of equal voxel counts
# (anisotropic spacing) covering the full extent of the source.

DISPLAY_VOLUME_SIZES = (512, 256, 171)

# Series number of a display series: base + the largest voxel count (512 -> 1512)
DISPLAY_VOLUME_SERIES_NUMBER_BASE = 1000

# Series description templates: `description` is the description of the source series, `size` the
# voxel count label ("512" for a cube, "512x512x300" otherwise). The display label and the size are
# placed in parentheses after the source description ("PELVIS _FEM (Display 512)").
DISPLAY_VOLUME_SERIES_DESCRIPTION_TEMPLATE = '%(description)s (Display %(size)s)'
DISPLAY_VOLUME_SERIES_DESCRIPTION_DEFAULT = 'Display Volume (%(size)s)'

# Image Type of a display slice (PS3.3 C.7.6.1.1.2): derived, secondary, axial, resampled copy
DCM_IMAGE_TYPE_DISPLAY_VOLUME = ['DERIVED', 'SECONDARY', 'AXIAL', 'RESAMPLED']

# Equipment headers of a display series (the derived images are created by Sonador 3D, not by the scanner)
DISPLAY_VOLUME_MANUFACTURER = 'Sonador 3D'
DISPLAY_VOLUME_SOFTWARE_VERSIONS_TEMPLATE = 'sonador3d_imaging SimpleITK %(sitk)s'

# Headers copied from the first instance of the source series to every instance of a display series:
# patient, study and clinical trial identifiers, the frame of reference and the acquisition attributes
# that still describe the resampled data. SeriesInstanceUID, SeriesNumber, SeriesDescription, the
# per-slice geometry, the pixel attributes and the lossy compression history are set by the encoder.
DISPLAY_VOLUME_SOURCE_HEADERS = set(itertools.chain(
	DCM_MEDIA_PATIENT_TAGS, DCM_MEDIA_STUDY_TAGS, DCM_MEDIA_CLINICAL_TRIAL_TAGS, (
		DCMHEADER_SPECIFIC_CHARSET, DCMHEADER_MODALITY, DCMHEADER_SOP_CLASS_UID,
		DCMHEADER_FRAME_OF_REFERENCE_UID, DCMHEADER_POSITION_REFERENCE_INDICATOR, DCMHEADER_PATIENT_POSITION,
		DCMHEADER_BODY_PART_EXAMINED, DCMHEADER_PROTOCOL_NAME, DCMHEADER_LATERALITY, DCMHEADER_KVP,
		DCMHEADER_ACQUISITION_NUMBER, DCMHEADER_RESCALE_TYPE, DCMHEADER_WINDOW_CENTER, DCMHEADER_WINDOW_WIDTH,
	)))

# Type 2 headers of the CT Image module (PS3.3 C.8.2.1) written empty when the source does not provide them
DISPLAY_VOLUME_REQUIRED_HEADERS = (DCMHEADER_ACQUISITION_NUMBER, DCMHEADER_KVP)

# Type 1 headers that SOP classes other than CT Image Storage require and the source headers above do not
# carry, with the value written when the source does not provide one (a value of the source or of
# `series_headers` is kept): Secondary Capture Image Storage requires ConversionType
DISPLAY_VOLUME_SOP_CLASS_HEADERS = {
	DCM_MEDIA_STORAGE_SECONDARY_CAPTURE: {DCMHEADER_CONVERSION_TYPE: DCM_CONVERSION_TYPE_WORKSTATION},
}

# Largest kernel (in voxels) of the anti-aliasing Gaussian applied before a volume is coarsened
DISPLAY_VOLUME_ANTIALIAS_KERNEL_WIDTH = 64

# Names of the SimpleITK interpolators, for the derivation description of a display series
SITK_INTERPOLATOR_LABELS = {
	sitk.sitkNearestNeighbor: 'nearest neighbour',
	sitk.sitkLinear: 'linear',
	sitk.sitkBSpline: 'B-spline',
	sitk.sitkGaussian: 'Gaussian',
	sitk.sitkLabelGaussian: 'label Gaussian',
	sitk.sitkHammingWindowedSinc: 'Hamming windowed sinc',
	sitk.sitkCosineWindowedSinc: 'cosine windowed sinc',
	sitk.sitkWelchWindowedSinc: 'Welch windowed sinc',
	sitk.sitkLanczosWindowedSinc: 'Lanczos windowed sinc',
	sitk.sitkBlackmanWindowedSinc: 'Blackman windowed sinc',
}


# Compression of display volumes. `dcm_compression` accepts None (uncompressed Explicit VR Little Endian),
# a pydicom transfer syntax UID or one of the aliases below. The syntaxes are those the primary consumers
# decode: Cornerstone3D (@cornerstonejs/dicom-image-loader) and SimpleITK/GDCM read all of them. There is
# no DICOM transfer syntax for LZW, LZ4 or standalone Huffman/entropy coding: "huffman" maps to lossless
# JPEG (Huffman coded predictive coding), "zlib" to the deflated transfer syntax (zlib of the whole
# dataset), "dct" to 12 bit JPEG. Every slice is encoded as its own frame; the JPEG 2000 Part 2
# multi-component syntaxes (PS3.5 A.4.4, wavelet transform across frames) are not offered. High-Throughput
# JPEG 2000 is offered lossless only ("htj2k"); the lossy HTJ2K syntax is not offered by these helpers.

DISPLAY_VOLUME_COMPRESSION_DEFAULT = RLELossless

DISPLAY_VOLUME_TRANSFER_SYNTAXES = (
	DeflatedExplicitVRLittleEndian,  # zlib, lossless, whole dataset
	RLELossless,                     # PackBits run length, lossless
	JPEGLosslessSV1,                 # lossless JPEG (Huffman, first-order prediction), process 14 SV1
	JPEGLossless,                    # lossless JPEG, process 14
	JPEGLSLossless,                  # JPEG-LS lossless
	JPEGLSNearLossless,              # JPEG-LS near-lossless (bounded error, lossy)
	JPEG2000Lossless,                # JPEG 2000 reversible wavelet, lossless
	JPEG2000,                        # JPEG 2000 irreversible wavelet, lossy
	HTJ2KLossless,                   # High-Throughput JPEG 2000 (Part 15), lossless
	JPEGExtended12Bit,               # JPEG DCT 12 bit, lossy
	JPEGBaseline8Bit,                # JPEG DCT 8 bit, lossy
)

DISPLAY_VOLUME_COMPRESSION_ALIASES = {
	'none': None, 'uncompressed': None, 'explicit': None,
	'deflate': DeflatedExplicitVRLittleEndian, 'deflated': DeflatedExplicitVRLittleEndian, 'zlib': DeflatedExplicitVRLittleEndian,
	'rle': RLELossless,
	'jpeg-lossless': JPEGLosslessSV1, 'jpeg-lossless-sv1': JPEGLosslessSV1, 'huffman': JPEGLosslessSV1,
	'jpeg-lossless-p14': JPEGLossless,
	'jpeg-ls': JPEGLSLossless, 'jpegls': JPEGLSLossless,
	'jpeg-ls-near': JPEGLSNearLossless, 'jpegls-near': JPEGLSNearLossless,
	'jpeg2000': JPEG2000Lossless, 'j2k': JPEG2000Lossless,
	'jpeg2000-lossy': JPEG2000, 'j2k-lossy': JPEG2000,
	'htj2k': HTJ2KLossless, 'htj2k-lossless': HTJ2KLossless,
	'jpeg': JPEGExtended12Bit, 'dct': JPEGExtended12Bit, 'jpeg-12': JPEGExtended12Bit,
	'jpeg-baseline': JPEGBaseline8Bit, 'jpeg-8': JPEGBaseline8Bit,
}

# Lossy transfer syntaxes and the LossyImageCompressionMethod recorded for them (PS3.3 C.7.6.1.1.5)
DCM_LOSSY_IMAGE_COMPRESSION_METHOD_JPEG = 'ISO_10918_1'
DCM_LOSSY_IMAGE_COMPRESSION_METHOD_JPEG_LS = 'ISO_14495_1'
DCM_LOSSY_IMAGE_COMPRESSION_METHOD_JPEG_2000 = 'ISO_15444_1'

DCM_TRANSFER_SYNTAX_LOSSY_METHODS = {
	JPEGBaseline8Bit: DCM_LOSSY_IMAGE_COMPRESSION_METHOD_JPEG,
	JPEGExtended12Bit: DCM_LOSSY_IMAGE_COMPRESSION_METHOD_JPEG,
	JPEGLSNearLossless: DCM_LOSSY_IMAGE_COMPRESSION_METHOD_JPEG_LS,
	JPEG2000: DCM_LOSSY_IMAGE_COMPRESSION_METHOD_JPEG_2000,
}

# Sample precision of the DCT based JPEG syntaxes (PS3.5 Table 8.2.1-1): unsigned samples of exactly 8 bits
# (baseline, BitsAllocated 8) or 12 bits (extended, BitsAllocated 16). Representations of fewer bits are
# widened to the precision, larger or signed ones are rejected.
DCM_TRANSFER_SYNTAX_MAX_BITS = {
	JPEGBaseline8Bit: 8,
	JPEGExtended12Bit: 12,
}

# SOP classes whose image module fixes the pixel container: CT Image Storage requires BitsAllocated 16 and
# BitsStored 12 to 16 (PS3.3 C.8.2.1.1.4, C.8.2.1.1.5), hence no 8 bit transfer syntax and at least 12 stored
# bits. Every other class stores 8 bit samples (BitsAllocated 8) when at most 8 bits are stored, 16 bit otherwise.
DISPLAY_VOLUME_SOP_CLASSES_16_BIT = (DCM_MEDIA_STORAGE_CT_IMAGE,)
DISPLAY_VOLUME_SOP_CLASSES_16_BIT_MIN_BITS_STORED = 12

# Smallest BitsStored written: GDCM decodes an 8 bit container only when BitsStored equals BitsAllocated
# (gdcmImageCodec), so a representation of fewer than 8 bits is widened to 8, which never saturates
DISPLAY_VOLUME_MIN_BITS_STORED = 8

# Encoders: pydicom encoding plugins (RLE: pydicom/gdcm/pylibjpeg, JPEG-LS: pyjpegls, JPEG 2000: pylibjpeg-openjpeg)
# and the imagecodecs codecs that fill the gaps (lossless JPEG, JPEG 2000, HTJ2K, JPEG-LS, 8/12 bit JPEG). The
# imagecodecs JPEG-LS and JPEG encoders take unsigned samples only; JPEG 2000 and HTJ2K take signed ones too.
DISPLAY_VOLUME_ENCODING_BACKEND_NATIVE = 'native'
DISPLAY_VOLUME_ENCODING_BACKEND_PYDICOM = 'pydicom'
DISPLAY_VOLUME_ENCODING_BACKEND_IMAGECODECS = 'imagecodecs'
DISPLAY_VOLUME_PYDICOM_PLUGINS = ('gdcm', 'pydicom', 'pyjpegls', 'pylibjpeg')
DISPLAY_VOLUME_PYDICOM_ENCODER_OPTIONS = ('jls_error', 'j2k_cr', 'j2k_psnr')

DISPLAY_VOLUME_IMAGECODECS_CODECS = {
	JPEGLosslessSV1: 'jpeg8',  # libjpeg-turbo 3 lossless mode (SOF3), predictor 1
	JPEGLossless: 'jpeg8',
	JPEGLSLossless: 'jpegls',
	JPEGLSNearLossless: 'jpegls',
	JPEG2000Lossless: 'jpeg2k',
	JPEG2000: 'jpeg2k',
	HTJ2KLossless: 'htj2k',
	JPEGExtended12Bit: 'jpeg8',
	JPEGBaseline8Bit: 'jpeg8',
}

# Defaults of the lossy codec parameters (`dcm_compression_options`): JPEG-LS maximum sample error,
# JPEG 2000 target PSNR (dB; a number or a list of quality layers), JPEG quality (1-100)
DISPLAY_VOLUME_JLS_ERROR_DEFAULT = 2
DISPLAY_VOLUME_J2K_PSNR_DEFAULT = 60.0
DISPLAY_VOLUME_JPEG_QUALITY_DEFAULT = 90

# Behaviour when the intensities of a display volume cannot be stored with the preferred pixel
# representation (the rescale of the source, else signed 16 bit with rescale 0/1): fall back to the next
# representation and finally to a derived, quantizing rescale (warning), or raise a ValueError at the
# first representation that does not hold the values
DISPLAY_VOLUME_RESCALE_OVERFLOW_ADAPT = 'adapt'
DISPLAY_VOLUME_RESCALE_OVERFLOW_RAISE = 'raise'
