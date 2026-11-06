'''	Tests of the display volume helpers (`sonador3d_imaging.helpers.dcm_encode_display_volume`): a synthetic
	CT series written with pydicom, read through SimpleITK/GDCM, encoded against a mocked ImagingSeries whose
	uploads are captured in memory, and decoded again with pydicom and SimpleITK.

	The series has differing RescaleSlope values per instance so that the resampled intensities exceed the
	stored range of the first instance (the regression of MR !2, P1).
'''
import io, os, struct, tempfile
from unittest import mock

import numpy as np
import pytest
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import generate_frames
from pydicom.uid import generate_uid, CTImageStorage, SecondaryCaptureImageStorage, ExplicitVRLittleEndian, \
	DeflatedExplicitVRLittleEndian, RLELossless, JPEGLosslessSV1, JPEGLSLossless, JPEGLSNearLossless, JPEG2000Lossless, JPEG2000, \
	HTJ2KLossless, JPEGExtended12Bit, JPEGBaseline8Bit

from sonador.apisettings import DicomMetaKey, DicomMeta, IMAGING_SERVER_RESOURCE_SERIES, IMAGING_SERVER_RESOURCE_STUDY
from sonador.imaging.orthanc import ImagingSeries
from sonador.imaging.orthanc.base import DcmInstance

import sonador3d_imaging.helpers as H
from sonador3d_imaging.volume import SonadorImagingVolume


ROWS, COLS, SLICES = 12, 10, 6
SPACING = (0.7, 0.9, 1.5)  # x, y, z (mm)
INTERCEPT = -1024
STORED_MAX = 3500


def _pattern(k, stored_max=STORED_MAX, bits_stored=12, signed=False, bits_allocated=None):
	'''	Stored values of slice k: a bump with its peak at the center (peak = stored_max), unsigned 12 bit by
		default; centred on zero when signed. The dtype follows bits_allocated (bits_stored rounded up to 8 or 16).
	'''
	bits_allocated = bits_allocated or (8 if bits_stored <= 8 else 16)
	yy, xx = np.mgrid[0:ROWS, 0:COLS]
	bump = np.exp(-((yy - ROWS / 2.0)**2 + (xx - COLS / 2.0)**2) / 8.0)
	values = np.rint(stored_max * (0.2 + 0.8 * bump) - 20 * k)
	if signed:
		return (values - stored_max // 2).clip(-(2**(bits_stored - 1)), 2**(bits_stored - 1) - 1).astype(np.int8 if bits_allocated <= 8 else np.int16)
	return values.clip(0, 2**bits_stored - 1).astype(np.uint8 if bits_allocated <= 8 else np.uint16)


def write_ct_series(path, slopes=(1, 2, 2, 2, 2, 2), acquisition_number=7, lossy=None, bits_stored=12, intercept=INTERCEPT,
		stored_max=STORED_MAX, signed=False, bits_allocated=16, sop_class=CTImageStorage, conversion_type=None):
	'''	Write a CT series of len(slopes) instances; returns the file names (slice order)
	'''
	study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
	files = []
	for k, slope in enumerate(slopes):
		ds = Dataset()
		ds.file_meta = FileMetaDataset()
		ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
		ds.file_meta.MediaStorageSOPClassUID = sop_class
		ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
		ds.SOPClassUID = sop_class
		ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
		ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
		ds.PatientName, ds.PatientID = 'Display^Test', 'DV-1'
		ds.StudyID, ds.AccessionNumber, ds.StudyDescription, ds.StudyDate, ds.StudyTime = '1', 'ACC1', 'CT PELVIS', '20260101', '120000'
		ds.Modality, ds.SeriesNumber, ds.SeriesDescription = 'CT', 2, 'PELVIS'
		ds.BodyPartExamined, ds.PatientPosition, ds.KVP = 'PELVIS', 'HFS', 120
		if acquisition_number is not None:
			ds.AcquisitionNumber = acquisition_number
		if conversion_type is not None:
			ds.ConversionType = conversion_type
		ds.InstanceNumber = k + 1
		ds.ImagePositionPatient = [0.0, 0.0, k * SPACING[2]]
		ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
		ds.PixelSpacing = [SPACING[1], SPACING[0]]
		ds.SliceThickness = SPACING[2]
		ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
		ds.Rows, ds.Columns = ROWS, COLS
		ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = bits_allocated, bits_stored, bits_stored - 1, int(signed)
		ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType = intercept, slope, 'HU'
		ds.WindowCenter, ds.WindowWidth = 300, 1500
		if lossy:
			ds.LossyImageCompression = '01'
			ds.LossyImageCompressionRatio, ds.LossyImageCompressionMethod = lossy
		ds.PixelData = _pattern(k, stored_max, bits_stored, signed, bits_allocated).tobytes()

		filename = os.path.join(path, '%03d.dcm' % k)
		ds.save_as(filename, enforce_file_format=True)
		files.append(filename)

	return files


def read_series(files):
	'''	SimpleITK volume of the series (rescale applied by GDCM) and the reader with per-slice metadata
	'''
	reader = sitk.ImageSeriesReader()
	reader.MetaDataDictionaryArrayUpdateOn()
	reader.SetFileNames(files)
	return reader.Execute(), reader


def _tagstr(el):
	value = el.value
	if isinstance(value, (list, pydicom.multival.MultiValue)):
		return '\\'.join(str(v) for v in value)
	return str(value)


def make_series(files):
	'''	Mocked ImagingSeries whose first instance headers come from the first file (Orthanc style strings)
		and whose uploads are kept in `series._uploads`
	'''
	ds0 = pydicom.dcmread(files[0])
	tags = {el.keyword: _tagstr(el) for el in ds0 if el.keyword and el.VR not in ('SQ', 'OB', 'OW', 'UN') and not el.tag.is_private}

	sx = mock.create_autospec(ImagingSeries, instance=True)
	sx.pk = 'series-pk'
	sx.series_uid = ds0.SeriesInstanceUID
	sx.description = ds0.SeriesDescription
	sx.modality = 'CT'
	sx.hmeta_key = DicomMetaKey(IMAGING_SERVER_RESOURCE_SERIES, 'SeriesInstanceUID', sx.series_uid)
	sx.hmeta = DicomMeta(sx.description, 'CT', meta=sx.hmeta_key)
	sx.parent.hmeta_key = DicomMetaKey(IMAGING_SERVER_RESOURCE_STUDY, 'StudyInstanceUID', ds0.StudyInstanceUID)
	sx.parent.hmeta = DicomMeta('study', 'CT', meta=sx.parent.hmeta_key)
	sx.model_patient.patientid = ds0.PatientID
	sx.dcm0 = mock.create_autospec(DcmInstance, instance=True)
	sx.dcm0.tags = tags
	sx.dcm0.dcmfile.return_value = ds0

	uploads = []

	def upload_image(stream, **kwargs):
		uploads.append(stream.read())
		return mock.Mock(ok=True)

	sx.pacs = mock.Mock()
	sx.pacs.upload_image.side_effect = upload_image
	sx._uploads = uploads

	instances = []
	for filename in files:
		d = pydicom.dcmread(filename, stop_before_pixels=True)
		inst = mock.create_autospec(DcmInstance, instance=True)
		inst.pk = 'inst-%d' % d.InstanceNumber
		inst.dicomdata = {'SOPInstanceUID': d.SOPInstanceUID, 'ImagePositionPatient': _tagstr(d['ImagePositionPatient'])}
		inst.sop_instance_uid, inst.sop_class_uid, inst.image_position_patient = d.SOPInstanceUID, None, None
		instances.append(inst)
	sx.slices_collection = instances
	sx._ds0 = ds0
	return sx


def uploaded_datasets(series):
	return sorted((pydicom.dcmread(io.BytesIO(b)) for b in series._uploads), key=lambda d: int(d.InstanceNumber))


def _frame(ds):
	return next(generate_frames(ds.PixelData, number_of_frames=1))


def decoded_pixels(ds):
	'''	Pixel array of an uploaded dataset: pydicom, else imagecodecs for the syntaxes pydicom has no decoder
		plugin for in this environment (HTJ2K, baseline JPEG without pylibjpeg)
	'''
	try:
		return ds.pixel_array
	except (RuntimeError, NotImplementedError):
		imagecodecs = pytest.importorskip('imagecodecs')
		if not ds.file_meta.TransferSyntaxUID.is_encapsulated:
			raise
		return imagecodecs.imread(_frame(ds)).reshape(ds.Rows, ds.Columns)


def decoded_intensities(ds):
	return decoded_pixels(ds).astype(np.float64) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)


def codestream_precision(ds):
	'''	(precision, signed) declared by the codestream of the first frame: SOF3 (lossless JPEG), SOF55 (JPEG-LS),
		SOF1 (12 bit JPEG) or the SIZ marker (JPEG 2000 / HTJ2K)
	'''
	frame = _frame(ds)
	if frame.startswith(b'\xff\x4f'):  # SOC: JPEG 2000 / HTJ2K codestream, SIZ follows
		idx = frame.find(b'\xff\x51')
		assert idx >= 0, 'no SIZ marker found'
		ssiz = frame[idx + 2 + 38]
		return (ssiz & 0x7f) + 1, bool(ssiz & 0x80)
	assert frame.startswith(b'\xff\xd8'), 'no SOI marker'
	for marker in (b'\xff\xc3', b'\xff\xf7', b'\xff\xc1', b'\xff\xc0'):
		idx = frame.find(marker)
		if idx >= 0:
			return frame[idx + 4], False
	raise AssertionError('no SOF marker found')


def sitk_read_back(series):
	'''	The uploaded series read as a volume by SimpleITK (rescale applied); numpy array (z, y, x)
	'''
	with tempfile.TemporaryDirectory() as tmp:
		for k, b in enumerate(series._uploads):
			with open(os.path.join(tmp, '%03d.dcm' % k), 'wb') as f:
				f.write(b)
		reader = sitk.ImageSeriesReader()
		reader.SetFileNames(reader.GetGDCMSeriesFileNames(tmp))
		img = reader.Execute()
	return img, sitk.GetArrayFromImage(img)


def series_meta(hcache):
	keys = [k for k in hcache if k.resource == IMAGING_SERVER_RESOURCE_SERIES]
	assert len(keys) == 1
	return keys[0], hcache[keys[0]]


@pytest.fixture(scope='module')
def ct(tmp_path_factory):
	'''	Source series with mixed rescale slopes: files, volume, reader
	'''
	path = tmp_path_factory.mktemp('ct')
	files = write_ct_series(str(path))
	img, reader = read_series(files)
	return files, img, reader


@pytest.fixture
def series(ct):
	return make_series(ct[0])


@pytest.fixture
def ivolume(ct, series):
	return SonadorImagingVolume(series, volume=ct[1], reader=ct[2])


LOSSLESS = [None, 'rle', 'deflate', 'jpeg-ls', 'jpeg2000', 'htj2k', 'jpeg-lossless']
LOSSLESS_SYNTAXES = {None: ExplicitVRLittleEndian, 'rle': RLELossless, 'deflate': DeflatedExplicitVRLittleEndian,
	'jpeg-ls': JPEGLSLossless, 'jpeg2000': JPEG2000Lossless, 'htj2k': HTJ2KLossless, 'jpeg-lossless': JPEGLosslessSV1}


def _skip_without_encoder(compression, plugin=None):
	transfer_syntax = H.display_compression_transfer_syntax(compression)
	backends = H.display_compression_backends(transfer_syntax)
	if not backends:
		pytest.skip('no encoder installed for %s' % transfer_syntax.name)
	if plugin and (plugin if plugin == 'imagecodecs' else 'pydicom:%s' % plugin) not in backends:
		pytest.skip('%s is not installed for %s' % (plugin, transfer_syntax.name))


def test_source_exceeds_first_instance_range(ct):
	'''	The fixture is only meaningful when GDCM applies the rescale of every instance
	'''
	arr = sitk.GetArrayFromImage(ct[1])
	assert arr.max() > 4095 + INTERCEPT, 'the per-instance rescale slopes were not applied by the reader'
	assert arr[0].max() == int(_pattern(0).max()) * 1 + INTERCEPT
	assert arr[1].max() == int(_pattern(1).max()) * 2 + INTERCEPT


@pytest.mark.parametrize('compression', LOSSLESS)
def test_intensities_preserved_with_mixed_rescale(ivolume, series, compression):
	'''	P1: the stored representation covers the resampled intensities; nothing is saturated to the range of
		the first instance, and the decoded intensities equal the display volume for every lossless syntax
	'''
	_skip_without_encoder(compression)
	# 32 x 32 slices: GDCM refuses to decode RLE frames of only a few pixels (6 x 5), which is a reader limit
	hcache = H.dcm_encode_display_volume(ivolume, size=(32, 32, 3), dcm_compression=compression)
	key, meta = series_meta(hcache)
	display = sitk.GetArrayFromImage(meta.attrs['image'])
	assert display.max() > 4095 + INTERCEPT, 'the display volume should exceed the 12 bit range of the first instance'

	datasets = uploaded_datasets(series)
	assert len(datasets) == 3
	for k, ds in enumerate(datasets):
		assert ds.file_meta.TransferSyntaxUID == LOSSLESS_SYNTAXES[compression]
		assert float(ds.RescaleIntercept) == INTERCEPT and float(ds.RescaleSlope) == 1.0 and ds.RescaleType == 'HU'
		assert ds.PixelRepresentation == 0 and ds.BitsStored == 13 and ds.HighBit == 12 and ds.BitsAllocated == 16
		assert ds.LossyImageCompression == '00' and 'LossyImageCompressionMethod' not in ds
		assert np.array_equal(decoded_intensities(ds), display[k]), 'slice %d intensities differ' % k

	assert meta.attrs['representation'].preserved and meta.attrs['representation'].bits_stored == 13
	assert meta.attrs['transfer_syntax'] == (LOSSLESS_SYNTAXES[compression] if compression else None)

	img, back = sitk_read_back(series)
	assert np.array_equal(back, display)
	assert np.allclose(img.GetSpacing(), meta.attrs['image'].GetSpacing()) and np.allclose(img.GetOrigin(), meta.attrs['image'].GetOrigin())


@pytest.mark.parametrize('compression', [None, 'rle', 'jpeg2000'])
def test_instance_uids_consistent(ivolume, series, compression):
	'''	P2: the cache keys, the SOPInstanceUID of the dataset and the MediaStorageSOPInstanceUID agree
	'''
	_skip_without_encoder(compression)
	# 32 x 32 slices: the JPEG 2000 encoders need dimensions compatible with their wavelet decomposition
	hcache = H.dcm_encode_display_volume(ivolume, size=(32, 32, 2), dcm_compression=compression)
	_, meta = series_meta(hcache)
	cached = [k.uid for k in meta.attrs['instances']]

	datasets = uploaded_datasets(series)
	written = [str(ds.SOPInstanceUID) for ds in datasets]
	media = [str(ds.file_meta.MediaStorageSOPInstanceUID) for ds in datasets]
	assert cached == written == media and len(set(cached)) == 2


def test_spacing_selects_grid_without_size(ivolume, monkeypatch):
	'''	P2: a spacing-only call works, the default size applies only without a grid selector, and an
		explicit size together with an explicit spacing is rejected
	'''
	hcache = H.dcm_encode_display_volume(ivolume, spacing=2.0, dry_run=True)
	_, meta = series_meta(hcache)
	extent = np.array((COLS, ROWS, SLICES)) * np.array(SPACING)
	assert meta.attrs['grid'].size == tuple(int(np.ceil(e / 2.0)) for e in extent) == (4, 6, 5)
	assert np.allclose(meta.attrs['grid'].spacing, (2.0, 2.0, 2.0))

	monkeypatch.setattr(H, 'DISPLAY_VOLUME_SIZES', (4, 3))
	hcache = H.dcm_encode_display_volume(ivolume, dry_run=True)
	_, meta = series_meta(hcache)
	assert meta.attrs['grid'].size == (4, 4, 4)

	with pytest.raises(ValueError):
		H.dcm_encode_display_volume(ivolume, size=4, spacing=2.0, dry_run=True)


def test_acquisition_number_present(ct, tmp_path):
	'''	P2: AcquisitionNumber (Type 2) is carried from the source or written empty
	'''
	series = make_series(ct[0])
	H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=ct[1]), size=(4, 4, 2))
	for ds in uploaded_datasets(series):
		assert 'AcquisitionNumber' in ds and ds.AcquisitionNumber == 7 and ds.KVP == 120

	files = write_ct_series(str(tmp_path), slopes=(1, 1, 1), acquisition_number=None)
	img, _ = read_series(files)
	series = make_series(files)
	assert 'AcquisitionNumber' not in series.dcm0.tags
	H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(4, 4, 2), dcm_compression=None)
	for ds in uploaded_datasets(series):
		assert 'AcquisitionNumber' in ds and ds['AcquisitionNumber'].value is None


def test_lossy_history_of_source_is_kept(tmp_path):
	'''	P2: a source that was lossy compressed stays "01" with its ratio and method after a lossless export
	'''
	files = write_ct_series(str(tmp_path), slopes=(1, 1, 1), lossy=(4.5, 'ISO_10918_1'))
	img, _ = read_series(files)
	series = make_series(files)
	H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(4, 4, 2), dcm_compression='rle')
	for ds in uploaded_datasets(series):
		assert ds.LossyImageCompression == '01'
		assert [float(v) for v in H._tag_list(ds.LossyImageCompressionRatio)] == [4.5]
		assert H._tag_list(ds.LossyImageCompressionMethod) == ['ISO_10918_1']


@pytest.mark.parametrize('compression, method, options, tolerance', [
	('jpeg-ls-near', 'ISO_14495_1', {'jls_error': 3}, 3),
	('jpeg2000-lossy', 'ISO_15444_1', {'j2k_psnr': 50}, 400),
])
def test_lossy_encoding_records_history(ivolume, series, compression, method, options, tolerance):
	'''	Lossy syntaxes record LossyImageCompression "01", the method and the per-slice ratio; the decoded
		intensities stay within the codec tolerance
	'''
	_skip_without_encoder(compression)
	# Slices large enough for the codec overhead not to dominate the compression ratio
	hcache = H.dcm_encode_display_volume(ivolume, size=(64, 64, 2), dcm_compression=compression, dcm_compression_options=options)
	_, meta = series_meta(hcache)
	display = sitk.GetArrayFromImage(meta.attrs['image'])
	for k, ds in enumerate(uploaded_datasets(series)):
		assert ds.LossyImageCompression == '01'
		assert H._tag_list(ds.LossyImageCompressionMethod) == [method]
		ratios = [float(v) for v in H._tag_list(ds.LossyImageCompressionRatio)]
		assert len(ratios) == 1 and ratios[0] > 1.0
		assert np.abs(decoded_intensities(ds) - display[k]).max() <= tolerance


def test_jpeg_12bit_requires_fitting_representation(ct, ivolume, tmp_path):
	'''	12 bit JPEG is refused before upload when the stored values need 13 bits, and works for a 12 bit source
	'''
	_skip_without_encoder('jpeg')
	series = ivolume.series
	with pytest.raises(ValueError):
		H.dcm_encode_display_volume(ivolume, size=(8, 8, 2), dcm_compression='jpeg')
	assert series.pacs.upload_image.call_count == 0

	files = write_ct_series(str(tmp_path), slopes=(1, 1, 1))
	img, _ = read_series(files)
	series = make_series(files)
	hcache = H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(8, 8, 2), dcm_compression='jpeg',
		dcm_compression_options={'jpeg_quality': 95})
	_, meta = series_meta(hcache)
	display = sitk.GetArrayFromImage(meta.attrs['image'])
	datasets = uploaded_datasets(series)
	assert all(ds.file_meta.TransferSyntaxUID == JPEGExtended12Bit and ds.LossyImageCompression == '01' for ds in datasets)
	_, back = sitk_read_back(series)
	assert np.abs(back.astype(np.float64) - display).max() < 64


def test_rescale_overflow(ct, series):
	'''	Intensities that no representation holds exactly are rejected ("raise") or quantized with a derived
		rescale ("adapt"); never saturated
	'''
	wide = sitk.Cast(ct[1], sitk.sitkFloat32) * 400.0  # up to ~2.4e6
	ivolume = SonadorImagingVolume(series, volume=wide)
	with pytest.raises(ValueError):
		H.dcm_encode_display_volume(ivolume, size=(4, 4, 2), rescale_overflow='raise')
	assert series.pacs.upload_image.call_count == 0

	hcache = H.dcm_encode_display_volume(ivolume, size=(4, 4, 2), rescale_overflow='adapt', dcm_compression=None)
	_, meta = series_meta(hcache)
	representation = meta.attrs['representation']
	assert not representation.preserved and representation.slope > 1.0 and representation.unsigned and representation.bits_stored == 16
	display = sitk.GetArrayFromImage(meta.attrs['image'])
	for k, ds in enumerate(uploaded_datasets(series)):
		assert np.abs(decoded_intensities(ds) - display[k]).max() <= representation.slope / 2.0 + 1e-6
		assert ds.pixel_array.min() >= 0 and ds.pixel_array.max() <= 65535


def test_preserve_rescale_false_stores_signed_intensities(ivolume, series):
	hcache = H.dcm_encode_display_volume(ivolume, size=(4, 4, 2), preserve_rescale=False, dcm_compression=None)
	_, meta = series_meta(hcache)
	display = sitk.GetArrayFromImage(meta.attrs['image'])
	for k, ds in enumerate(uploaded_datasets(series)):
		assert float(ds.RescaleIntercept) == 0 and float(ds.RescaleSlope) == 1 and ds.PixelRepresentation == 1 and ds.BitsStored == 16
		assert np.array_equal(decoded_intensities(ds), display[k])


def test_derivation_description_records_both_extents(ct, ivolume):
	'''	P2: the description states the extent of the display grid and of the source instead of claiming
		equality; spacing-based grids are larger
	'''
	src_grid = H.image_grid(ct[1])
	grid = H.display_image_grid(ct[1], size=(5, 6, 3))
	text = H.display_volume_derivation_description('1.2.3', src_grid, grid)
	assert 'extent 7 x 10.8 x 9 mm' in text and text.count('extent 7 x 10.8 x 9 mm') == 2
	assert 'same frame of reference' in text and 'physical extent' not in text

	grid = H.display_image_grid(ct[1], spacing=2.0)
	text = H.display_volume_derivation_description('1.2.3', src_grid, grid)
	assert '(2 x 2 x 2 mm, extent 8 x 12 x 10 mm)' in text and 'extent 7 x 10.8 x 9 mm)' in text

	hcache = H.dcm_encode_display_volume(ivolume, spacing=2.0, dry_run=True)
	_, meta = series_meta(hcache)
	assert meta.attrs['grid'].size == (4, 6, 5)


def test_display_volumes_one_series_per_size(ivolume):
	hcache = H.dcm_encode_display_volumes(ivolume, sizes=(4, 3), dry_run=True)
	descriptions = sorted(v.description for k, v in hcache.items() if k.resource == IMAGING_SERVER_RESOURCE_SERIES)
	assert descriptions == ['PELVIS (Display 3)', 'PELVIS (Display 4)']
	for bad in (dict(size=4), dict(spacing=2.0), dict(series_uid='1.2')):
		with pytest.raises(ValueError):
			H.dcm_encode_display_volumes(ivolume, sizes=(4,), dry_run=True, **bad)


def test_compression_registry(ivolume, series):
	assert H.display_compression_transfer_syntax('huffman') == JPEGLosslessSV1
	assert H.display_compression_transfer_syntax(ExplicitVRLittleEndian) is None
	assert H.display_compression_transfer_syntax('RLE') == RLELossless
	assert H.display_compression_transfer_syntax(JPEG2000) == JPEG2000
	assert H.display_compression_transfer_syntax('htj2k') == HTJ2KLossless
	for bad in ('lzw', 'lz4', '3d-dct', 'htj2k-lossy', '1.2.840.10008.1.2.4.100', '1.2.840.10008.1.2.4.203'):
		with pytest.raises(ValueError):
			H.display_compression_transfer_syntax(bad)

	support = H.supported_display_compressions()
	assert support[None] == ('native',) and support[DeflatedExplicitVRLittleEndian] == ('native',)
	assert set(support) == set([None] + list(H.DISPLAY_VOLUME_TRANSFER_SYNTAXES))
	assert support[RLELossless], 'pydicom always provides an RLE encoder'

	# An unsupported compression is rejected before anything is resampled or uploaded
	with pytest.raises(ValueError):
		H.dcm_encode_display_volume(ivolume, size=(4, 4, 2), dcm_compression='lz4')
	with pytest.raises(ValueError):
		H.dcm_encode_display_volume(ivolume, size=(4, 4, 2), dcm_compression='rle', dcm_encoding_plugin='no-such-plugin')
	assert series.pacs.upload_image.call_count == 0


def test_negative_slope_mixed_rescale(tmp_path):
	'''	P2 (round 2): a negative RescaleSlope reverses the stored endpoints; the representation keeps the
		rescale of the source (slope -1, intercept 3000), widens to 13 bit and every slice is uploaded exactly
	'''
	files = write_ct_series(str(tmp_path), slopes=(-1, -2, -2), intercept=3000)
	img, _ = read_series(files)
	arr = sitk.GetArrayFromImage(img)
	assert arr.min() < -2000 and arr.max() > 0, 'GDCM should apply the negative slopes'

	series = make_series(files)
	ivolume = SonadorImagingVolume(series, volume=img)
	representation = H.display_pixel_representation(ivolume, series.dcm0.tags)
	assert representation.preserved and representation.slope == -1.0 and representation.intercept == 3000.0
	assert representation.unsigned and representation.bits_stored == 13

	for compression in (None, 'rle'):
		series = make_series(files)
		hcache = H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(32, 32, 3), dcm_compression=compression)
		_, meta = series_meta(hcache)
		display = sitk.GetArrayFromImage(meta.attrs['image'])
		datasets = uploaded_datasets(series)
		assert len(datasets) == 3
		for k, ds in enumerate(datasets):
			assert float(ds.RescaleSlope) == -1.0 and float(ds.RescaleIntercept) == 3000.0 and ds.BitsStored == 13
			assert np.array_equal(decoded_intensities(ds), display[k])
		_, back = sitk_read_back(series)
		assert np.array_equal(back, display)


def test_jpeg_baseline_writes_8bit_dataset(tmp_path, ivolume):
	'''	P2 (round 2): JPEG baseline stores 8 bit samples: BitsAllocated/BitsStored/HighBit 8/8/7, the saved
		files read back; the syntax is refused for CT Image Storage before anything is resampled
	'''
	_skip_without_encoder('jpeg-baseline')
	files = write_ct_series(str(tmp_path), slopes=(1, 1, 1), intercept=0, stored_max=250, bits_stored=8, bits_allocated=8,
		sop_class=SecondaryCaptureImageStorage)
	img, _ = read_series(files)
	series = make_series(files)
	hcache = H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(64, 64, 2), dcm_compression='jpeg-baseline',
		dcm_compression_options={'jpeg_quality': 95})
	_, meta = series_meta(hcache)
	display = sitk.GetArrayFromImage(meta.attrs['image'])
	datasets = uploaded_datasets(series)
	assert len(datasets) == 2
	for k, ds in enumerate(datasets):
		assert ds.file_meta.TransferSyntaxUID == JPEGBaseline8Bit and ds.SOPClassUID == SecondaryCaptureImageStorage
		assert ds.ConversionType == 'WSD'  # Type 1 of the SC Equipment module, source has none
		assert (ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation) == (8, 8, 7, 0)
		assert codestream_precision(ds) == (8, False)
		assert np.abs(decoded_intensities(ds) - display[k]).max() < 16
	_, back = sitk_read_back(series)
	assert np.abs(back.astype(np.float64) - display).max() < 16

	# CT Image Storage requires BitsAllocated 16: refused before resampling, no upload
	series = make_series(files)
	with pytest.raises(ValueError, match='BitsAllocated'):
		H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(8, 8, 2), dcm_compression='jpeg-baseline',
			sopclass_uid=CTImageStorage)
	assert series.pacs.upload_image.call_count == 0
	with pytest.raises(ValueError):
		H.dcm_encode_display_volume(ivolume, size=(8, 8, 2), dcm_compression='jpeg-baseline')
	assert ivolume.series.pacs.upload_image.call_count == 0


@pytest.mark.parametrize('compression, plugin, signed', [
	('jpeg-ls', 'imagecodecs', False),
	('jpeg-ls', 'pyjpegls', False),
	('jpeg-ls', 'pyjpegls', True),
	('jpeg-ls-near', 'imagecodecs', False),
	('jpeg-lossless', 'imagecodecs', False),
	('jpeg2000', 'imagecodecs', False),
	('jpeg2000', 'imagecodecs', True),
	('jpeg2000-lossy', 'imagecodecs', True),
	('jpeg2000', 'pylibjpeg', True),
	('htj2k', 'imagecodecs', False),
	('htj2k', 'imagecodecs', True),
	('jpeg', 'imagecodecs', False),
])
def test_codestream_precision_matches_headers(tmp_path, compression, plugin, signed):
	'''	P2 (round 2): the sample precision and signedness of the codestream equal BitsStored / PixelRepresentation
		(unsigned 12 bit source kept as 12 bit; signed 12 bit source kept as signed 12 bit)
	'''
	_skip_without_encoder(compression, plugin)
	files = write_ct_series(str(tmp_path), slopes=(1, 1, 1), intercept=0, signed=signed)
	img, _ = read_series(files)
	series = make_series(files)
	hcache = H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(32, 32, 2), dcm_compression=compression,
		dcm_encoding_plugin=plugin)
	_, meta = series_meta(hcache)
	representation = meta.attrs['representation']
	assert representation.preserved and representation.bits_stored == 12 and representation.unsigned == (not signed)
	display = sitk.GetArrayFromImage(meta.attrs['image'])
	lossy = H.display_compression_transfer_syntax(compression) in H.DCM_TRANSFER_SYNTAX_LOSSY_METHODS
	for k, ds in enumerate(uploaded_datasets(series)):
		assert (ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation) == (16, 12, 11, int(signed))
		assert codestream_precision(ds) == (12, signed if compression.startswith(('jpeg2000', 'htj2k')) else False)
		if lossy:
			assert np.abs(decoded_intensities(ds) - display[k]).max() <= 64
		else:
			assert np.array_equal(decoded_intensities(ds), display[k])
	_, back = sitk_read_back(series)
	assert np.abs(back.astype(np.float64) - display).max() <= (64 if lossy else 0)


def test_imagecodecs_rejects_signed_jpegls(tmp_path):
	'''	The imagecodecs JPEG-LS / lossless JPEG encoders take unsigned samples only; the rejection names the alternatives
	'''
	_skip_without_encoder('jpeg-ls', 'imagecodecs')
	files = write_ct_series(str(tmp_path), slopes=(1, 1, 1), intercept=0, signed=True)
	img, _ = read_series(files)
	series = make_series(files)
	with pytest.raises(ValueError, match='unsigned'):
		H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(32, 32, 2), dcm_compression='jpeg-ls',
			dcm_encoding_plugin='imagecodecs')


@pytest.mark.parametrize('plugin', ['pylibjpeg', 'imagecodecs'])
@pytest.mark.parametrize('options', [{'j2k_psnr': 50}, {'j2k_psnr': [50]}, {'j2k_psnr': 50.0}])
def test_j2k_quality_options_accept_scalars_on_every_backend(ivolume, series, plugin, options):
	'''	P2 (round 2): a scalar or a list PSNR works for pylibjpeg (lists required by the encoder) and imagecodecs
	'''
	_skip_without_encoder('jpeg2000-lossy', plugin)
	hcache = H.dcm_encode_display_volume(ivolume, size=(64, 64, 2), dcm_compression='jpeg2000-lossy', dcm_encoding_plugin=plugin,
		dcm_compression_options=options)
	_, meta = series_meta(hcache)
	display = sitk.GetArrayFromImage(meta.attrs['image'])
	for k, ds in enumerate(uploaded_datasets(series)):
		assert ds.file_meta.TransferSyntaxUID == JPEG2000 and ds.LossyImageCompression == '01'
		assert np.abs(decoded_intensities(ds) - display[k]).max() <= 400


def test_j2k_compression_ratio_option(ivolume, series):
	'''	`j2k_cr` is a pylibjpeg option: scalar accepted there, rejected by imagecodecs before any upload
	'''
	backends = H.display_compression_backends(JPEG2000)
	if 'pydicom:pylibjpeg' in backends:
		hcache = H.dcm_encode_display_volume(ivolume, size=(64, 64, 2), dcm_compression='jpeg2000-lossy', dcm_encoding_plugin='pylibjpeg',
			dcm_compression_options={'j2k_cr': 10})
		_, meta = series_meta(hcache)
		for ds in uploaded_datasets(series):
			assert [float(v) for v in H._tag_list(ds.LossyImageCompressionRatio)][0] > 5
		series._uploads.clear()
		series.pacs.upload_image.reset_mock()
	if 'imagecodecs' in backends:
		with pytest.raises(ValueError, match='j2k_cr'):
			H.dcm_encode_display_volume(ivolume, size=(64, 64, 2), dcm_compression='jpeg2000-lossy', dcm_encoding_plugin='imagecodecs',
				dcm_compression_options={'j2k_cr': 10})
		assert series.pacs.upload_image.call_count == 0


def test_rescale_overflow_raise_rejects_unpreservable_source_rescale(ct, tmp_path):
	'''	P3 (round 2): "raise" rejects as soon as the rescale of the source cannot be kept, although the intensities
		would fit the signed 0/1 fallback that "adapt" then uses exactly
	'''
	files = write_ct_series(str(tmp_path), slopes=(0.1, 0.1, 0.1), intercept=10000)  # intensities 10000 + 0.1 * stored
	img, _ = read_series(files)
	img = sitk.Cast(img, sitk.sitkFloat32) - 11000.0  # ~ -1000 .. -650 HU, stored values with the source rescale: -110000 ..
	series = make_series(files)
	with pytest.raises(ValueError, match='preserve'):
		H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(4, 4, 2), rescale_overflow='raise', dcm_compression=None)
	assert series.pacs.upload_image.call_count == 0

	hcache = H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(4, 4, 2), rescale_overflow='adapt', dcm_compression=None)
	_, meta = series_meta(hcache)
	representation = meta.attrs['representation']
	assert not representation.preserved and representation.intercept == 0 and representation.slope == 1 and not representation.unsigned
	display = sitk.GetArrayFromImage(meta.attrs['image'])
	for k, ds in enumerate(uploaded_datasets(series)):
		assert np.array_equal(decoded_intensities(ds), np.rint(display[k]))


def _subdir(path, name):
	sub = path / name
	sub.mkdir()
	return str(sub)


@pytest.mark.parametrize('compression', [None, 'rle'])
def test_secondary_capture_conversion_type(tmp_path, compression):
	'''	P2 (round 3): Secondary Capture Image Storage requires ConversionType (PS3.3 Table C.8-24): written as
		"WSD" when the source has none (SC source, or CT source with an explicit sopclass_uid), the value of the
		source is kept when it has one, and `series_headers` override both; CT output does not receive it
	'''
	# SC source without ConversionType (an incomplete source), SC output
	files = write_ct_series(_subdir(tmp_path, 'sc'), slopes=(1, 1, 1), sop_class=SecondaryCaptureImageStorage)
	img, _ = read_series(files)
	series = make_series(files)
	H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(8, 8, 2), dcm_compression=compression)
	datasets = uploaded_datasets(series)
	assert len(datasets) == 2 and all(ds.SOPClassUID == SecondaryCaptureImageStorage and ds.ConversionType == 'WSD' for ds in datasets)
	_, back = sitk_read_back(series)
	assert back.shape == (2, 8, 8)

	# SC source with ConversionType: the value of the source is kept
	files = write_ct_series(_subdir(tmp_path, 'sc-si'), slopes=(1, 1, 1), sop_class=SecondaryCaptureImageStorage, conversion_type='SI')
	img, _ = read_series(files)
	series = make_series(files)
	H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(8, 8, 2), dcm_compression=compression)
	assert all(ds.ConversionType == 'SI' for ds in uploaded_datasets(series))

	# CT source, explicit SC SOP class: default written; series_headers override it
	files = write_ct_series(_subdir(tmp_path, 'ct'), slopes=(1, 1, 1))
	img, _ = read_series(files)
	series = make_series(files)
	H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(8, 8, 2), dcm_compression=compression,
		sopclass_uid=SecondaryCaptureImageStorage)
	assert all(ds.SOPClassUID == SecondaryCaptureImageStorage and ds.ConversionType == 'WSD' for ds in uploaded_datasets(series))
	series = make_series(files)
	H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(8, 8, 2), dcm_compression=compression,
		sopclass_uid=SecondaryCaptureImageStorage, series_headers={'ConversionType': 'SYN'})
	assert all(ds.ConversionType == 'SYN' for ds in uploaded_datasets(series))

	# CT output: no ConversionType
	series = make_series(files)
	H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(8, 8, 2), dcm_compression=compression)
	assert all(ds.SOPClassUID == CTImageStorage and 'ConversionType' not in ds for ds in uploaded_datasets(series))


@pytest.mark.parametrize('compression', [None, 'deflate', 'rle', 'jpeg-lossless', 'jpeg-ls', 'jpeg2000', 'htj2k'])
def test_8bit_source_stored_as_8bit_samples(tmp_path, compression):
	'''	Round 4: an 8 bit Secondary Capture source is stored as 8 bit samples (8/8/7) for every lossless syntax,
		so that the pixel attributes, the encoded frame, the pydicom decode and the SimpleITK read-back agree
		(RLE wrote a one segment frame under 16 bit attributes before)
	'''
	if compression:
		_skip_without_encoder(compression)
	files = write_ct_series(str(tmp_path), slopes=(1, 1, 1), intercept=0, stored_max=250, bits_stored=8, bits_allocated=8,
		sop_class=SecondaryCaptureImageStorage)
	img, _ = read_series(files)
	series = make_series(files)
	hcache = H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(32, 32, 2), dcm_compression=compression)
	_, meta = series_meta(hcache)
	display = sitk.GetArrayFromImage(meta.attrs['image'])
	datasets = uploaded_datasets(series)
	assert len(datasets) == 2
	for k, ds in enumerate(datasets):
		assert (ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation) == (8, 8, 7, 0)
		assert ds.SOPClassUID == SecondaryCaptureImageStorage and ds.ConversionType == 'WSD'
		if ds.file_meta.TransferSyntaxUID.is_encapsulated and compression != 'rle':
			assert codestream_precision(ds) == (8, False)
		assert np.array_equal(decoded_intensities(ds), display[k])
	_, back = sitk_read_back(series)
	assert np.array_equal(back, display)


@pytest.mark.parametrize('compression', [None, 'rle'])
@pytest.mark.parametrize('signed', [False, True])
def test_narrow_source_widened_to_8_bits(tmp_path, compression, signed):
	'''	Round 5: a source of fewer than 8 stored bits (7 bit Secondary Capture in a 16 bit container) is stored as
		8 bit samples (8/8/7) with the rescale of the source: GDCM decodes an 8 bit container only when BitsStored
		equals BitsAllocated, so 8/7/6 output could not be read back with SimpleITK
	'''
	files = write_ct_series(str(tmp_path), slopes=(1, 1, 1), intercept=0, stored_max=60 if signed else 120, bits_stored=7,
		bits_allocated=16, signed=signed, sop_class=SecondaryCaptureImageStorage)
	img, _ = read_series(files)
	series = make_series(files)
	hcache = H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(32, 32, 2), dcm_compression=compression)
	_, meta = series_meta(hcache)
	display = sitk.GetArrayFromImage(meta.attrs['image'])
	datasets = uploaded_datasets(series)
	assert len(datasets) == 2
	for k, ds in enumerate(datasets):
		assert (ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation) == (8, 8, 7, int(signed))
		assert (float(ds.RescaleIntercept), float(ds.RescaleSlope)) == (0.0, 1.0)
		assert ds.SOPClassUID == SecondaryCaptureImageStorage and ds.ConversionType == 'WSD'
		assert np.array_equal(decoded_intensities(ds), display[k])
	_, back = sitk_read_back(series)
	assert np.array_equal(back, display)


def test_8bit_ct_source_widened_to_12_bits(tmp_path):
	'''	Round 4: CT Image Storage requires BitsAllocated 16 and BitsStored 12 to 16: an 8 bit CT source is stored
		as 12 bit samples in a 16 bit container (16/12/11), exactly
	'''
	files = write_ct_series(str(tmp_path), slopes=(1, 1, 1), intercept=0, stored_max=250, bits_stored=8, bits_allocated=8)
	img, _ = read_series(files)
	series = make_series(files)
	hcache = H.dcm_encode_display_volume(SonadorImagingVolume(series, volume=img), size=(32, 32, 2))
	_, meta = series_meta(hcache)
	display = sitk.GetArrayFromImage(meta.attrs['image'])
	datasets = uploaded_datasets(series)
	assert len(datasets) == 2
	for k, ds in enumerate(datasets):
		assert ds.SOPClassUID == CTImageStorage and 'ConversionType' not in ds
		assert (ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation) == (16, 12, 11, 0)
		assert np.array_equal(decoded_intensities(ds), display[k])
	_, back = sitk_read_back(series)
	assert np.array_equal(back, display)
