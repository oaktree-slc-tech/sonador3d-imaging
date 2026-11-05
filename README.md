# Sonador 3D: Imaging
This library is part of [Sonador 3D](https://code.oak-tree.tech/oak-tree/medical-imaging/sonador3d) and provides base classes and the core interfaces needed for working with "imaging volumes" (3D representations of pixel data) and interfacing with [SimpleITK](https://simpleitk.org/).

Please refer to the [Sonador 3D repository](https://code.oak-tree.tech/oak-tree/medical-imaging/sonador3d) for additional information and installation instructions.


## Display volumes
`sonador3d_imaging.helpers` can create "display" copies of an imaging volume: the volume resampled to a coarser grid so that viewers and other downstream tools can load the series faster than the acquisition. A size-based grid is a cube of equal voxel counts (anisotropic spacing) covering exactly the extent of the source; a spacing-based grid (isotropic or not) covers the extent rounded up to whole voxels. `dcm_encode_display_volume` encodes the copy as a derived DICOM series of the same SOP class, study and frame of reference as the source (new `SeriesInstanceUID`, per-slice `ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`, `SliceThickness`, `ImageType` `DERIVED\SECONDARY\AXIAL\RESAMPLED`, `DerivationDescription` recording both extents, `SourceImageSequence` and `ReferencedSeriesSequence` for provenance), uploads it to the imaging server of the series and returns a header cache (`hcache`) of the study and series created.

The copy keeps the intensity units of the source (HU for a CT); the samples are filtered and interpolated, so individual values differ from the acquisition. The stored representation always covers the intensity range of the copy (`display_pixel_representation`), trying in order:

1. the `RescaleIntercept`/`RescaleSlope` of the source (negative slopes included) with `BitsStored`/`PixelRepresentation` widened as needed: a 12 bit source whose resampled values need 13 bits is stored as 13 bit;
2. signed 16 bit intensities with rescale 0/1 (also when `preserve_rescale=False`);
3. a derived rescale (`RescaleIntercept` at the minimum, `RescaleSlope` covering the range), which quantizes and is logged as a warning.

Values are never saturated. `rescale_overflow="adapt"` (default) falls through the steps; `rescale_overflow="raise"` raises a `ValueError` at the first step whose stored values overflow, that is as soon as the rescale of the source cannot be kept (before the signed 0/1 step is considered), before anything is uploaded. `BitsAllocated` follows `BitsStored`: 8 bit samples when at most 8 bits are stored, 16 bit otherwise; CT Image Storage requires `BitsAllocated` 16 and `BitsStored` 12 to 16, so a narrower representation is widened to 12 bits for it; a representation of fewer than 8 bits is widened to 8, since GDCM decodes an 8 bit container only when `BitsStored` equals `BitsAllocated` (`stored_pixel_container`).

```python
from sonador3d_imaging.volume import SonadorImagingVolume
from sonador3d_imaging.helpers import dcm_encode_display_volume, dcm_encode_display_volumes

ivolume = SonadorImagingVolume(sx)                   # sx: the source ImagingSeries (e.g. the primary CT)

# Either one series at a time ...
hcache = dcm_encode_display_volume(ivolume, size=512)              # 512 x 512 x 512 voxels, SeriesNumber 1512
hcache = dcm_encode_display_volume(ivolume, spacing=2.0, hcache=hcache)  # isotropic 2 mm grid

# ... or one series per size in a single call (do not combine with the calls above for the same sizes:
# every call creates a new series)
hcache = dcm_encode_display_volumes(ivolume, sizes=(512, 256, 171))
```

`resample_display_image(img, size=..., spacing=...)` returns the resampled `SimpleITK.Image` without encoding it; `display_image_grid` computes the grid alone. The encoder accepts `dry_run=True` (encode without uploading), `series_headers`/`instance_headers` overrides, `copy_private_creators` (private blocks copied from the source, identified by their private creator string) and a `presave_callback(k, dcmfile, display)`.

### Compression
`dcm_compression` selects the transfer syntax of the pixel data: a pydicom UID or one of the aliases below (`display_compression_transfer_syntax`). RLE Lossless is the default. The encoder installed for the syntax is used (`display_compression_backends`: pydicom encoding plugins first, then [imagecodecs](https://pypi.org/project/imagecodecs/)); `dcm_encoding_plugin` forces one (`"gdcm"`, `"pydicom"`, `"pyjpegls"`, `"pylibjpeg"` or `"imagecodecs"`). `supported_display_compressions()` lists the encoders installed for each syntax; it does not know the stored representation, whose restrictions are listed in the table (the imagecodecs JPEG-LS and lossless JPEG encoders take unsigned samples only, so they reject the signed output of `preserve_rescale=False` or of a source without a rescale). Lossy codecs take their parameters from `dcm_compression_options` (`jls_error`, `j2k_psnr` or `j2k_cr`, `jpeg_quality`; numbers or, for the JPEG 2000 quality layers, lists) and record `LossyImageCompression` `01` with the method and per-slice ratio; a source that was lossy compressed before stays `01` and keeps its history. The sample precision of every codestream equals `BitsStored`.

| `dcm_compression` | Transfer syntax | Lossless | Encoder | Stored values | Cornerstone3D | SimpleITK (GDCM) |
|---|---|---|---|---|---|---|
| `None`, `"none"` | Explicit VR Little Endian | yes | native | any | yes | yes |
| `"deflate"`, `"zlib"` | Deflated Explicit VR Little Endian (whole dataset) | yes | native | any | yes | yes |
| `"rle"` (default) | RLE Lossless | yes | pydicom, gdcm, pylibjpeg-rle | any | yes | yes |
| `"jpeg-lossless"`, `"huffman"` | JPEG Lossless SV1 (process 14, predictor 1; Huffman coded) | yes | imagecodecs | unsigned | yes | yes |
| `"jpeg-ls"` | JPEG-LS Lossless | yes | pyjpegls (signed or unsigned), imagecodecs (unsigned) | see encoder | yes | yes |
| `"jpeg-ls-near"` | JPEG-LS Near-Lossless (`jls_error`, default 2) | bounded error | pyjpegls (signed or unsigned), imagecodecs (unsigned) | see encoder | yes | yes |
| `"jpeg2000"`, `"j2k"` | JPEG 2000 Lossless (reversible wavelet) | yes | pylibjpeg-openjpeg, imagecodecs | any | yes | yes |
| `"jpeg2000-lossy"` | JPEG 2000 (`j2k_psnr`, default 60 dB; `j2k_cr` with pylibjpeg) | no | pylibjpeg-openjpeg, imagecodecs | any | yes | yes |
| `"htj2k"` | High-Throughput JPEG 2000 Lossless (`1.2.840.10008.1.2.4.201`) | yes | imagecodecs (OpenJPH) | any | yes | yes (GDCM with OpenJPEG >= 2.5) |
| `"jpeg"`, `"dct"` | JPEG Extended 12 bit (`jpeg_quality`, default 90) | no | imagecodecs | unsigned, at most 12 bits; stored as 12 | yes | yes |
| `"jpeg-baseline"` | JPEG Baseline 8 bit (`jpeg_quality`) | no | imagecodecs | unsigned, at most 8 bits; stored as 8 bit samples (`BitsAllocated` 8), which CT Image Storage does not allow: pass a `sopclass_uid` that does (e.g. Secondary Capture Image Storage, whose required `ConversionType` is written as `WSD` unless the source or `series_headers` provide one) | yes | yes |

Not offered by these helpers: LZW, LZ4 and standalone Huffman/entropy coders, which have no DICOM transfer syntax (Huffman coding is the entropy stage of lossless JPEG, LZ77/Huffman that of deflate); compression across slices: every slice is encoded as its own single-frame instance, so the JPEG 2000 Part 2 multi-component transfer syntaxes (PS3.5 A.4.4, `1.2.840.10008.1.2.4.92` / `.93`, which can apply a wavelet transform across the frames of a multi-frame object) are not used; and the lossy HTJ2K syntaxes, whose encoder parameters (OpenJPH quantization steps) do not map onto the PSNR / ratio options used here.
