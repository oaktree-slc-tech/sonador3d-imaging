import logging
from collections import OrderedDict, namedtuple

import numpy as np
import SimpleITK as sitk

from client.utils.general import first

from sonador.imaging.orthanc.base import ImagingSeries
from sonador.imaging.orthanc.seg import ImageSegmentData

from .apisettings import IMAGING_VOLUME_TYPE_NUMPY, IMAGING_VOLUME_TYPE_ITK

logger = logging.getLogger(__name__)


DcmSliceZCoordinateElements = namedtuple('DcmSliceZCoordinateElements', 
    ('spacing', 'slice_z', 'slice_thickness', 'slice'))


def dcmslice_zcoord_elements(s):
    ''' Retrieve slice z position and thickness from the provided DICOM slice instance
    '''
    spacing = s.pixel_spacing
    
    # Z location of slice
    slice_z = s.slice_location if s.slice_location \
        else s.image_position_patient.z if s.image_position_patient \
        else None

    # Thickness of the slice: use "thickness" if available otherwise use
    # twice the width of the scan
    slice_thickness = spacing.thickness if spacing.thickness is not None \
        else 2*spacing.x if spacing else None

    return DcmSliceZCoordinateElements(spacing, slice_z, slice_thickness, s)


def dcmslice_match_zpos(zpos, dcmslice_zpos):
    ''' Determine if the provided z-position matches the DICOM slice z-position.
        A z-position is considered a match if it:

        1. z-position and DICOM z-position match exactly
        2. z-position falls within the segmentation z-position +/- half the thickness

        @returns bool: True if matches, False otherwise
    '''
    # Unpack the DICOM slice data
    spacing, slice_z, slice_thickness, s = dcmslice_zpos

    return spacing and slice_z is not None and zpos is not None \
        and slice_thickness is not None and (
            slice_z == zpos or zpos - slice_thickness/2 <= slice_z <= zpos + slice_thickness/2)


def dcmseg2labelmap(ivolume, dcmseg: ImageSegmentData, dtype=None, voltype=IMAGING_VOLUME_TYPE_NUMPY, **kwargs):
    ''' Package a DICOM segmentation file as a labelmap array.
        
        @input dcmseries (SonadorImagingVolume): medical imaging series for which the labelmap should be created
        @input dcmseg (ImageSegmentData): segment from which the labelmap should be created
        @input dtype (numpy.ndarray data type): data type which should be used for the segmentation arrays
        
        @returns numpy.ndarray labelmap matching the dimensions (slices, rows, columns) of the input DICOM series 
    '''
    dtype = dtype or dcmseg.data.frames.dtype

    # Load mask to SimpleITK: DICOMseg mask volumes may be partial volumes as they do not contain
    # data for slices without labeled pixels.
    frames_img = sitk.GetImageFromArray(dcmseg.data.frames)
    frames_img.SetSpacing(ivolume.volume.GetSpacing())
    frames_img.SetDirection(ivolume.volume.GetDirection())
    frames_img.SetOrigin([
        float(p) for p in dcmseg.data.frame_descriptions[0].PlanePositionSequence[0].ImagePositionPatient])

    # Resample to volume
    frames_volume = sitk.Resample(frames_img, ivolume.volume)

    if voltype == IMAGING_VOLUME_TYPE_NUMPY:
        return sitk.GetArrayFromImage(frames_volume)

    elif voltype == IMAGING_VOLUME_TYPE_ITK:
        return frames_volume

    raise TypeError('Invalid volume type: %s' % voltype)
