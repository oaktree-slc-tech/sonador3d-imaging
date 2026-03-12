'''	Utilities for computing quantitative evaluation metrics on medical imaging data and derived segmentation masks. 
	The sonador3d_imaging.metrics module provides helper functions for assessing geometric agreement between 
	reference annotations and automated outputs, including commonly used measures such as Dice similarity and 
	Hausdorff distance. These tools operate on SimpleITK.Image objects or compatible array representations and 
	are intended to support validation, quality assessment, and analysis of segmentation results within 
	imaging workflows and research notebooks.
'''
import SimpleITK as sitk
import numpy as np

from dataclasses import dataclass


@dataclass
class HausdorffDistanceResults:
	'''	Convenience classs for working with the results of Hausdorff distance results
	'''
	img_filter: sitk.HausdorffDistanceImageFilter

	@property
	def average(self):
		return self.img_filter.GetAverageHausdorffDistance()

	@property
	def distance(self):
		return self.img_filter.GetHausdorffDistance()


@dataclass
class LabelmapDiceResults:
	'''	Convenience class for working DICE coefficient results
	'''
	img_filter: sitk.LabelOverlapMeasuresImageFilter

	@property
	def dice(self):
		return self.img_filter.GetDiceCoefficient(1)


def labelmap_dice_score(img: sitk.Image, img_ref: sitk.Image):
	'''	Calculate the DICE score between two SimpleITK images
	'''
	_filter = sitk.LabelOverlapMeasuresImageFilter()
	_filter.Execute(img, img_ref)

	return LabelmapDiceResults(img_filter=_filter)


def hausdorff_distance(img: sitk.Image, img_ref: sitk.Image):
	'''	Calculates the Hausdorff distance between the provided image and reference image.
	'''
	_hfilter = sitk.HausdorffDistanceImageFilter()
	_hfilter.Execute(img, img_ref)

	return HausdorffDistanceResults(img_filter=_hfilter)