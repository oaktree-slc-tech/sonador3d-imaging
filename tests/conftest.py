'''	pytest configuration of the sonador3d_imaging tests: when the package is not installed, import it (and the
	Sonador client it depends on) from the directory that holds the working copies.
'''
import os, sys

try:
	import sonador3d_imaging  # noqa: F401
except ImportError:
	sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
