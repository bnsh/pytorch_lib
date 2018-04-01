#! /usr/bin/python

"""This builds the C extension for the ImageUtil class"""

import os
from torch.utils.ffi import create_extension

def main():
	this_dir = os.getcwd()
	sources = []
	extra_objects = [os.path.join(this_dir, x) for x in ["ImageUtil.o", "ImageUtil_impl.o"]]
	library_dirs = ["/usr/local/cuda/lib64"]
	libraries = ["nppi", "npps"]
	libraries = [ \
		"nppidei_static", \
		"nppc_static", \
		"culibos", \
		"nppig_static", \
		"nppicc_static", \
		"nppist_static", \
		"npps_static", \
		"nppif_static", \
	]
	headers = ["ImageUtil.H"]
	defines = [("WITH_CUDA", None)]
	with_cuda = True

	ffi = create_extension( \
		"imageutil_cext", \
		headers=headers, \
		sources=sources, \
		extra_objects=extra_objects, \
		library_dirs=library_dirs, \
		libraries=libraries, \
		define_macros=defines, \
		relative_to=__file__, \
		with_cuda=with_cuda \
	)
	ffi.build()

if __name__ == "__main__":
	main()
