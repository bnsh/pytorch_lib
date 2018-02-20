CC=g++
CFLAGS=-pthread -DNDEBUG -g -fwrapv -O2 -Wall -Werror -fno-strict-aliasing -Wdate-time -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security -fPIC -DWITH_CUDA -I/usr/local/lib/python2.7/dist-packages/torch/utils/ffi/../../lib/include -I/usr/local/lib/python2.7/dist-packages/torch/utils/ffi/../../lib/include/TH -I/usr/local/lib/python2.7/dist-packages/torch/utils/ffi/../../lib/include/THC -I/usr/local/cuda/include -I/usr/include/python2.7

PYTHON=$(wildcard *.py */*.py)
PYLINT=$(filter-out ImageUtil_cext/.%.pylint, $(join $(dir $(PYTHON)), $(addprefix ., $(notdir $(PYTHON:py=pylint)))))

SRCS=\
	ImageUtil.C \
	ImageUtil_impl.C \

OBJS=$(SRCS:C=o)

BINS=\
	ImageUtil_cext/_ImageUtil_cext.so

all: pylint $(BINS)

clean:
	/bin/rm -fr $(OBJS) ImageUtil_cext *.pyc $(PYLINT)

pylint: $(PYLINT)

ImageUtil.py: $(BINS)

$(BINS): $(OBJS) image_util_build.py
	ls -l $(OBJS)
	/usr/bin/python image_util_build.py

%.o: %.C
	$(CC) -c $(CFLAGS) $(^) -o $(@)

.%.pylint: %.py
	/usr/local/bin/pylint -r n $(^)
	/usr/bin/touch $(@)
