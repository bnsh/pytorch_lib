CC=g++
CFLAGS=-pthread -DNDEBUG -g -fwrapv -O2 -Wall -Werror -fno-strict-aliasing -Wdate-time -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security -fPIC -DWITH_CUDA -I/usr/local/lib/python2.7/dist-packages/torch/utils/ffi/../../lib/include -I/usr/local/lib/python2.7/dist-packages/torch/utils/ffi/../../lib/include/TH -I/usr/local/lib/python2.7/dist-packages/torch/utils/ffi/../../lib/include/THC -I/usr/local/cuda/include -I/usr/include/python2.7 -std=c++11

PYTHON=$(wildcard *.py */*.py)
PYLINT3=$(filter-out imageutil_cext/.%.pylint3, $(join $(dir $(PYTHON)), $(addprefix ., $(notdir $(PYTHON:py=pylint3)))))

SRCS=\

OBJS=$(SRCS:C=o)

BINS=\

all: pylint $(BINS)

checkin:
	/usr/bin/ci -l -m- -t- $(PYTHON) $(SRCS)

clean:
	/bin/rm -fr $(OBJS) imageutil_cext *.pyc $(PYLINT)

push: all checkin
	/usr/bin/rsync -avz -e ssh --progress /home/binesh/src/pytorchlib/ gpu:src/pytorchlib/
	/usr/bin/rsync -avz -e ssh --progress /home/binesh/src/pytorchlib/ dgx:src/pytorchlib/
	/usr/bin/rsync -avz -e ssh --progress /home/binesh/src/pytorchlib/ t450s:src/pytorchlib/
	/usr/bin/rsync -avz -e ssh --progress /home/binesh/src/pytorchlib/ p1gen2:src/pytorchlib/

pylint: $(PYLINT3)

ImageUtil.py: $(BINS)

%.o: %.C
	$(CC) -c $(CFLAGS) $(^) -o $(@)

.%.pylint3: %.py
	python3 -m pylint -r n $(^)
	/usr/bin/touch $(@)
