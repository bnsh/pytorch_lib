PYTHON=$(wildcard *.py)
PYLINT=$(addprefix ., $(PYTHON:py=pylint))

all: pylint

pylint: $(PYLINT)

.%.pylint: %.py
	python3 -m pylint -r n $(^)
	/usr/bin/touch $(@)
