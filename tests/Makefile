PYTHON=$(wildcard *.py)
PYLINT=$(addprefix ., $(PYTHON:py=pylint))

all: pylint

pylint: $(PYLINT)

.%.pylint: %.py
	/usr/local/bin/pylint -r n $(^)
	> $(@)
