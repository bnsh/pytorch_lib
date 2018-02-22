#! /usr/bin/python

"""Just a likely soon to be useless function that times how long things take."""

import sys
import time
from collections import Counter
from contextlib import contextmanager

class Timer(object):
	__timing = Counter()
	__inprogress = {}

	@staticmethod
	def reset():
		Timer.__timing = Counter()
		Timer.__inprogress = {}

	@staticmethod
	def tic(name):
		Timer.__inprogress[name] = time.time()

	@staticmethod
	def toc(name):
		elapsed = time.time() - Timer.__inprogress[name]
		del Timer.__inprogress[name]
		Timer.__timing[name] += elapsed

	@staticmethod
	@contextmanager
	def tictoc(name):
		Timer.tic(name)
		yield
		Timer.toc(name)

	@staticmethod
	def report():
		for name, elapsed in sorted(Timer.get_timing().iteritems(), key=lambda x: -x[1]):
			sys.stderr.write("	%s: %.7f\n" % (name, elapsed))

	@staticmethod
	def get_timing():
		return Timer.__timing

def main():
	pass

if __name__ == "__main__":
	main()
