#! /usr/bin/python

"""Just a likely soon to be useless function that times how long things take."""

import time
from collections import Counter

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
	def get_timing():
		return Timer.__timing

def main():
	pass

if __name__ == "__main__":
	main()
