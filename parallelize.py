from multiprocessing import Pool, cpu_count

CPUS = cpu_count()

def parallelize(methods, args):
	results = []
	if CPUS > 1:
		for i in xrange(0, len(methods), CPUS):
			pool = Pool()
			for j in xrange(CPUS):
				if i + j >= len(methods):
					break
				results.append(pool.apply_async(methods[i + j], args = args[i + j]))
			pool.close()
			pool.join()
		map(lambda x: x.get(), results)
	else:
		for i in xrange(len(methods)):
			results.append(methods[i](*args[i]))
	return results


if __name__ == "__main__":
	import numpy as np

	def multiply(x, y):
		return x * y

	x = np.ones((3000, 3000))
	y = np.ones((3000, 3000))

	parallelize([multiply, multiply, multiply, multiply, multiply], [(x, y), (x, y), (x, y), (x, y), (x, y)])
