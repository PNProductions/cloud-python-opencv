from multiprocessing import Pool, cpu_count

CPUS = cpu_count()

def parallelize(methods, args):
  results = []
  if CPUS > 1 and len(methods) > 1:
    pool = Pool(CPUS)
    for method, arg in zip(methods, args):
      results.append(pool.apply_async(method, arg))
    pool.close()
    pool.join()
    out = map(lambda x: x.get(), results)
  else:
    for method, arg in zip(methods, args):
      results.append(method(*arg))
  return results


if __name__ == "__main__":
  import numpy as np

  def multiply(x, y):
    return x * y

  x = np.ones((3000, 3000))
  y = np.ones((3000, 3000))

  parallelize([multiply, multiply, multiply, multiply, multiply], [(x, y), (x, y), (x, y), (x, y), (x, y)])
