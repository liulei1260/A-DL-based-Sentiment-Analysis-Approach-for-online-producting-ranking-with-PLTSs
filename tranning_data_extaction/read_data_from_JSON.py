import json
import numpy as np
import pandas as pd
from types import SimpleNamespace
import multiprocessing as mp

def howmany_within_range2(i, point):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    pp=json.loads(point)
    if 'TV' in pp['title']:
      return (i, pp['title'],pp['asin'],''.join(pp['category']))
    else:
      return None


def collect_result(result):
    global results
    if (result is not None):
      results.append(result)

with open('meta_Electronics.json',"r") as f:
    x = f.readlines()

results = []

pool = mp.Pool(4)


for i, point in enumerate(x):
    pool.apply_async(howmany_within_range2, args=(i, point), callback=collect_result)

pool.close()
pool.join()

results.sort(key=lambda x: x[0])
TVdata=pd.DataFrame(results,columns=['i','title','asin','category'])
TVdata.drop(['i'], axis=1)
TVdata.to_csv('TV_info_p.csv')
print(str(np.shape(results)))
print(results[0])