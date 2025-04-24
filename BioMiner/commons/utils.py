import json
import pandas as pd
from openai import OpenAI
from joblib import Parallel, delayed, cpu_count
from easydict import EasyDict
import yaml
from tqdm import tqdm
import os, pickle
from typing import *
import lmdb
import os.path as osp

def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
  if n_jobs is None:
    n_jobs = cpu_count() - 1

  results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
  )

  return results


def get_config_easydict(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)


def to_lmdb(
        datas: List[Any],
        output_file: str,
) -> None:
    os.makedirs(osp.dirname(output_file), exist_ok = True)

    env = lmdb.open(
        output_file,
        map_size=int(1e12),
        create=True,
        subdir=False,
        readonly=False,
    )
    txn = env.begin(write=True)
    for idx, d in tqdm(enumerate(datas), desc='saving to lmdb '):
        txn.put(str(idx).encode('ascii'), pickle.dumps(d))
        if idx % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()

    return