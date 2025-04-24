import base64
import requests
import numpy as np
from loguru import logger
from joblib import Parallel, delayed
import os
import argparse


def to_b64(file_path):
    try:
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        raise Exception(f'File: {file_path} - Info: {e}')


def do_parse(file_path, url='http://127.0.0.1:8000/predict', **kwargs):
    try:
        response = requests.post(url, json={
            'file': to_b64(file_path),
            'file_name': os.path.basename(file_path).split('.')[0],
            'kwargs': kwargs
        })

        if response.status_code == 200:
            output = response.json()
            output['file_path'] = file_path
            return output
        else:
            raise Exception(response.text)
    except Exception as e:
        logger.error(f'File: {file_path} - Info: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_dir', type=str, default='./input_pdf')
    args = parser.parse_args()

    files = os.listdir(args.pdf_dir)
    files = [os.path.join(args.pdf_dir, f) for f in files]
    n_jobs = np.clip(len(files), 1, 8)
    results = Parallel(n_jobs, prefer='threads', verbose=10)(
        delayed(do_parse)(p) for p in files
    )
