"""Inspect schema/lineage without materializing a Hugging Face datasets cache."""
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from pathlib import Path

if __name__ == '__main__':
    path = hf_hub_download('mteb/SUGARCREPE_fmt', 'data/train-00000-of-00008.parquet', repo_type='dataset',
                          revision='e9e0e3ce40c94023c819d3413edb5d090ec9f0d8')
    row = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
    def describe(value):
        if isinstance(value, bytes):
            return f'<{len(value)} bytes>'
        if isinstance(value, dict):
            return {k: describe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [describe(v) for v in value[:2]]
        return value
    print(describe(row), flush=True)
