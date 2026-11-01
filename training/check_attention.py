"""Verify bidirectional attention for the pinned PolDense despite generic ST warnings."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'training/runs/e1-full')
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    lock = read_json(args.run_dir / 'model_lock.json')['text']
    tokenizer = AutoTokenizer.from_pretrained(lock['model'], revision=lock['revision'], local_files_only=True)
    model = AutoModel.from_pretrained(lock['model'], revision=lock['revision'], local_files_only=True,
                                      attn_implementation='sdpa').eval().requires_grad_(False)
    inputs = tokenizer(['[query]: To jest czerwony samochód.', '[query]: To jest zielony samochód.'],
                       padding=True, return_tensors='pt')
    assert inputs['input_ids'][0, 0] == inputs['input_ids'][1, 0]
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state[:, 0]
        difference = float((hidden[0] - hidden[1]).norm())
    if not difference > 1e-5:
        raise ValueError('First-token representation did not respond to changed future tokens')
    result = dict(model=lock, transformers=transformers.__version__, architecture=type(model).__name__,
                  attention='sdpa', config_is_causal=getattr(model.config, 'is_causal', None),
                  first_token_difference_norm=difference, bidirectional_check_passed=True,
                  explanation='Identical first tokens respond to later changed tokens with dropout disabled.')
    write_json(args.run_dir / 'attention_check.json', result)
    print(result)


if __name__ == '__main__':
    main()
