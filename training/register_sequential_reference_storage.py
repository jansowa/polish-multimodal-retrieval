"""Record explicit user authorization for sequential reference-weight removal."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training.series_20261023 import OUT,ROOT,read,freeze,file_sha


def main():
    original=OUT/'reference_eval_registration_v2_20261029.json'
    registration=copy.deepcopy(read(original))
    registration['schema']=3
    registration['supersedes']=dict(path=str(original.relative_to(ROOT)),sha256=file_sha(original),
        reason='User explicitly instructed: load one model, benchmark it, delete its weights, then load the next. Storage policy only; scientific protocol, prompts, models, checkpoints and datasets unchanged.')
    registration['cache_and_storage'].update(automatic_deletion=True,
        weights_retained_until_user_manually_deletes_them=False,
        cleanup_scope='Only completed reference-model weight files in its native HF repository cache, plus its registered image component; retain configs, author sources, results and embeddings. Never remove TinyCLIP-39M or PolDense.',
        user_authorization='Ale po co miałbyś pobierać 20 modeli?!?!?! Ładuj po jednym, benchmarkuj i usuwaj!',
        simultaneous_reference_models=1,entire_list_weight_storage_not_required=True)
    registration['implementation_sha256']['training/run_series_reference_evaluation.py']=file_sha(ROOT/'training/run_series_reference_evaluation.py')
    registration['implementation_sha256']['src/pl_mmb_benchmark/efficiency.py']=file_sha(ROOT/'src/pl_mmb_benchmark/efficiency.py')
    freeze(OUT/'reference_eval_registration_v3_20261029.json',registration)
    print('SEQUENTIAL_REFERENCE_STORAGE_REGISTERED',flush=True)


if __name__=='__main__':main()
