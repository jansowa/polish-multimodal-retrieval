"""Record user's pause and release only B reservations after B workers stop."""
from datetime import datetime, timezone
import json
from pathlib import Path
from prepare import COORD, OUT, ROOT, load_common, committed_bytes, sha, write
from acquire import ASSETS


def main():
    # Verify no acquisition/research process remains before releasing its budget.
    active = []
    names = ('vizwiz_assets.py', 'deep_research.py', 'prepare_candidates.py', 'acquire.py')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if 'experiments/ood_eval_20261030/' in command and any(name in command for name in names):
            active.append(dict(pid=int(process.name), command=command))
    if active:
        raise RuntimeError('B workers still active: ' + repr(active))
    common = load_common()
    registry = json.loads((COORD / 'DISK_RESERVATIONS.json').read_text())
    for token, entry in registry.get('reservations', {}).items():
        if entry['owner'] == 'B':
            common.reserve('B', token.split(':', 1)[1], {'C': 0, 'D': 0}, committed_bytes=committed_bytes())
    images = list((ASSETS / 'vizwiz/images').glob('*.jpg'))
    complete = [image for image in images if image.with_suffix('.json').exists()]
    value = dict(schema_version=1, study_id='data-ood-20261030', generation=1, owner='B',
        phase='PAUSED_BY_USER', status='paused', updated_at=datetime.now(timezone.utc).isoformat(),
        contract_sha256=sha(COORD / 'COORDINATION.md'), pause_reason='User is shutting down computer; resume only when asked.',
        CUDA_operations=0, models_evaluated=0, manual_annotations_requested=0, active_B_workers=[],
        vizwiz_originals_saved=len(images), vizwiz_originals_with_fingerprints=len(complete),
        asset_root=common.windows_path(ASSETS), B_reservations_remaining_bytes=0,
        resume_notes=[
            'Read COORDINATION.md and all current A marker hashes again; status is not readiness.',
            'COMMON_READY was verified. Use existing WSL common helper for all reservations and GPU lock.',
            'A_UNIVERSE_FROZEN appeared but was not yet consumed or fully verified by B.',
            'Flickr test parquet (1000 images, 5000 EN/PL captions) downloaded, source SHA verified; original filenames present.',
            'Original DOCCI descriptions and VizWiz caption annotation ZIP saved on C; official rejection flags inspected.',
            'VizWiz 1250-member candidate acquisition interrupted by user; script resumes completed originals without deletion.',
            'Use /home/jsowa/.venvs/pl-mmb-legacy-cu130/bin/python for VizWiz fingerprints; repository .venv lacks scipy.',
            'deep_research_v2.json may be absent/incomplete: translator choice, weights, translations not started.',
            'No final dataset selection/freeze, overlap audit, model scores, GPU use, B_EVAL_FROZEN or B_RESULTS_READY.',
            '7 CPU tests passed: test_cpu.py; no new manual annotations.',
            'Check saved image/.json pairs and retained .part files before resuming; never delete datasets/cache.',
            'Model evaluation requires A_FINALISTS_FROZEN + A_GPU_DONE + B_EVAL_FROZEN and shared GPU lock.'
        ])
    write(OUT / 'resume_state.json', value)
    write(COORD / 'B_STATUS.json', value)
    # write() settles B:metadata itself; all B reservations remain at zero.
    print(json.dumps(dict(status='paused', saved_vizwiz_images=len(images), verified_image_metadata=len(complete),
                          B_workers=0, B_reservations_remaining_bytes=0)))


if __name__ == '__main__':
    main()
