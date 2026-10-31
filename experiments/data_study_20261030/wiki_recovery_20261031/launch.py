"""Finalize user-approved storage policy; verify all queued scripts exist."""
from config import *


def main():
    policy=dict(status='approved_before_bulk_download',generation=2,created_at=utc(),
        user_authorization='No dobra, odpowiada mi Twoja propozycja. Zakolejkuj pracę, bo chcę iść spać.',
        acquisition='Download verified full original into RAM; compute original SHA1/rawSHA256/pixelSHA256/pHash before resize',
        archive='Whole frame, aspect ratio preserved, no upscale, long edge<=1536, JPEG quality92/subsampling0',
        current_training='Separate lossless PNG224 created directly from original using pinned TinyCLIP processor; bit-exact normalized input parity required',
        retained_originals='Keep all originals already on disk, including 20 smoke downloads; no new bulk raw archive',
        expected_new_storage_GB='20-25 based on 20-image sample; hard new-task cap remains36GiB',
        profiling_report=artifact('size_measurement',REPORT/'storage_measurement.json'),
        smoke_report=artifact('preprocessing_smoke',REPORT/'compact_smoke.json'),
        stages=['download_compact','audit_and_generation2_freeze','same_R0_R1_R2_R3_training_recipe','paired_bootstrap_and_report'],
        trained_architectures_unchanged=True,minimum_verified_Wiki_images=MIN_WIKI)
    target=REPORT/'archive_policy.json'
    if not target.exists():atomic_json(target,policy,immutable=True)
    for file in ('order_downloads.py','download_compact.py','audit_generation.py','train_generation.py','finish.py','overnight_queue.py','keep_awake.ps1'):
        assert Path(__file__).with_name(file).is_file(),file
    execution=dict(status='ready_for_background_launch',created_at=utc(),
        scripts=[artifact('queue_code',p) for p in Path(__file__).parent.glob('*.py')],
        archive_policy=artifact('approved_policy',target),systemd_unit='codex-wiki-recovery-20261031.service',
        log_root=str(STORE/'logs'),status_file=str(REPORT/'STATUS.json'),GPU_lock=str(GPU_LOCK))
    atomic_json(REPORT/'execution.json',execution)
    reserve();status('ready_for_background_launch',queue=policy['stages'],archive_profile='1536 + exact224',minimum_Wiki_images=MIN_WIKI)
    print('QUEUE_READY',str(REPORT/'STATUS.json'),flush=True)


if __name__=='__main__':main()
