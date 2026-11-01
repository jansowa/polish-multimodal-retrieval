# Coordination API v1 (WSL Python 3)

Add `experiments/data_ood_common_20261030` to `sys.path`, then:

```python
from common import gpu_lock, reserve, verify_marker, COORD
verify_marker(COORD / 'COMMON_READY.json')
# B must additionally verify A_FINALISTS_FROZEN, A_GPU_DONE, B_EVAL_FROZEN.
reserve('B', 'download', {'C': 100_000_000}, committed_bytes=0,
        temporary_bytes=20_000_000)
with gpu_lock(blocking=False):
    pass  # All CUDA work belongs inside this context.
# After writing: remaining bytes decrease; committed_bytes is the owner's
# TOTAL persistent study bytes already on disk, across all its reservations.
reserve('B', 'download', {'C': 0}, committed_bytes=80_000_000)
```

Reservations are remaining/unwritten bytes, including temporary bytes; update
after every write. They cannot guarantee space against unrelated pipelines, so
recheck immediately before large writes. C and D each retain 4 GiB; A/B persistent
caps are 8/3 GiB and shared temporary cap is 4 GiB. All callers use the same WSL
`disk_reservations.lock`; never edit DISK_RESERVATIONS.json directly. Keep
committed accounting monotonic except explicitly authorized cleanup of owned data.
`atomic_json(path, obj, immutable=True)` publishes complete JSON without replacement.
`publish` is for A-owned readiness markers; B constructs its own owner envelope.
Never treat STATUS as readiness. `verify_marker` verifies contract and artifacts;
consumers also validate phase, generation, status and their own dependencies.
GPU lock is exactly the historical worker.lock, using POSIX flock.
