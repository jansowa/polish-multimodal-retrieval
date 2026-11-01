import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training import wiki_pilot_data as data
if __name__=='__main__':
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--parity-only',action='store_true');parser.add_argument('--references-only',action='store_true');args=parser.parse_args()
    with (data.vn.OUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data.torch.set_num_threads(4);data.torch.backends.cuda.matmul.allow_tf32=False;data.torch.backends.cudnn.allow_tf32=False
        if args.references_only:
            ctx=data.vn.Context();reference=data.protection(ctx)
            print('REFERENCE_PHASHES_READY',reference['reference_phashes'],flush=True)
        else:data.prepare(args.parity_only)
