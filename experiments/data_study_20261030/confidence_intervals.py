"""Conditional paired component bootstrap for the closed representatives; CPU."""
import sys
from preflight import ROOT,REPORT,STORAGE,read
sys.path.insert(0,str(ROOT/'src'))
from common import COORD,atomic_json,verify_marker
from pl_mmb_benchmark.full_gallery import paired_bootstrap


def main():
    marker=verify_marker(COORD/'A_FINALISTS_FROZEN.json')
    selection=read(REPORT/'selection_decision.json');wiki=selection['chosen_Wiki']
    reps=selection['representatives'];out={}
    names=read(REPORT/'training_report.json')['variants']['R0']
    for a,b in [('R1','R0'),(wiki,'R1')]:
        key=a+'_minus_'+b;out[key]={}
        for name in names:
            va=read(STORAGE/'evaluation'/a/f'seed-{reps[a]}'/(name+'.json'))
            vb=read(STORAGE/'evaluation'/b/f'seed-{reps[b]}'/(name+'.json'))
            out[key][name]={d:paired_bootstrap(va,vb,d,replicates=10000,seed=20261023) for d in ('t2i','i2t')}
    atomic_json(REPORT/'conditional_CI.json',dict(status='complete',
        conditional_on='closed representative checkpoints, fixed galleries, observed qrels/components; distinct from seed mean/SD',
        replicates=10000,seed=20261023,results=out))
    print('CONDITIONAL_BOOTSTRAP_COMPLETE',list(out))


if __name__=='__main__':main()
