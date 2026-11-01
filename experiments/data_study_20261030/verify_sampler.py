"""Whole-run parity and FULL-slot preservation, including the 145-row batches."""
import json
import torch
from run import Sampler, historical, ROOT, REPORT, atomic_json


def main():
    fixture=[dict(target_id=str(i//2),id=str(i)) for i in range(38)]
    wiki=[dict(target_id='wiki'+str(i),id='w'+str(i)) for i in range(7)]
    checks={}
    for seed in (42,43,44):
        old=historical.Sampler(dict(mode='replace',base_pool=list(range(41))),41,fixture,41,seed=seed)
        exact=Sampler(41,fixture,[],seed,0)
        mixes={share:Sampler(41,fixture,wiki,seed,share) for share in (3,10)}
        total=0;full=0;wk={3:0,10:0}
        for step in range(5500):
            n=145 if step%275==274 else 256;a=old.draw(n);b=exact.draw(n)
            assert torch.equal(a,b)
            for share,mix in mixes.items():
                c=mix.draw(n);assert torch.equal(c[:n//2],a[:n//2])
                assert (c[n//2:]>=41).all()
                wk[share]+=int((c>=41+len(fixture)).sum())
            total+=n;full+=n//2
        assert total==1405780 and full==702880
        assert all(wk[s]==total*s//100 for s in wk)
        checks[str(seed)]=dict(historical_R0_R1_exact=True,FULL_slots_exact=True,
            total=total,FULL=full,Wiki=wk)
    atomic_json(REPORT/'sampler_verification.json',dict(status='passed',checks=checks))
    print(json.dumps(checks))


if __name__=='__main__':main()
