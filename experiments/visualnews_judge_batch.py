"""Execute a bounded registered comparison with one shared concurrency limit."""
import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace
import visualnews_judge_pilot as pilot

async def main(args):
    for variant in args.variants.split(','):
        print('Starting '+variant, flush=True)
        await pilot.run(SimpleNamespace(variant=variant,scope=args.scope,repeats=args.repeats,limit=None,concurrency=args.concurrency))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--variants',default='A_direct,B_taxonomy,C_claims,D_skeptical,T1,T2'); p.add_argument('--scope',default='all'); p.add_argument('--repeats',type=int,default=1); p.add_argument('--concurrency',type=int,default=8)
    asyncio.run(main(p.parse_args()))
