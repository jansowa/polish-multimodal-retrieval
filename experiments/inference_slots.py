"""A hard shared request cap with configured shares and borrowing on queue exhaustion."""
import asyncio
from contextlib import asynccontextmanager


class SharedSlots:
    def __init__(self,total,demands,shares=None,work_conserving=False):
        if total<2 or total%2 or len(demands)!=2:raise ValueError('Two queues require an even total')
        self.total=total;self.demands=demands
        self.shares=dict(shares) if shares is not None else {name:total//2 for name in demands}
        if set(self.shares)!=set(demands) or any(type(value) is not int or value<1 for value in self.shares.values()) or sum(self.shares.values())!=total:
            raise ValueError('Positive queue shares must sum to the shared total')
        self.active={name:0 for name in demands}
        self.waiting={name:0 for name in demands}
        self.work_conserving=work_conserving
        self.condition=asyncio.Condition()

    def limit(self,name):
        other=next(key for key in self.demands if key!=name)
        if self.work_conserving and self.waiting[other]==0:return self.total
        return self.shares[name] if self.demands[other]() else self.total

    @asynccontextmanager
    async def slot(self,name):
        async with self.condition:
            self.waiting[name]+=1
            try:
                while sum(self.active.values())>=self.total or self.active[name]>=self.limit(name):
                    try:await asyncio.wait_for(self.condition.wait(),1)
                    except asyncio.TimeoutError:pass
                self.active[name]+=1
            finally:
                self.waiting[name]-=1
                self.condition.notify_all()
        try:yield
        finally:
            async with self.condition:
                self.active[name]-=1;self.condition.notify_all()

    def snapshot(self):
        return dict(total_max_parallel_requests=self.total,active=dict(self.active),
            waiting_for_inference=dict(self.waiting),configured_shares=dict(self.shares),work_conserving=self.work_conserving,
            limits={name:self.limit(name) for name in self.demands})
