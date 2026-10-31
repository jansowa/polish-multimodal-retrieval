"""Conservative recovery of a valid JSON state with trailing write-collision bytes."""
import json
import os
import signal
from config import *


def main():
    # Stop only this queue's acquisition child. Its parent owns retry/resume.
    stopped=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            script=next((a.decode() for a in args if a.endswith(b'/download_compact.py')),None)
            if script and Path(script).resolve()==Path(__file__).with_name('download_compact.py').resolve():
                os.kill(int(proc.name),signal.SIGTERM);stopped.append(int(proc.name))
        except (OSError,UnicodeError):pass
    p=ROOT/'outputs/wikimedia_http_state.json';incident=REPORT/'http_state_recovery';incident.mkdir(exist_ok=True)
    with lock(p.with_suffix(p.suffix+'.lock')):
        raw=p.read_bytes();text=raw.decode('utf-8')
        value,end=json.JSONDecoder().raw_decode(text)
        assert isinstance(value.get('cooldowns'),dict) and isinstance(value.get('next_request_at'),(int,float))
        assert isinstance(value.get('adaptive_interval'),(int,float))
        if text[end:].strip():
            backup=incident/('corrupt_state_'+str(__import__('time').time_ns())+'.txt');backup.write_bytes(raw)
            atomic_json(p,value)
    recovered=[]
    for receipt in (STORE/'compact_receipts').glob('*.json'):
        v=read(receipt)
        if v.get('status')=='failed' and v.get('reason','').startswith('JSONDecodeError:'):
            recovered.append(v)
            v.update(status='retry',next_retry_at=0,reason='Shared HTTP state write collision repaired; retry actual acquisition')
            atomic_json(receipt,v)
    atomic_json(incident/'recovery.json',dict(status='complete',preserved_cooldowns_and_pacing=value,
        recovered_receipts=len(recovered),previous_failures=recovered,stopped_own_acquisition_pids=stopped,
        fix='unique atomic temp file in this worker only; shared code and other pipeline allocations unchanged'))
    print('HTTP_STATE_REPAIRED; REQUEUED',len(recovered),'OWN_CHILD_RESTART',stopped,flush=True)


if __name__=='__main__':main()
