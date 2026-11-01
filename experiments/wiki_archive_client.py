"""Persistent Windows -> WSL CPU bridge; archive helper never performs network I/O."""
import base64
import json
from pathlib import Path
import subprocess
import threading

ROOT=Path(__file__).resolve().parents[1]
STORE='C:/Users/jsowa/benchmark-translations/data-ood-20261030/A/wiki-recovery-20261031'


class ArchiveDeferred(RuntimeError):
    def __init__(self,result):
        self.retry_at=result['retry_at'];super().__init__(result['reason'])


class ArchiveTransportError(RuntimeError):pass


class WikiArchiveClient:
    def __init__(self):self.process=None;self.log=None;self.lock=threading.RLock()

    def start(self):
        path=ROOT/'reports/wikimedia_pipeline/archive_helper.log';path.parent.mkdir(parents=True,exist_ok=True)
        self.log=path.open('ab',buffering=0)
        helper=ROOT/'experiments/wiki_compact_archive.py'
        text=str(helper).replace('\\','/');helper_wsl='/mnt/'+text[0].lower()+'/'+text[3:]
        self.process=subprocess.Popen(['wsl.exe','-e','/home/jsowa/.venvs/pl-mmb-benchmark/bin/python','-B','-u',helper_wsl,'--store',STORE],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW)

    def request(self,operation,**kwargs):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                if self.log:self.log.close()
                self.start()
            try:
                self.process.stdin.write(json.dumps(dict(op=operation,**kwargs),ensure_ascii=False)+'\n');self.process.stdin.flush()
                line=self.process.stdout.readline()
            except (BrokenPipeError,OSError) as exc:raise ArchiveTransportError('Archive helper transport failed: '+str(exc)) from exc
            if not line:raise ArchiveTransportError('Archive helper exited; see archive_helper.log')
            response=json.loads(line)
            if not response['ok']:raise ValueError(response['error'])
            result=response['result']
            if result.get('action')=='deferred':raise ArchiveDeferred(result)
            return result

    def begin(self,record):return self.request('begin',record=record)
    def commit(self,token,raw):return self.request('commit',token=token,raw_base64=base64.b64encode(raw).decode())
    def abort(self,token):return self.request('abort',token=token)

    def close(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.stdin.write('{"op":"close"}\n');self.process.stdin.flush();self.process.stdin.close()
                self.process.wait(timeout=30)
            if self.log:self.log.close()
