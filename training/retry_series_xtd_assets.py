"""Resume only missing XTD originals, with one connection and bounded retries."""
import concurrent.futures
import time
import urllib.request
import prepare_series_xtd_assets as job

Pool=concurrent.futures.ThreadPoolExecutor
open_url=urllib.request.urlopen


def retry_url(*args,**kwargs):
    for attempt in range(3):
        try:return open_url(*args,**kwargs)
        except Exception:
            if attempt==2:raise
            time.sleep((2,5)[attempt])


if __name__=='__main__':
    job.freeze(job.DEST/'retry_acquisition_decision.json',dict(status='registered_before_retry',
        prior_decision_sha256=job.file_sha(job.DEST/'acquisition_decision.json'),
        script_sha256=job.file_sha(__file__),workers=1,max_attempts=3,backoff_seconds=[2,5],
        scope='Only missing originals; existing receipts and file SHA reverified; no model runs'))
    concurrent.futures.ThreadPoolExecutor=lambda max_workers:Pool(max_workers=1)
    urllib.request.urlopen=retry_url
    job.main()
