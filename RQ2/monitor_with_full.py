"""ASCII-only monitor. Ctrl-C stops the monitor, not the detached experiment."""
import argparse
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import time
from .run_asr_with_full import DEFAULT_ROOT,ORDER


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def show(root):
    print(datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST'),flush=True)
    s=read(root/'batch_status.json')
    print(json.dumps({k:s.get(k) for k in ['status','stage','current_condition','pid','error_type']},ensure_ascii=True),flush=True)
    total=0
    for c in ORDER:
        child=read(root/f'with_{c}/status.json');n=child.get('n_done',0);total+=n
        fallback=s.get('conditions',{}).get(c,{})
        print(f"With {c}: {child.get('status',fallback.get('status','pending'))} {n}/100",flush=True)
    print(f'Generation: {total}/600',flush=True)
    j=read(root/'asr_e_status.json')
    if j:
        scored=0
        for model,meta in j.get('models',{}).items():
            tag=meta['tag'];count=0
            for c in ORDER:
                verdict=read(root/f'with_{c}/eval_{tag}.json')
                if isinstance(verdict,list):
                    count+=sum((x.get('evaluation_result') or {}).get('response_safety_category') in ['SAFE_REFUSAL','UNSAFE_COMPLIANCE'] for x in verdict)
            scored+=count;print(f'Judge {model}: {meta["status"]} {count}/600',flush=True)
        print(f'Judged: {scored}/1200',flush=True)
        print(json.dumps({k:j.get(k) for k in ['status','current_model','current_condition','error_type']},ensure_ascii=True),flush=True)
    print('',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--once',action='store_true')
    p.add_argument('--interval',type=float,default=10)
    a=p.parse_args()
    if a.interval<=0:p.error('interval must be positive')
    try:
        while True:
            show(a.output_root.resolve())
            if a.once:return
            time.sleep(a.interval)
    except KeyboardInterrupt:return


if __name__=='__main__':main()
