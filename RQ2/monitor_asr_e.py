"""ASCII-only ASR-e progress. No prompts, responses, reasons or credentials printed."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent


def show():
    files=list((ROOT/'outputs/asr_e').glob('batch_*.json'))
    if not files:
        print('ASR-e batch has not started.',flush=True)
        return
    path=max(files,key=lambda p:p.stat().st_mtime)
    state=json.loads(path.read_text(encoding='utf-8'))
    tag=path.stem.removeprefix('batch_')
    print(datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST'),flush=True)
    print(json.dumps({'batch':state['status'],'model':state['settings']['judge_model'],
                      'current_condition':state.get('current_condition')},ensure_ascii=True),flush=True)
    total=0
    for c,item in state['conditions'].items():
        target=ROOT/f'outputs/full/without_{c}_seed0_n100/eval_{tag}.json'
        rows=json.loads(target.read_text(encoding='utf-8')) if target.exists() else []
        labels=[(r.get('evaluation_result') or {}).get('response_safety_category') for r in rows]
        safe=labels.count('SAFE_REFUSAL');unsafe=labels.count('UNSAFE_COMPLIANCE')
        total+=safe+unsafe
        print(json.dumps({'condition':c,'status':item['status'],'scored':safe+unsafe,
                          'safe':safe,'unsafe':unsafe,'unscored':100-safe-unsafe},ensure_ascii=True),flush=True)
    print(f'Total scored: {total}/600',flush=True)
    print('',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--interval',type=float,default=5)
    args=parser.parse_args()
    if args.interval<=0:parser.error('--interval must be positive')
    try:
        while True:
            show()
            if args.once:return
            time.sleep(args.interval)
    except KeyboardInterrupt:return


if __name__=='__main__':main()
