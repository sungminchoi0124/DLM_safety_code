"""Judge the six stored Without runs using one explicit ASR-e configuration.

Requires a judge model and local OPENAI_API_KEY. --preflight-only makes no API calls.
Generation files are read-only; judgments and reasoning remain in ignored outputs.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse
from DiffuGuard.eval_asr import load_env
from .data import REPO
from .evaluate_asr import validate
from .run_asr import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge-model')
    parser.add_argument('--judge-prompt',choices=['code','figure11'],default='code')
    parser.add_argument('--question',choices=['auto','vanilla'],default='auto')
    parser.add_argument('--reasoning-effort',default='none')
    parser.add_argument('--max-completion-tokens',type=int,default=800)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    load_env()
    model=args.judge_model or os.environ.get('JUDGE_MODEL')
    settings={'judge_model':model,'judge_prompt':args.judge_prompt,'question':args.question,
              'reasoning_effort':args.reasoning_effort,'max_completion_tokens':args.max_completion_tokens}
    endpoint=os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
    provenance={**settings,'base_url':endpoint,'reasoning_override':os.environ.get('JUDGE_REASONING',''),
                'evaluation_source_sha256':hashlib.sha256((REPO/'DiffuGuard/third_party/DiffuGuard/evaluation.py').read_bytes()).hexdigest()}
    paths={c:REPO/f'RQ2/outputs/full/without_{c}_seed0_n100' for c in ['0','A','B','C','D','E']}
    sources={}
    for c,p in paths.items():
        raw=[json.loads(s) for s in (p/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
        validate(raw)
        if len(raw)!=100:raise ValueError('Expected 100 full-run responses')
        sources[c]=hashlib.sha256((p/'raw.jsonl').read_bytes()).hexdigest()
    missing=[]
    if not model:missing.append('judge_model')
    if not os.environ.get('OPENAI_API_KEY'):missing.append('OPENAI_API_KEY')
    info={'total_responses':600,'settings':settings,'endpoint_host':urlparse(endpoint).hostname,
          'missing':missing,'api_calls_started':False}
    print(json.dumps(info,ensure_ascii=True,indent=2),flush=True)
    if args.preflight_only:return
    if missing:parser.error('Missing: '+', '.join(missing)+'; configure the judge model and local .env before execution')
    if args.workers<1 or args.max_completion_tokens<1:parser.error('Workers/token budget must be positive')
    root=REPO/'RQ2/outputs/asr_e';root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.judge.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tag=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:12]
    state={'status':'running','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
           'settings':provenance,'conditions':{c:{'status':'pending'} for c in paths}}
    status_path=root/f'batch_{tag}.json'
    write_json(status_path,state)
    reports={}
    try:
        for c,p in paths.items():
            state['current_condition']=c
            state['conditions'][c]['status']='running';write_json(status_path,state)
            target=p/f'eval_{tag}.json'
            marker=Path(str(target)+'.run_config.json')
            stamp={**provenance,'source_raw_sha256':sources[c]}
            if marker.exists():
                if json.loads(marker.read_text())!=stamp:
                    raise ValueError('Existing judgments use different endpoint/settings/source')
            elif target.exists():
                raise ValueError('Existing verdicts have no matching batch provenance; refusing silent reuse')
            write_json(marker,stamp)
            command=[sys.executable,'-m','RQ2.evaluate_asr',str(p),'--judge-model',model,
                     '--judge-prompt',args.judge_prompt,'--question',args.question,
                     '--reasoning-effort',args.reasoning_effort,'--max-completion-tokens',str(args.max_completion_tokens),
                     '--workers',str(args.workers)]
            print(f'Judging Without {c}: 100 stored responses',flush=True)
            with (root/f'{tag}_without_{c}.log').open('a') as stream:
                subprocess.run(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True)
            summary=json.loads((p/f'summary_{tag}.json').read_text())
            if hashlib.sha256((p/'raw.jsonl').read_bytes()).hexdigest()!=sources[c]:
                raise ValueError('Generation source changed during judging')
            reports[c]=summary
            state['conditions'][c].update(status='done' if summary['unscored']==0 else 'partial',
                                          safe=summary['safe'],unsafe=summary['unsafe'],unscored=summary['unscored'],asr_e=summary['asr_e'])
            write_json(status_path,state)
            if summary['unscored']:
                raise RuntimeError('Some judgments are unscored; batch stopped for retry rather than recording completion')
        write_json(REPO/f'RQ2/results/without_0_A_B_C_D_E_asr_e_{tag}.json',
                   {'settings':provenance,'n_per_condition':100,'conditions':reports})
        csv_path=REPO/f'RQ2/results/without_0_A_B_C_D_E_asr_e_{tag}.csv'
        csv_rows=[]
        for c,summary in reports.items():
            csv_rows.append({'condition':c,'n':summary['n'],'asr_k_percent':summary['asr_k'],
                             'asr_e_percent':summary['asr_e'],'safe':summary['safe'],
                             'unsafe':summary['unsafe'],'unscored':summary['unscored'],
                             'empty_response_rate':summary['empty_response_rate'],
                             'non_answer_rate':summary['non_answer_rate'],
                             'wall_seconds':summary['wall_ms_sum']/1000,
                             'total_nfe':summary['total_nfe_sum'],'judge_model':model,
                             'judge_prompt':args.judge_prompt,'question':args.question,
                             'reasoning_effort':args.reasoning_effort,
                             'max_completion_tokens':args.max_completion_tokens})
        with csv_path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
        state.update(status='done',current_condition=None,summary_csv=str(csv_path))
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__)
        raise
    finally:
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        write_json(status_path,state)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    main()
