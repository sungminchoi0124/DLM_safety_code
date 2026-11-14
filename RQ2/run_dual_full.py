"""Run a verified ten-item E pilot, then E 100 items; no ASR-e calls."""
import fcntl
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from .data import REPO
from .evaluate_asr import validate
from .run_asr import write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def check_cache(raw):
    validate(raw)
    builds = sum(b['cache_build_nfe'] for r in raw for b in r['blocks'])
    reuses = sum(b['cache_reuse_nfe'] for r in raw for b in r['blocks'])
    if builds != 3*len(raw) or reuses < 1:
        raise RuntimeError('Dual cache reuse was not demonstrated; full run blocked')
    return {'cache_build_nfe':builds,'cache_reuse_nfe':reuses,
            'items_with_cache_reuse':sum(any(b['cache_reuse_nfe']>0 for b in r['blocks']) for r in raw),
            'first_block_cache_disabled':True,'remaining_masks_zero':True,
            'dual_replace_nfe':sum(b['dual_replace_nfe'] for r in raw for b in r['blocks'])}


def main():
    full = REPO/'RQ2/outputs/full'
    full.mkdir(parents=True,exist_ok=True)
    lock = (full/'.without_full.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    paths = {'pilot':REPO/'RQ2/outputs/pilot/without_E_seed0_limit10',
             'full':full/'without_E_seed0_n100'}
    for p in paths.values():
        if p.exists() and any(p.iterdir()):
            raise ValueError(f'Refusing overwrite: {p}')
    memory = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(memory.strip().splitlines()[0])>1000:
        raise RuntimeError('GPU occupied')
    state={'status':'running','pid':os.getpid(),'started_utc':now(),'phase':'pilot',
           'asr_e':'not_run','pilot':{'status':'pending','path':str(paths['pilot'])},
           'full':{'status':'pending','path':str(paths['full'])}}
    state_path=full/'without_E_batch_status.json'
    write_json(state_path,state)
    reports={}
    try:
        for phase,limit in [('pilot',10),('full',100)]:
            p=paths[phase]
            state.update(phase=phase)
            state[phase].update(status='running',started_utc=now())
            command=[sys.executable,'-m','RQ2.run_asr','--accel','E','--limit',str(limit),
                     '--verify-original','--output-dir',str(p)]
            state[phase]['command']=command
            write_json(state_path,state)
            print(f'{now()} starting E {phase}: {limit} items',flush=True)
            log=REPO/f'RQ2/outputs/logs/without_E_seed0_{"limit10" if phase=="pilot" else "n100"}.log'
            with log.open('x') as stream:
                subprocess.run(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True)
            with log.open('a') as stream:
                subprocess.run([sys.executable,'-m','RQ2.evaluate_asr',str(p)],cwd=REPO,
                               stdout=stream,stderr=subprocess.STDOUT,check=True)
            raw=[json.loads(s) for s in (p/'raw.jsonl').read_text().splitlines()]
            cache=check_cache(raw)
            reference=json.loads((p/'reference_check.json').read_text())
            if not reference['tokens_equal'] or reference['reference_nfe']!=reference['actual_nfe']:
                raise RuntimeError('Original Dual reference check failed')
            meta=json.loads((p/'config_resolved.json').read_text())
            summary=json.loads((p/'summary_local.json').read_text())
            report={'condition':'E','run_dir':str(p),'summary':summary,'cache_verification':cache,
                    'reference_check':reference,'generation_seconds':meta['generation_seconds'],
                    'model_load_seconds':meta['model_load_seconds'],'process_seconds':meta['process_seconds'],
                    'model_revision':meta['model_revision'],'prompt_sha256':meta['prompt_sha256'],
                    'ids_sha256':meta['ids_sha256'],'protocol':meta['protocol'],
                    'protocol_sha256':meta['protocol_sha256'],'command':command,
                    'git_commit':meta['git_commit'],'git_dirty':meta['git_dirty']}
            if phase=='full':
                previous=[json.loads(s) for s in (paths['pilot']/'raw.jsonl').read_text().splitlines()]
                matches=sum(a['output_sha256']==b['output_sha256'] for a,b in zip(raw[:10],previous))
                report['pilot_first_10_output_hash_matches']=matches
                if matches!=10:raise RuntimeError('E pilot/full reproducibility mismatch')
            a_path=REPO/f'RQ2/outputs/{"pilot/without_A_seed0_limit10" if phase=="pilot" else "full/without_A_seed0_n100"}/raw.jsonl'
            a_rows=[json.loads(s) for s in a_path.read_text().splitlines()]
            report['same_scored_response_hash_as_A']=sum(a['output_sha256']==b['output_sha256'] for a,b in zip(raw,a_rows))
            report['scoring_note']='DIJA scoring uses legacy filled-user-turn extraction. Cache begins after block 0; equality with A may follow from this protocol and does not independently demonstrate defense robustness.'
            write_json(REPO/f'RQ2/results/without_E_{phase}.json',report)
            reports[phase]=report
            state[phase].update(status='done',finished_utc=now(),n_items=summary['n'],
                                asr_k=summary['asr_k'],cache_verification=cache)
            write_json(state_path,state)
            print(f'{now()} E {phase} validated; cache reuse calls={cache["cache_reuse_nfe"]}',flush=True)
        all_reports={c:json.loads((REPO/f'RQ2/results/without_{c}_full.json').read_text()) for c in ['0','A','B','C','D','E']}
        for c in all_reports:
            for key in ['model_revision','prompt_sha256','ids_sha256']:
                assert all_reports[c][key]==all_reports['0'][key],key
            assert all_reports[c]['summary']['item_ids']==all_reports['0']['summary']['item_ids']
        write_json(REPO/'RQ2/results/without_0_A_B_C_D_E_full.json',{
            'n_per_condition':100,'asr_e':'not_run','same_ids_and_order':True,
            'rows':[{'accel':c,'threshold':all_reports[c]['reference_check']['config']['threshold'],
                     'asr_k':all_reports[c]['summary']['asr_k'],
                     'wall_ms_sum':all_reports[c]['summary']['wall_ms_sum'],
                     'total_nfe':all_reports[c]['summary']['total_nfe_sum'],
                     'speedup_vs_0':all_reports['0']['summary']['wall_ms_sum']/all_reports[c]['summary']['wall_ms_sum']}
                    for c in ['0','A','B','C','D','E']]})
        csv_path=REPO/'RQ2/results/without_0_A_B_C_D_E_summary.csv'
        csv_rows=[]
        for c in ['0','A','B','C','D','E']:
            item=all_reports[c]
            summary=item['summary']
            csv_rows.append({'condition':c,'n':100,'threshold':item['reference_check']['config']['threshold'],
                             'cache':item['reference_check']['config']['cache'],'asr_k_percent':summary['asr_k'],
                             'wall_seconds':summary['wall_ms_sum']/1000,'total_nfe':summary['total_nfe_sum'],
                             'speedup_vs_0':all_reports['0']['summary']['wall_ms_sum']/summary['wall_ms_sum'],
                             'asr_e':'not_run'})
        with csv_path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
        state.update(status='done',phase=None,finished_utc=now())
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__,finished_utc=now())
        state[state['phase']].update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        write_json(state_path,state)
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


if __name__=='__main__':
    main()
