"""Sequential full With runs: 0 -> A -> D -> E -> B -> C, then both judges.

This program runs in the foreground; launch with nohup to detach it.
--preflight-only validates local prerequisites without generation/API requests.
--resume skips verified completed conditions; partial failed conditions are archived
and rerun in full. Judge checkpoints resume without rejudging completed records.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .data import REPO, PROMPT_SHA, load_items
from .evaluate_asr import summarize
from .run_asr import write_json
from .types import THRESHOLDS, CACHE_MODES
from .upstream import git, MODEL_REVISION

ORDER = ['0','A','D','E','B','C']
DEFAULT_ROOT = REPO/'RQ2/outputs/full/with_d321365_seed0_n100'
COMBINED = Path('/root/rq2-b-d321365/RQ2/src/combined.py')
PROTOCOL = REPO/'RQ2/configs/asr_with_full_protocol.json'
SOURCES = ['RQ2/run_asr_with_full.py','RQ2/run_asr_with0_pilot.py','RQ2/cache_probe.py',
           'RQ2/judge_with_full.py','RQ2/data.py','RQ2/types.py','RQ2/upstream.py',
           'RQ2/evaluate_asr.py','RQ2/run_asr.py','DiffuGuard/eval_asr.py',
           'DiffuGuard/configs/llada_dija.yaml','RQ2/configs/asr_with_full_protocol.json',
           'RQ2/configs/asr_e_judge.json','RQ2/configs/asr_e_judge_gpt4omini.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def preflight(root, generation_only):
    import torch
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['status'] != 'adopted_by_user_for_With_full_runs':
        raise ValueError('Full-run protocol has not been adopted')
    if protocol['condition_order'] != ORDER or protocol['input_sha256'] != PROMPT_SHA:
        raise ValueError('Protocol order/input changed')
    if any(protocol[k]!=v for k,v in dict(model_revision=MODEL_REVISION,gen_length=128,block_length=32,
                                        temperature=.5,base_seed=0,alpha0=.3,repair_steps=8,
                                        cache_start_block=1,protected_fixed_prompt=True).items()):
        raise ValueError('Adopted protocol differs from the validated decoder settings')
    if git(COMBINED.parent,'rev-parse','HEAD') != protocol['combined_commit']:
        raise ValueError('Combined decoder commit differs from the adopted one')
    if git(COMBINED.parent,'status','--porcelain'):
        raise ValueError('Combined decoder worktree has local changes')
    items = load_items(REPO/'RQ2/data/ids/dija_100.txt')
    if len(items) != 100 or not torch.cuda.is_available():
        raise ValueError('Fixed 100 items and a CUDA GPU are required')
    versions = dict(torch=torch.__version__,transformers=importlib.metadata.version('transformers'))
    for c in ORDER:
        p=REPO/f'RQ2/outputs/pilot/with_{c}_seed0_n10_d321365'
        m=json.loads((p/'config_resolved.json').read_text());s=json.loads((p/'summary_local.json').read_text())
        if s['validation']!='passed' or s['n']!=10 or s['max_remaining_masks']!=0:
            raise ValueError('Pilot did not pass: '+c)
        if any(m[k]!=v for k,v in versions.items()) or m['combined_sha256']!=sha(COMBINED):
            raise ValueError('Pilot environment/decoder no longer matches: '+c)
        if sha(p/'source_snapshot/llada_dija.yaml')!=sha(REPO/'DiffuGuard/configs/llada_dija.yaml'):
            raise ValueError('Defense preset changed since the pilot')
    if not generation_only:
        from DiffuGuard.eval_asr import load_env
        import os
        load_env()
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY missing; configure locally or use --generation-only')
        provenance_common=dict(base_url=os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1',
                               reasoning_override=os.environ.get('JUDGE_REASONING',''),
                               evaluation_source_sha256=sha(REPO/'DiffuGuard/third_party/DiffuGuard/evaluation.py'))
        for name in ['asr_e_judge.json','asr_e_judge_gpt4omini.json']:
            cfg=json.loads((REPO/'RQ2/configs'/name).read_text())
            settings={k:cfg[k] for k in ['judge_model','judge_prompt','question','reasoning_effort','max_completion_tokens']}
            tag=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:12]
            for c in ORDER:
                d=REPO/f'RQ2/outputs/full/without_{c}_seed0_n100'
                marker=json.loads((d/f'eval_{tag}.json.run_config.json').read_text())
                if any(marker[k]!=v for k,v in dict(settings,**provenance_common).items()):
                    raise ValueError('Judge no longer matches existing Without labels')
                if sha(d/'raw.jsonl')!=marker['source_raw_sha256']:
                    raise ValueError('Without response source changed')
    return dict(order=ORDER,n_per_condition=100,combined_commit=protocol['combined_commit'],
                model_revision=MODEL_REVISION,input_sha256=PROMPT_SHA,versions=versions,
                source_sha256={name:sha(REPO/name) for name in SOURCES},combined_sha256=sha(COMBINED),
                judge_after_generation=not generation_only,
                projected_inference_minutes=133.04)


def validate_run(path,c):
    status=json.loads((path/'status.json').read_text())
    if status['status']!='done':raise ValueError('Condition not completed: '+c)
    m=json.loads((path/'config_resolved.json').read_text())
    if m['run_kind']!='full' or m['n_items']!=100 or m['accel']!=c:
        raise ValueError('Expected a full 100-item run')
    if m['combined_commit']!='d321365a1f2dad28fdda36e1b540b04c0a56b624' or m['model_revision']!=MODEL_REVISION:
        raise ValueError('Wrong model/decoder revision')
    rows=[json.loads(x) for x in (path/'raw.jsonl').read_text().splitlines() if x.strip()]
    if [x['item_id'] for x in rows]!=[f'dija-{i:03d}' for i in range(100)]:
        raise ValueError('Wrong item count/order')
    for i,x in enumerate(rows):
        if x['source_index']!=i or x['item_seed']!=i or x['seed']!=0:
            raise ValueError('Seed policy mismatch')
        if x['accel']!=c or x['threshold']!=THRESHOLDS[c] or x['cache']!=CACHE_MODES[c]:
            raise ValueError('Condition mismatch')
        if x['gen_length']!=128 or x['block_length']!=32 or x['temperature']!=.5:
            raise ValueError('Generation settings mismatch')
        if not (x['gen_nfe']==128 if c=='0' else 4<=x['gen_nfe']<=x['prompt_masks_initial']+128):
            raise ValueError('Generation NFE outside the adopted bounds')
        if x['n_masks'] or not x['fixed_prompt_unchanged']:
            raise ValueError('Masks/fixed prompt verification failed')
        if x['baseline_nfe']!=1 or x['audit_nfe']!=0 or x['repair_nfe']!=(8 if x['audit_record']['repaired'] else 0):
            raise ValueError('Defense NFE mismatch')
        if x['total_nfe']!=sum(x[k] for k in ['gen_nfe','baseline_nfe','audit_nfe','repair_nfe']) or x['total_nfe']!=x['measured_forward_calls']:
            raise ValueError('Measured NFE mismatch')
        if hashlib.sha256(x['output_text'].encode()).hexdigest()!=x['output_sha256']:
            raise ValueError('Response hash mismatch')
    s=summarize(rows)
    stored=json.loads((path/'summary_local.json').read_text())
    if stored['validation']!='passed' or stored['total_nfe_sum']!=s['total_nfe_sum']:
        raise ValueError('Summary verification failed')
    if c in ['D','E'] and (stored['cache_build_nfe_sum']!=300 or stored['cache_reuse_nfe_sum']<=0 or stored['repaired_prefix_verified_count']!=stored['repair_count']):
        raise ValueError('Cache verification failed')
    return dict(condition=c,status='done',raw_sha256=sha(path/'raw.jsonl'),n_done=100,
                asr_k=s['asr_k'],inference_seconds=s['wall_ms_sum']/1000,total_nfe=s['total_nfe_sum'])


def generate_all(root,state,save,launch,verify=validate_run):
    for c in ORDER:
        p=root/f'with_{c}';state.update(stage='generation',current_condition=c)
        if (p/'status.json').exists() and json.loads((p/'status.json').read_text()).get('status')=='done':
            result=verify(p,c)
            old_hash=state['conditions'].get(c,{}).get('raw_sha256')
            if old_hash and old_hash!=result['raw_sha256']:raise ValueError('Completed response source changed')
            state['conditions'][c]=result;save();print('Skip verified complete With '+c,flush=True);continue
        if p.exists():
            if any(p.iterdir()):
                archived=root/'failed_attempts'/f'with_{c}_{stamp()}'
                archived.parent.mkdir(exist_ok=True);p.rename(archived)
                state.setdefault('archived_attempts',[]).append(str(archived))
            else:p.rmdir()
        state['conditions'][c]=dict(status='running',n_done=0);save()
        print('Start With '+c+': 100 items',flush=True)
        try:
            launch(c,p)
            state['conditions'][c]=verify(p,c);save()
        except BaseException:
            state['conditions'][c]['status']='failed';save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--generation-only',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    a=parser.parse_args();root=a.output_root.resolve()
    if not root.is_relative_to(REPO/'RQ2/outputs'):parser.error('Output must be inside RQ2/outputs')
    if (root/'api_policy_blocks.json').exists() and not a.generation_only:
        parser.error('Policy-blocked inputs must not be replayed. Continue unblocked judging with: '
                     'python -m RQ2.resume_with_full_judging '+str(root))
    info=preflight(root,a.generation_only)
    if a.preflight_only:
        print(json.dumps(dict(preflight='passed',output_root=str(root),**info),indent=2));return
    lock=(REPO/'RQ2/outputs/.with_full.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root.mkdir(parents=True,exist_ok=True)
    state_path=root/'batch_status.json';manifest=root/'manifest.json'
    if state_path.exists():
        if not a.resume:parser.error('Batch exists; use --resume or a fresh output root')
        if json.loads(manifest.read_text())!=info:raise ValueError('Resume source/environment/settings differ; use a new batch path')
        shutil.copyfile(state_path,root/f'batch_before_resume_{stamp()}.json')
        state=json.loads(state_path.read_text())
    else:
        if any(root.iterdir()):raise ValueError('Output directory is not empty')
        write_json(manifest,info)
        for name in SOURCES:
            dest=root/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,dest)
        shutil.copyfile(COMBINED,root/'source_snapshot/combined.py')
        state=dict(status='running',stage='generation',started_utc=datetime.now(timezone.utc).isoformat(),
                   conditions={c:dict(status='pending',n_done=0) for c in ORDER})
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(state_path,state)
    def check_sources():
        if any(sha(REPO/k)!=v for k,v in info['source_sha256'].items()) or sha(COMBINED)!=info['combined_sha256']:
            raise ValueError('Source changed while batch was running')
    def launch(c,p):
        check_sources()
        command=[sys.executable,'-u','-m','RQ2.run_asr_with0_pilot','--accel',c,'--limit','100',
                 '--combined-path',str(COMBINED),'--output-dir',str(p)]
        with (root/f'with_{c}.log').open('a') as log:
            subprocess.run(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        state.update(status='running',pid=os.getpid());state.pop('error_type',None);save()
        generate_all(root,state,save,launch)
        summary=[state['conditions'][c] for c in ORDER]
        write_json(root/'generation_summary.json',dict(n_per_condition=100,rows=summary))
        with (root/'generation_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        results=REPO/'RQ2/results';results.mkdir(exist_ok=True)
        for ext in ['json','csv']:shutil.copyfile(root/f'generation_summary.{ext}',results/f'{root.name}_generation.{ext}')
        if not a.generation_only:
            check_sources();state.update(stage='judging',current_condition=None);save()
            with (root/'judging.log').open('a') as log:
                subprocess.run([sys.executable,'-u','-m','RQ2.judge_with_full',str(root)],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            for ext in ['json','csv']:shutil.copyfile(root/f'asr_e_comparison.{ext}',results/f'{root.name}_asr_e.{ext}')
        state.update(status='done',stage='done',current_condition=None,finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__)
        raise
    finally:
        save();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    main()
