"""Judge six completed 100-item With runs; reuse existing Without verdicts."""
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
from DiffuGuard.eval_asr import load_env
from .data import REPO
from .evaluate_asr import summarize
from .run_asr import write_json


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if not run_root.is_relative_to(REPO/'RQ2/outputs'):
        parser.error('run_root must be inside RQ2/outputs')
    if (run_root/'api_policy_blocks.json').exists():
        parser.error('Do not replay recorded policy blocks. Use RQ2.resume_with_full_judging instead.')
    load_env()
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY unavailable; configure locally, never print the key')
    root = REPO/'RQ2/outputs/asr_e'
    root.mkdir(exist_ok=True)
    lock = (root/'.judge.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    conditions = ['0','A','B','C','D','E']
    state_path = run_root/'asr_e_status.json'
    state = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 models={}, scope='With six full runs, 100 items each; reuse existing Without judgments')
    write_json(state_path, state)
    rows_out = []
    source_hashes = {}
    try:
        for config_name in ['asr_e_judge.json','asr_e_judge_gpt4omini.json']:
            cfg = json.loads((REPO/'RQ2/configs'/config_name).read_text())
            settings = {k:cfg[k] for k in ['judge_model','judge_prompt','question','reasoning_effort','max_completion_tokens']}
            model = settings['judge_model']
            tag = hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:12]
            provenance = dict(settings, base_url=os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1',
                              reasoning_override=os.environ.get('JUDGE_REASONING',''),
                              evaluation_source_sha256=sha(REPO/'DiffuGuard/third_party/DiffuGuard/evaluation.py'))
            state.update(current_model=model,current_condition=None)
            state['models'][model] = dict(status='running',tag=tag,conditions={})
            write_json(state_path,state)
            for c in conditions:
                state['current_condition']=c
                p=run_root/f'with_{c}'
                raw_path=p/'raw.jsonl';raw=[json.loads(x) for x in raw_path.read_text().splitlines() if x.strip()]
                assert [x['item_id'] for x in raw]==[f'dija-{i:03d}' for i in range(100)]
                assert all(x['n_masks']==0 and x['total_nfe']==x['measured_forward_calls'] for x in raw)
                assert json.loads((p/'status.json').read_text())['status']=='done'
                gen=p/'generations_for_judge.json';data=json.loads(gen.read_text())
                assert len(data)==100 and all(a['item_id']==b['item_id'] and a['response']==b['output_text'] and a['output_sha256']==b['output_sha256'] for a,b in zip(data,raw))
                source_hashes[str(raw_path)]=sha(raw_path)
                without=REPO/f'RQ2/outputs/full/without_{c}_seed0_n100'
                prior_marker=json.loads((without/f'eval_{tag}.json.run_config.json').read_text())
                if any(prior_marker[k]!=v for k,v in provenance.items()):
                    raise ValueError('Judge settings/endpoint/source differ from saved Without labels')
                prior=without/f'eval_{tag}.json';source_hashes[str(prior)]=sha(prior)
                wraw_path=without/'raw.jsonl';source_hashes[str(wraw_path)]=sha(wraw_path)
                target=p/f'eval_{tag}.json';marker=Path(str(target)+'.run_config.json')
                stamp=dict(provenance,source_raw_sha256=sha(raw_path),source_generations_sha256=sha(gen))
                if marker.exists():
                    if json.loads(marker.read_text())!=stamp:raise ValueError('Incompatible existing With judgments')
                elif target.exists():raise ValueError('Existing With judgment has no provenance')
                write_json(marker,stamp)
                state['models'][model]['conditions'][c]=dict(status='running');write_json(state_path,state)
                print(f'Judging With {c}: model={model} n=100',flush=True)
                env=dict(os.environ,JUDGE_MODEL=model,JUDGE_REASONING_EFFORT=settings['reasoning_effort'],JUDGE_MAX_COMPLETION_TOKENS=str(settings['max_completion_tokens']))
                cmd=[sys.executable,str(REPO/'DiffuGuard/eval_asr.py'),str(gen),'--output',str(target),'--judge_prompt',settings['judge_prompt'],'--question',settings['question'],'--workers',str(cfg['workers'])]
                with (run_root/f'judge_{tag}_{c}.log').open('a') as log:
                    subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                judged=json.loads(target.read_text());summary=summarize(raw,judged)
                summary.update(judge_model=model,settings=settings,item_ids=[x['item_id'] for x in raw])
                write_json(p/f'summary_{tag}.json',summary)
                wraw=[json.loads(x) for x in wraw_path.read_text().splitlines() if x.strip()]
                ids={x['item_id'] for x in wraw}
                wjudged=[x for x in json.loads(prior.read_text()) if x['item_id'] in ids]
                ws=summarize(wraw,wjudged)
                record=dict(judge_model=model,condition=c,n=100,without_asr_k=ws['asr_k'],with_asr_k=summary['asr_k'],without_asr_e=ws['asr_e'],with_asr_e=summary['asr_e'],reduction_pp=ws['asr_e']-summary['asr_e'],without_safe=ws['safe'],without_unsafe=ws['unsafe'],with_safe=summary['safe'],with_unsafe=summary['unsafe'],with_unscored=summary['unscored'])
                rows_out.append(record)
                state['models'][model]['conditions'][c]=dict(status='done' if summary['unscored']==0 else 'partial',safe=summary['safe'],unsafe=summary['unsafe'],unscored=summary['unscored'])
                write_json(state_path,state)
                write_json(run_root/'judge_partial.json',dict(rows=rows_out,source_sha256=source_hashes))
                if summary['unscored']:raise RuntimeError('Unscored judgments remain; stop and resume unchanged input/config')
            state['models'][model]['status']='done';write_json(state_path,state)
        for path,value in source_hashes.items():
            if sha(Path(path))!=value:raise ValueError('Stored generation or original Without verdict changed')
        result=dict(status='done',n_with_judgments=1200,rows=rows_out,source_sha256=source_hashes,
                    interpretation='Paired same 100 IDs/order; models and prompts match Without. Conditions are judged independently, including identical texts.')
        write_json(run_root/'asr_e_comparison.json',result)
        with (run_root/'asr_e_comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows_out[0]));writer.writeheader();writer.writerows(rows_out)
        state.update(status='done',current_model=None,current_condition=None)
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__)
        raise
    finally:
        state['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(state_path,state)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    main()
