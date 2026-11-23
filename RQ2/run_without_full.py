"""Sequential Without 0/A or B/C full runs with validation and numeric reports.

Run from repository root using python -m RQ2.run_without_full.
All raw responses and process logs stay in Git-ignored outputs.
"""
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .data import REPO, PROMPT, PROMPT_SHA, load_items
from .run_asr import write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', nargs=2, choices=['0','A','B','C'], default=['0','A'])
    args = parser.parse_args()
    conditions = tuple(args.conditions)
    if conditions not in (('0','A'), ('B','C')):
        parser.error('Supported sequential pairs are 0 A and B C')
    batch_name = 'without_' + '_'.join(conditions)
    out = REPO / 'RQ2/outputs/full'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.without_full.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    destinations = {c: out / f'without_{c}_seed0_n100' for c in conditions}
    for p in destinations.values():
        if p.exists() and any(p.iterdir()):
            raise ValueError(f'Results already exist: {p}; refusing duplicate batch')
    assert len(load_items(REPO / 'RQ2/data/ids/dija_100.txt')) == 100
    assert hashlib.sha256(PROMPT.read_bytes()).hexdigest() == PROMPT_SHA
    protocol_path = REPO / ('RQ2/configs/asr_protocol_B_C.json' if conditions == ('B','C') else 'RQ2/configs/asr_protocol.json')
    protocol = json.loads(protocol_path.read_text())
    assert protocol['status'] == f'adopted_by_C_for_Without_{conditions[0]}_{conditions[1]}_full_runs'
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if int(gpu.strip().splitlines()[0]) > 1000:
        raise RuntimeError('GPU already occupied; refusing concurrent experiment')
    state = {'status': 'running', 'pid': os.getpid(), 'started_utc': now(),
             'protocol_id': protocol['protocol_id'], 'asr_e': 'not_run',
             'conditions': {c: {'status': 'pending', 'path': str(p)} for c, p in destinations.items()}}
    state_path = out / f'{batch_name}_batch_status.json'
    write_json(state_path, state)
    reports = {}
    try:
        for c, p in destinations.items():
            state['current_condition'] = c
            state['conditions'][c].update(status='running', started_utc=now())
            write_json(state_path, state)
            command = [sys.executable, '-m', 'RQ2.run_asr', '--accel', c, '--limit', '100',
                       '--verify-original', '--output-dir', str(p)]
            state['conditions'][c]['command'] = command
            write_json(state_path, state)
            print(f'{now()} starting Without {c}: 100 items', flush=True)
            log = REPO / f'RQ2/outputs/logs/without_{c}_seed0_n100.log'
            with log.open('x') as stream:
                subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
            print(f'{now()} validating/scoring Without {c}', flush=True)
            with log.open('a') as stream:
                subprocess.run([sys.executable, '-m', 'RQ2.evaluate_asr', str(p)], cwd=REPO,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((p / 'summary_local.json').read_text())
            metadata = json.loads((p / 'config_resolved.json').read_text())
            raw = [json.loads(s) for s in (p / 'raw.jsonl').read_text().splitlines()]
            pilot = REPO / f'RQ2/outputs/pilot/without_{c}_seed0_limit10/raw.jsonl'
            previous = [json.loads(s) for s in pilot.read_text().splitlines()] if pilot.exists() else None
            matched = sum(a['output_sha256'] == b['output_sha256'] for a, b in zip(raw[:10], previous)) if previous is not None else None
            reports[c] = {'condition': c, 'run_dir': str(p), 'summary': summary,
                          'generation_seconds': metadata['generation_seconds'],
                          'model_load_seconds': metadata['model_load_seconds'],
                          'process_seconds': metadata['process_seconds'],
                          'reference_check': json.loads((p / 'reference_check.json').read_text()),
                          'pilot_first_10_output_hash_matches': matched,
                          'git_commit': metadata['git_commit'], 'git_dirty': metadata['git_dirty'],
                          'protocol': metadata['protocol'], 'protocol_sha256': metadata['protocol_sha256'],
                          'model_revision': metadata['model_revision'], 'prompt_sha256': metadata['prompt_sha256'],
                          'ids_sha256': metadata['ids_sha256'], 'command': metadata['command']}
            write_json(REPO / f'RQ2/results/without_{c}_full.json', reports[c])
            state['conditions'][c].update(status='done', finished_utc=now(),
                                          n_items=summary['n'], asr_k=summary['asr_k'],
                                          total_nfe=summary['total_nfe_sum'], wall_ms=summary['wall_ms_sum'])
            write_json(state_path, state)
        first, second = (reports[c]['summary'] for c in conditions)
        assert first['item_ids'] == second['item_ids']
        for k in ('model_revision', 'prompt_sha256', 'ids_sha256', 'protocol_sha256'):
            assert reports[conditions[0]][k] == reports[conditions[1]][k], k
        comparison = {'n': 100, 'scope': f'Without {conditions[0]} vs Without {conditions[1]} full ASR; ASR-e deferred',
                      **{f'without_{c}': reports[c] for c in conditions},
                      f'wall_speedup_{conditions[0]}_over_{conditions[1]}': first['wall_ms_sum'] / second['wall_ms_sum'],
                      'wall_reduction_percent': 100 * (1 - second['wall_ms_sum'] / first['wall_ms_sum']),
                      'asr_k_change_percentage_points': second['asr_k'] - first['asr_k']}
        comparison_path = REPO / f'RQ2/results/without_{conditions[0]}_vs_{conditions[1]}_full.json'
        write_json(comparison_path, comparison)
        if conditions == ('B','C'):
            four = {c: json.loads((REPO / f'RQ2/results/without_{c}_full.json').read_text()) for c in ['0','A','B','C']}
            for c in four:
                for key in ['model_revision','prompt_sha256','ids_sha256']:
                    assert four[c][key] == four['0'][key], key
                assert four[c]['summary']['item_ids'] == four['0']['summary']['item_ids']
            write_json(REPO / 'RQ2/results/without_0_A_B_C_full.json', {
                'n_per_condition': 100, 'asr_e': 'not_run', 'same_ids_and_order': True,
                'rows': [{'accel':c, 'threshold':four[c]['reference_check']['config']['threshold'],
                          'asr_k':four[c]['summary']['asr_k'], 'wall_ms_sum':four[c]['summary']['wall_ms_sum'],
                          'total_nfe':four[c]['summary']['total_nfe_sum'],
                          'max_remaining_masks':four[c]['summary']['max_remaining_masks'],
                          'empty_response_rate':four[c]['summary']['empty_response_rate'],
                          'speedup_vs_0':four['0']['summary']['wall_ms_sum']/four[c]['summary']['wall_ms_sum']}
                         for c in ['0','A','B','C']]})
        state.update(status='done', finished_utc=now(), current_condition=None, comparison_report=str(comparison_path))
        print(f'{now()} both full runs completed and validated', flush=True)
    except BaseException as exc:
        current = state.get('current_condition')
        if current:
            state['conditions'][current].update(status='failed', error_type=type(exc).__name__)
        state.update(status='failed', error_type=type(exc).__name__, finished_utc=now())
        raise
    finally:
        write_json(state_path, state)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
