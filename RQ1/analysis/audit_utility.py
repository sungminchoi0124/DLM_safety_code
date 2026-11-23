"""Saved RQ1 numeric results only; no model imports or inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
STEPS = (256, 128, 64, 32)


def boolean(value):
    if value not in ('True', 'False'):
        raise ValueError('Invalid boolean')
    return value == 'True'


def read_rows(path, defense):
    rows = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for r in csv.DictReader(stream):
            key = int(r['steps']), int(r['doc_id'])
            if key in rows or r['defense'] != defense or key[0] not in STEPS:
                raise ValueError('Duplicate key or unexpected condition')
            score = float(r['audit_score'])
            if not math.isfinite(score):
                raise ValueError('Nonfinite audit score')
            rows[key] = {m: boolean(r['correct_' + m]) for m in ('flexible', 'strict')}
            rows[key].update(score=score, repaired=boolean(r['repaired']))
    return rows


def pair_rows(vanilla, defended, ids):
    expected = {(s, d) for s in STEPS for d in ids}
    if len(ids) != 250 or len(set(ids)) != 250 or set(vanilla) != expected or set(defended) != expected:
        raise ValueError('Require identical complete 250 x 4 key sets')
    return {s: [(vanilla[s, d], defended[s, d]) for d in ids] for s in STEPS}


def auc(scores, labels):
    """P(score_loss > score_retained) + half ties; no fitted threshold."""
    n1 = sum(labels)
    n0 = len(labels) - n1
    if not n0 or not n1:
        return None
    ordered = sorted(zip(scores, labels))
    wins = 0.0
    negatives = 0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        positive = sum(y for _, y in ordered[i:j])
        negative = j - i - positive
        wins += positive * (negatives + negative / 2)
        negatives += negative
        i = j
    return wins / (n1 * n0)


def metrics(pairs, metric):
    cells = [0, 0, 0, 0]  # retained correct, loss, gain, retained wrong
    scores, losses = [], []
    for v, d in pairs:
        a, b = v[metric], d[metric]
        cells[0 if a and b else 1 if a else 2 if b else 3] += 1
        if a:
            scores.append(d['score'])
            losses.append(not b)
    keep, loss, gain, wrong = cells
    n = len(pairs)
    return dict(n=n, both_correct=keep, correct_to_wrong=loss, wrong_to_correct=gain,
                both_wrong=wrong, both_correct_rate=keep/n, correct_to_wrong_rate=loss/n,
                wrong_to_correct_rate=gain/n, both_wrong_rate=wrong/n,
                vanilla_accuracy=(keep+loss)/n, diffuguard_accuracy=(keep+gain)/n,
                delta_pp=100*(gain-loss)/n, vanilla_correct_n=keep+loss,
                loss_given_vanilla_correct=loss/(keep+loss) if keep+loss else None,
                audit_auc_loss=auc(scores, losses))


def interval(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    def q(p):
        x = (len(values)-1)*p
        lo = int(x)
        return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(x-lo)
    return [q(.025), q(.975)]


def analyze(paired, repetitions=2000, seed=0):
    result = {}
    # Resample doc_id positions once per draw, retaining all four steps together.
    for metric in ('flexible', 'strict'):
        point = {s: metrics(paired[s], metric) for s in STEPS}
        samples = {s: {'delta_pp': [], 'audit_auc_loss': []} for s in STEPS}
        average_auc = []
        rng = random.Random(seed)
        for _ in range(repetitions):
            indices = [rng.randrange(250) for _ in range(250)]
            draws = {s: metrics([paired[s][i] for i in indices], metric) for s in STEPS}
            for s in STEPS:
                for key in samples[s]:
                    samples[s][key].append(draws[s][key])
            a = [draws[s]['audit_auc_loss'] for s in STEPS]
            if all(x is not None for x in a):
                average_auc.append(sum(a)/4)
        for s in STEPS:
            for key, values in samples[s].items():
                point[s][key+'_ci95'] = interval(values)
                point[s][key+'_valid_bootstrap'] = sum(v is not None for v in values)
        observed = [point[s]['audit_auc_loss'] for s in STEPS]
        result[metric] = {'by_steps': point,
                          'equal_step_mean_auc': sum(observed)/4 if all(x is not None for x in observed) else None,
                          'equal_step_mean_auc_ci95': interval(average_auc),
                          'valid_mean_auc_bootstrap': len(average_auc)}
    scores = [d['score'] for _, d in paired[256]]
    spreads = [max(paired[s][i][1]['score'] for s in STEPS)-min(paired[s][i][1]['score'] for s in STEPS) for i in range(250)]
    result['audit'] = dict(unique_doc_ids=250, rows=1000,
        identical_across_steps=sum(x == 0 for x in spreads), max_within_doc_range=max(spreads),
        min=min(scores), max=max(scores), mean=sum(scores)/250,
        repaired_rows=sum(d['repaired'] for pairs in paired.values() for _, d in pairs))
    return result


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def markdown(result):
    lines = ['# RQ1 Audit Score와 GSM8K Utility: 탐색적 집계', '',
             '기존 CSV의 정답 여부를 `(steps, doc_id)`로 짝지었다. 응답 원문 재채점은 수행하지 않았다.',
             '정답 전환 비율의 분모는 각 Step의 전체 250문항이다. 조건부 손실률의 분모는 Vanilla 정답 문항이다.', '']
    for m in ('flexible', 'strict'):
        lines += ['## '+m, '',
          '| Steps | 정답 유지 n (%) | 정답→오답 n (%) | 오답→정답 n (%) | 오답 유지 n (%) | Vanilla % | DiffuGuard % | 차이 pp [95% CI] | 조건부 손실률 % | Audit AUC [95% CI] |',
          '|---|---|---|---|---|---|---|---|---|---|']
        for s,x in result[m]['by_steps'].items():
            cells = [f"{x[k]} ({x[k]/2.5:.1f})" for k in ('both_correct','correct_to_wrong','wrong_to_correct','both_wrong')]
            ci=x['delta_pp_ci95']; ac=x['audit_auc_loss_ci95']
            lines.append('| '+str(s)+' | '+' | '.join(cells)+
              f" | {100*x['vanilla_accuracy']:.1f} | {100*x['diffuguard_accuracy']:.1f} | {x['delta_pp']:.1f} [{ci[0]:.2f}, {ci[1]:.2f}] | {100*x['loss_given_vanilla_correct']:.2f} | {x['audit_auc_loss']:.3f} [{ac[0]:.3f}, {ac[1]:.3f}] |")
        ci=result[m]['equal_step_mean_auc_ci95']
        lines += ['',f"Step별 AUC의 동일 가중 평균: {result[m]['equal_step_mean_auc']:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}].",
                  f"평균 AUC의 유효 bootstrap 횟수: {result[m]['valid_mean_auc_bootstrap']}/{result['method']['bootstrap_repetitions']}. 한 클래스가 없는 재표집은 제외했다.",'']
    a=result['audit']
    lines += ['## Audit Score 반복과 수리', '',
      f"- 4개 Step에서 점수가 정확히 동일한 문항: {a['identical_across_steps']}/250. 문항 내 최대 범위: {a['max_within_doc_range']}.",
      f"- 고유 250문항 점수 최솟값/평균/최댓값: {a['min']:.9f} / {a['mean']:.9f} / {a['max']:.9f}.",
      f"- 수리 기록: {a['repaired_rows']}/1000. 기존 감사 기준값은 0.1.", '',
      '## 해석 범위', '',
      '- AUC는 Vanilla 정답 문항에서 높은 점수가 정답→오답을 구분하는 방향이다. 0.5는 순위 구분력 없음의 기준이며, 학습한 예측 모델의 성능이 아니다.',
      '- flexible의 평균 AUC 구간은 0.5를 포함한다. 높은 Audit Score가 손실 위험을 일관되게 구분한다는 증거를 이 자료에서 확인하지 못했다. 무관함을 입증한 것은 아니다.',
      '- 문항 단위 bootstrap에서 4개 Step을 함께 재표집했다. 1000행을 독립 표본으로 취급하지 않았다. CI는 탐색적 percentile 구간이며 다중비교 보정이나 새로운 문항·seed에 대한 검증을 대신하지 않는다.',
      '- strict는 출력 형식의 영향을 받으므로 flexible과 분리했다. strict 32 Step의 정답 유지 표본은 특히 작아 AUC 불확실성이 크다.',
      '- 전 문항이 수리되어 미수리 대조군이 없다. SAR, 생성 예산, 비례 수리 예산의 효과를 분리하거나 Audit Score의 인과 효과를 추정할 수 없다.',
      '- 코드 근거, 재현 명령, 입력 해시와 실행 커밋은 README.md 및 provenance.json을 참조한다.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'RQ1/results/audit_utility_exploratory')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error('Use at least 100 bootstrap replicates')
    folder = ROOT/'RQ1/results/gsm8k_250'
    inputs = [folder/n for n in ('per_item.csv', 'diffuguard_per_item.csv', 'runs.json', 'diffuguard_runs.json', 'diffuguard_summary.md')]
    idpath = ROOT/'RQ1/data/ids/gsm8k_250.txt'
    with idpath.open() as f:
        ids = [int(r['source_key']) for r in csv.DictReader(f, delimiter='\t')]
    v, d = read_rows(inputs[0], 'vanilla'), read_rows(inputs[1], 'diffuguard')
    paired = pair_rows(v, d, ids)
    runs = [json.loads(p.read_text()) for p in inputs[2:4]]
    shared = ('gen_length', 'block_length', 'temperature', 'cfg_scale', 'mask_id', 'fill_all_masks')
    run_info = []
    for defense, records in zip(('vanilla', 'diffuguard'), runs):
        if len(records) != 4 or {r['gen']['steps'] for r in records} != set(STEPS):
            raise ValueError('Missing run metadata')
        for r in records:
            if r['seed'] != 0 or r['defense'] != defense or r['decoder'] != 'semi_ar':
                raise ValueError('Unexpected run metadata')
            run_info.append({k: r.get(k) for k in ('defense', 'team_repo_commit', 'diffuguard_commit', 'model_revision', 'torch_version', 'transformers_version', 'seed')} | {'gen': r['gen']})
    for s in STEPS:
        a,b = [next(r for r in rs if r['gen']['steps']==s) for rs in runs]
        for key in ('model_revision', 'diffuguard_commit', 'prompt_file_sha256', 'seed'):
            if a[key] != b[key]:
                raise ValueError('Paired metadata mismatch: '+key)
        if any(a['gen'][k] != b['gen'][k] for k in shared):
            raise ValueError('Generation setting mismatch')
    result = analyze(paired, args.bootstrap, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    result['method'] = {'bootstrap_repetitions': args.bootstrap, 'bootstrap_seed': args.seed,
        'unit': 'doc_id; same resampled indices for all four steps',
        'ci': 'percentile 95%; exploratory, no multiplicity correction',
        'auc_population': 'vanilla correct within each step; positive=DiffuGuard wrong',
        'aggregate_auc': 'equal-weight mean of four within-step AUCs; not pooled 1000 independent rows'}
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (args.output/'summary.md').write_text(markdown(result), encoding='utf-8')
    flat = [dict(metric=m, steps=s, **{k:v for k,v in r.items() if not isinstance(v,list)}) for m in ('flexible','strict') for s,r in result[m]['by_steps'].items()]
    with (args.output/'transitions.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    tracked = inputs+[idpath, Path(__file__), ROOT/'DiffuGuard/run_gsm8k.py', ROOT/'DiffuGuard/semi_ar.py', ROOT/'RQ1/scripts/summarize_gsm8k.py']
    provenance = {'source_main_commit': 'abd562d717550caca9417ded3a1a7538963638b1',
        'input_tree_oid': git('rev-parse','HEAD:RQ1/results/gsm8k_250'),
        'analysis_checkout_commit':git('rev-parse','HEAD'), 'python':sys.version,
        'command':f'python -m RQ1.analysis.audit_utility --bootstrap {args.bootstrap} --seed {args.seed}',
        'files_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
        'runs':run_info}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
