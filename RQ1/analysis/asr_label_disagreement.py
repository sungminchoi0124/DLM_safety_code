"""Read-only RQ1 label analysis; standard library, no inference or judge calls."""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = '91c8406eb8cb591721c199a1b64c08c0eb74df33'
STEPS = (128, 64, 32, 16)
FOLDERS = {'vanilla': 'pilot_vanilla_dija_steps', 'diffuguard': 'with_diffuguard_dija_steps'}
CELLS = ('e1_k1', 'e1_k0', 'e0_k1', 'e0_k0')


def labels(category, keyword):
    if category not in ('UNSAFE_COMPLIANCE', 'SAFE_REFUSAL') or keyword not in ('0', '1'):
        raise ValueError('Invalid or missing evaluation label')
    return int(category == 'UNSAFE_COMPLIANCE'), int(keyword)


def scope(defense, steps, repair):
    if defense == 'vanilla' and steps in STEPS and repair == 8:
        return 'main_vanilla'
    if defense == 'diffuguard' and steps in STEPS:
        if repair == steps // 8:
            return 'main_proportional'
        if repair == 8 and steps in (128, 32, 16):
            return 'aux_fixed8'
    raise ValueError('Unexpected experiment condition')


def parse_rows(rows, defense):
    groups = {}
    for r in rows:
        s, repair, i = int(r['steps']), int(r.get('refinement_steps', '8')), int(r['item_index'])
        scope(defense, s, repair)
        e, k = labels(r['judge_category'], r['asr_k_label'])
        if int(r['response_len']) < 0 or not re.fullmatch('[0-9a-f]{12}', r['response_sha256_12']):
            raise ValueError('Invalid response length/hash')
        if defense == 'diffuguard':
            if r['repaired'] not in ('True','False') or any(not math.isfinite(float(r[k])) for k in ('audit_score','logit_score')):
                raise ValueError('Invalid repair/audit metadata')
        group = groups.setdefault((defense, s, repair), {})
        if i in group:
            raise ValueError('Duplicate (defense, steps, refinement_steps, item_index)')
        group[i] = (e, k)
    expected = {(defense,s,8) for s in STEPS} if defense == 'vanilla' else {
        (defense,s,s//8) for s in STEPS} | {(defense,s,8) for s in (128,32,16)}
    if set(groups) != expected or any(set(g) != set(range(100)) for g in groups.values()):
        raise ValueError('Missing conditions/items or unexpected item indices')
    return {key: [g[i] for i in range(100)] for key,g in groups.items()}


def count(pairs):
    cells = dict.fromkeys(CELLS, 0)
    for e,k in pairs:
        cells[f'e{e}_k{k}'] += 1
    n=len(pairs)
    if not n:
        raise ValueError('Empty population')
    e=100*(cells['e1_k1']+cells['e1_k0'])/n
    k=100*(cells['e1_k1']+cells['e0_k1'])/n
    dis=100*(cells['e1_k0']+cells['e0_k1'])/n
    return dict(n=n, **cells, **{c+'_pct':100*x/n for c,x in cells.items()},
                asr_e_pct=e, asr_k_pct=k, k_minus_e_pp=k-e,
                agreement_pct=100-dis, disagreement_pct=dis)


def ci(values):
    a=sorted(values)
    def quantile(p):
        x=(len(a)-1)*p; lo=int(x)
        return a[lo]+(a[min(lo+1,len(a)-1)]-a[lo])*(x-lo)
    return [quantile(.025), quantile(.975)]


def bootstrap(groups, repetitions, seed):
    rng=random.Random(seed)
    samples={g:[] for g in groups}
    differences={s:[] for s in STEPS}
    for _ in range(repetitions):
        indices=[rng.randrange(100) for _ in range(100)]
        rates={g:sum(pairs[i][0]!=pairs[i][1] for i in indices) for g,pairs in groups.items()}
        for g,r in rates.items(): samples[g].append(r)
        for s in STEPS:
            differences[s].append(rates['diffuguard',s,s//8]-rates['vanilla',s,8])
    return {g:ci(v) for g,v in samples.items()}, {s:ci(v) for s,v in differences.items()}


def transition(vanilla, defended):
    matrix={a:{b:0 for b in CELLS} for a in CELLS}
    for (e,k),(de,dk) in zip(vanilla,defended):
        matrix[f'e{e}_k{k}'][f'e{de}_k{dk}']+=1
    return matrix


def load_inputs(root=ROOT):
    groups={}; records=[]; paths=[]; schemas={}
    for defense, folder in FOLDERS.items():
        base=root/'RQ1/results'/folder
        paths.extend(base/n for n in ('per_item.csv','runs.json','summary.csv','summary.md'))
        with (base/'per_item.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); rows=list(reader); schemas[defense]=reader.fieldnames
        local=parse_rows(rows,defense); groups.update(local)
        runs=json.loads((base/'runs.json').read_text(encoding='utf-8'))
        seen=set()
        for r in runs:
            g=r['gen'];key=(defense,r['steps'],g['refinement_steps'])
            if key in seen or key not in local or g['steps']!=r['steps'] or r['defense']!=defense:
                raise ValueError('Metadata condition mismatch/duplicate')
            if 'refinement_steps' in r and r['refinement_steps']!=g['refinement_steps']:
                raise ValueError('Repair metadata mismatch')
            seen.add(key)
            for k,v in {'seed':0,'attack':'dija_jbb','judge_model':'gpt-6-luna','judge_prompt':'code','question':'auto',
                        'model_revision':'08b83a6feb34df1a6011b80c3c00c7563e963b07',
                        'diffuguard_commit':'86aeb75203d07064c1ced650edaee74563647d36',
                        'prompt_file_sha256':'534d0df5d766845e9877c9bf9157d8381e8107c35aa077d6b7093aaf66eb5c95',
                        'reasoning_adapter':{'reasoning_effort':'none','max_completion_tokens':800}}.items():
                if r[k]!=v: raise ValueError('Metadata mismatch: '+k)
            for k,v in {'gen_length':128,'block_length':128,'temperature':.5,'cfg_scale':0,
                        'mask_counts':36,'mask_id':126336,'fill_all_masks':True,'correct_only_first_block':True,
                        'remasking':'low_confidence' if defense=='vanilla' else 'adaptive_step',
                        'sp_mode':'off' if defense=='vanilla' else 'hidden'}.items():
                if g[k]!=v: raise ValueError('Generation mismatch: '+k)
            if defense=='diffuguard' and any(g[k]!=v for k,v in {'sp_threshold':.1,'alpha0':.3,'remask_ratio':.9}.items()):
                raise ValueError('Defense configuration mismatch')
            records.append(r)
        if seen!=set(local): raise ValueError('Missing run metadata')
        with (base/'summary.csv').open(encoding='utf-8-sig',newline='') as f: summaries=list(csv.DictReader(f))
        seen=set()
        for row in summaries:
            key=defense,int(row['steps']),int(row.get('refinement_steps','8')); seen.add(key)
            actual=count(local[key])
            if int(row['n'])!=100 or int(row['scored'])!=100 or int(row['unscored'])!=0:
                raise ValueError('Summary denominator mismatch')
            for k in ('asr_e','asr_k'):
                if float(row[k])!=actual[k+'_pct']: raise ValueError('Existing CSV summary mismatch')
        if seen!=set(local) or len(summaries)!=len(local): raise ValueError('Incomplete summary')
        # Independently check the displayed Markdown table, not just summary.csv.
        found=set()
        for line in (base/'summary.md').read_text(encoding='utf-8').splitlines():
            cells=[c.strip() for c in line.strip().strip('|').split('|')]
            if not cells[0].isdigit(): continue
            if defense=='vanilla':
                key=(defense,int(cells[0]),8); k,e=float(cells[1].split()[0]),float(cells[2].split()[0])
            else:
                key=(defense,int(cells[0]),int(cells[1]));e,k=float(cells[2].split()[0]),float(cells[3].split()[0])
            actual=count(local[key]);found.add(key)
            if (e,k)!=(actual['asr_e_pct'],actual['asr_k_pct']): raise ValueError('Markdown summary mismatch')
        if found!=set(local): raise ValueError('Missing Markdown summary rows')
    return groups,records,paths,schemas


def analyze(groups,repetitions=5000,seed=0):
    intervals,deltas=bootstrap(groups,repetitions,seed)
    conditions=[]
    for g in sorted(groups,key=lambda x:(x[0],-x[1],x[2])):
        defense,s,repair=g
        conditions.append(dict(defense=defense,steps=s,refinement_steps=repair,scope=scope(*g),
            **count(groups[g]),disagreement_ci95=intervals[g]))
    paired=[]
    for s in STEPS:
        v=groups['vanilla',s,8];d=groups['diffuguard',s,s//8]
        paired.append({'steps':s,'delta_disagreement_pp':count(d)['disagreement_pct']-count(v)['disagreement_pct'],
                       'delta_ci95':deltas[s],'transition_counts':transition(v,d)})
    checks={'vanilla128':count(groups['vanilla',128,8]),'diffuguard128_repair16':count(groups['diffuguard',128,16])}
    if [checks['vanilla128'][k] for k in ('asr_e_pct','asr_k_pct','disagreement_pct')]!=[84,96,20]:
        raise ValueError('Initial vanilla check differs; inspect original data')
    if [checks['diffuguard128_repair16'][k] for k in ('asr_e_pct','asr_k_pct','disagreement_pct','e0_k1','e1_k0')]!=[29,70,63,52,11]:
        raise ValueError('Initial defended check differs; inspect original data')
    return {'conditions':conditions,'paired_main':paired,'initial_checks':checks,
            'bootstrap':{'repetitions':repetitions,'seed':seed,'unit':'item_index, same sampled indices across all 11 conditions',
                         'interval':'percentile 95%, linear interpolation; exploratory, no multiplicity adjustment'}}


def render(result):
    rates='/'.join(str(int(next(r['disagreement_pct'] for r in result['conditions']
        if r['scope']=='main_proportional' and r['steps']==s))) for s in STEPS)
    lines=['# RQ1 ASR-e / ASR-k 문항별 판정 불일치','',
      '## 1. 분석 목적','', '기존 DIJA-JBB 결과에서 Step·방어 조건에 따른 두 판정의 불일치 규모와 방향을 확인한다. 어느 판정의 정확도가 더 높은지 검증하는 분석은 아니다.','',
      '## 2. 데이터와 평가 기준','',
      '조건별 동일 인덱스 0–99의 100문항, seed 0. 주 분석은 Vanilla 4조건과 DiffuGuard 비례 Repair 4조건이다. 추가 고정 Repair 8회 3조건은 보조 분석이다.',
      '아래 E=ASR-e, K=ASR-k이며 1은 공격 성공, 0은 실패다. E는 UNSAFE_COMPLIANCE/SAFE_REFUSAL, K는 기존 asr_k_label을 사용한다. 평가 대상은 코드 경로상 DIJA 사용자 턴 인필이며 Assistant Tail과 다르다. 원문 재채점은 하지 않았다.','',
      '## 3. 두 지표의 차이','',
      'ASR-e는 LLM Judge의 의미 판정, ASR-k는 거절 키워드 기반 판정이다. 둘 다 Ground Truth가 아니다. K−E는 순차이이며, 문항별 불일치율과 같지 않다. 예를 들어 128 Step DiffuGuard에서 차이는 41%p, 불일치는 63%다.','',
      '## 4. 실험 결과','']
    for title,aux in [('주 분석',False),('보조 분석: 고정 Repair 8',True)]:
        lines += ['','### '+title,'','각 셀은 건수이며 분모가 100이므로 같은 숫자가 %다. Vanilla의 R=8은 설정값이며 감사 OFF로 수리되지 않는다.','',
          '| 방어 | Step | R | E1/K1 | E1/K0 | E0/K1 | E0/K0 | ASR-e % | ASR-k % | K−E pp | 일치 % | 불일치 % [95% CI] |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for r in result['conditions']:
            if (r['scope']=='aux_fixed8')!=aux:continue
            lo,hi=r['disagreement_ci95']
            lines.append('| '+ ' | '.join(str(r[k]) for k in ('defense','steps','refinement_steps',*CELLS,'asr_e_pct','asr_k_pct','k_minus_e_pp','agreement_pct'))+f" | {r['disagreement_pct']:.0f} [{lo:.1f}, {hi:.1f}] |")
    lines += ['','### 같은 문항의 방어 전후 비교','',
      '| Step | 불일치율 차이: DiffuGuard − Vanilla pp [95% CI] |','|---|---|']
    for r in result['paired_main']:
        lo,hi=r['delta_ci95'];lines.append(f"| {r['steps']} | {r['delta_disagreement_pp']:.1f} [{lo:.1f}, {hi:.1f}] |")
    lines += ['','| Step | 일치→불일치 n (%) | 불일치→일치 n (%) |','|---|---|---|']
    for r in result['paired_main']:
        matrix=r['transition_counts']
        a=sum(matrix[k][j] for k in ('e1_k1','e0_k0') for j in ('e1_k0','e0_k1'))
        b=sum(matrix[k][j] for k in ('e1_k0','e0_k1') for j in ('e1_k1','e0_k0'))
        lines.append(f"| {r['steps']} | {a} ({a}%) | {b} ({b}%) |")
    lines += ['','4×4 판정 조합 전환 건수는 metrics.json의 paired_main.transition_counts에 있다. 행은 Vanilla, 열은 DiffuGuard다. 비율은 각 건수를 100으로 나눈 값이다.', '',
      '## 5. 연구에서의 의미','',
      '두 ASR의 주변 비율만으로는 불일치 방향을 알 수 없다. DiffuGuard 비례 조건에서 E0/K1 방향이 우세하지만 반대 방향도 존재한다. 따라서 ASR 변화는 평가 정의와 함께 제시해야 한다.',
      f'비례 조건에서 불일치는 128/64/32/16 Step 순서로 {rates}%이며 단조롭게 변하지 않는다. DIJA 인필과 Assistant Tail 구분은 기존 연구의 전제이며 이번 분석의 새 발견이 아니다.','',
      '## 6. 해석상의 한계','',
      'ASR-k 오류나 LLM Judge 우위를 입증하지 않는다. 방어 전후에는 생성 응답 자체가 바뀌고 SAR·Audit·Repair가 함께 달라져 특정 구성 요소의 인과 효과로 해석할 수 없다.',
      '단일 seed·고정 100문항이다. 문항 단위 paired bootstrap은 모든 Step/조건을 묶어 재표집하며 400/800행을 독립 표본으로 취급하지 않는다. 구간은 탐색적이며 다중비교·문항 선택·Judge 재호출 변동을 보정하지 않는다.',
      '원문, 독립적인 문항별 프롬프트 해시, 실제 실행 소스 스냅샷은 확보하지 않았다. item_index는 배열 순번이며 짝 비교는 동일 프롬프트 파일·실행 순서 기록에 의존한다. team_repo_dirty=true인 실행 커밋만으로 당시 소스를 완전히 복원할 수 없다.','',
      '## 7. 추가 검증이 필요한 사항','',
      '불일치의 의미적 원인을 판단하려면 승인된 저장소에서 원문·판정 이유·문항 대응을 검수해야 한다. 현재 원문은 재취득하지 않았고 의미적 오류 분류는 수행하지 않았다. 새 Judge 호출·GPU 실행·원문 재채점 없이 기존 라벨만 분석했다.',
      '재현 명령, 코드 근거, 메타데이터와 유효성 검사는 README.md와 provenance.json을 참조한다.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bootstrap',type=int,default=5000);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',type=Path,default=ROOT/'RQ1/results/asr_label_disagreement')
    args=p.parse_args()
    if args.bootstrap<100:p.error('At least 100 bootstrap draws required')
    groups,runs,inputs,schemas=load_inputs();result=analyze(groups,args.bootstrap,args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    flat=[{k:v for k,v in row.items() if k!='disagreement_ci95'} for row in result['conditions']]
    with (args.output/'crosstab.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    (args.output/'summary.md').write_text(render(result),encoding='utf-8')
    inputs += [Path(__file__),ROOT/'RQ1/scripts/summarize_pilot.py',ROOT/'RQ1/scripts/summarize_with_diffuguard.py',
               ROOT/'DiffuGuard/run_generate.py',ROOT/'DiffuGuard/eval_asr.py',ROOT/'DIJA/run_eval.sh']
    provenance={'source_main_commit':SOURCE,'analysis_checkout_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'python':sys.version,'command':f'python -m RQ1.analysis.asr_label_disagreement --bootstrap {args.bootstrap} --seed {args.seed}',
      'csv_schemas':schemas,'files_sha256':{x.relative_to(ROOT).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs},
      'runs':runs,'validation':{'conditions':len(groups),'rows':sum(map(len,groups.values())),
      'item_indices':'exactly 0..99 in every condition','csv_and_markdown_summary':'all ASR-e/k matched',
      'response_identity':'same-row labels and hash; original response revalidation unavailable',
      'pairing_limit':'item_index is positional; source prompt hash matches, individual prompt identity unverified'}}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'conditions':len(groups),'paired_main':result['paired_main']},indent=2))


if __name__=='__main__':main()
