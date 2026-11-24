"""Synthetic and saved-data checks, no model or API dependency."""
import unittest
from RQ1.analysis.asr_label_disagreement import labels, scope, parse_rows, count, bootstrap, transition, load_inputs, analyze, render, STEPS


class DisagreementTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(labels('UNSAFE_COMPLIANCE','0'),(1,0))
        self.assertEqual(labels('SAFE_REFUSAL','1'),(0,1))
        for c,k in [('UNKNOWN','1'),('SAFE_REFUSAL','2'),('','0')]:
            with self.assertRaises(ValueError):labels(c,k)

    def test_cells_and_direction(self):
        r=count([(1,1),(1,0),(0,1),(0,1),(0,0)])
        self.assertEqual(sum(r[k] for k in ('e1_k1','e1_k0','e0_k1','e0_k0')),5)
        self.assertEqual((r['asr_e_pct'],r['asr_k_pct'],r['k_minus_e_pp'],r['disagreement_pct']),(40,60,20,60))

    def test_repair_scope(self):
        for s in STEPS:self.assertEqual(scope('diffuguard',s,s//8),'main_proportional')
        for s in (128,32,16):self.assertEqual(scope('diffuguard',s,8),'aux_fixed8')
        with self.assertRaises(ValueError):scope('diffuguard',32,2)

    def test_missing_and_duplicate(self):
        row={'steps':'128','item_index':'0','asr_k_label':'1','judge_category':'SAFE_REFUSAL',
             'response_len':'1','response_sha256_12':'a'*12}
        with self.assertRaisesRegex(ValueError,'Duplicate'):parse_rows([row,row],'vanilla')
        with self.assertRaisesRegex(ValueError,'Missing'):parse_rows([row],'vanilla')

    def test_joint_bootstrap(self):
        pairs=[(i%2,0) for i in range(100)]
        groups={(d,s,8 if d=='vanilla' else s//8):pairs for d in ('vanilla','diffuguard') for s in STEPS}
        intervals,deltas=bootstrap(groups,100,3)
        self.assertTrue(all(x==[0,0] for x in deltas.values()))
        self.assertTrue(all(x==next(iter(intervals.values())) for x in intervals.values()))

    def test_transition_matrix(self):
        m=transition([(1,1),(0,1)],[(0,1),(0,0)])
        self.assertEqual(m['e1_k1']['e0_k1'],1)
        self.assertEqual(m['e0_k1']['e0_k0'],1)
        self.assertEqual(sum(sum(row.values()) for row in m.values()),2)

    def test_saved_inputs_metadata_summaries_and_initial_checks(self):
        groups,_,_,_=load_inputs()
        self.assertEqual(len(groups),11)
        v=count(groups['vanilla',128,8]);d=count(groups['diffuguard',128,16])
        self.assertEqual((v['asr_e_pct'],v['asr_k_pct'],v['disagreement_pct']),(84,96,20))
        self.assertEqual((d['asr_e_pct'],d['asr_k_pct'],d['disagreement_pct'],d['e0_k1'],d['e1_k0']), (29,70,63,52,11))

    def test_render_uses_computed_rates(self):
        groups,_,_,_=load_inputs()
        result=analyze(groups,100,0)
        rates='/'.join(str(int(count(groups['diffuguard',s,s//8])['disagreement_pct'])) for s in STEPS)
        self.assertIn(rates+'%',render(result))
        for r in result['paired_main']:
            self.assertEqual(sum(sum(row.values()) for row in r['transition_counts'].values()),100)


if __name__=='__main__':unittest.main()
