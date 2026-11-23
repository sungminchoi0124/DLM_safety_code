"""Small synthetic checks; standard library only."""
import tempfile
import unittest
from pathlib import Path
from RQ1.analysis.audit_utility import auc, metrics, pair_rows, read_rows, interval, analyze, STEPS


class AuditUtilityTests(unittest.TestCase):
    def test_auc_ties_direction_and_missing_class(self):
        self.assertEqual(auc([1, 2], [False, True]), 1)
        self.assertEqual(auc([2, 1], [False, True]), 0)
        self.assertEqual(auc([1, 1], [False, True]), .5)
        self.assertIsNone(auc([1], [True]))

    def test_four_transitions(self):
        pairs = [({'flexible':a}, {'flexible':b, 'score':.5})
                 for a,b in ((True,True),(True,False),(False,True),(False,False))]
        r=metrics(pairs,'flexible')
        for key in ('both_correct','correct_to_wrong','wrong_to_correct','both_wrong'):
            self.assertEqual(r[key],1)
        self.assertEqual(r['delta_pp'],0)
        self.assertEqual(r['loss_given_vanilla_correct'],.5)

    def test_pair_missing_or_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            pair_rows({}, {}, list(range(250)))
        with self.assertRaises(ValueError):
            pair_rows({}, {}, [0]*250)

    def test_duplicate_csv_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test.csv'
            row='vanilla,256,0,True,False,0,False\n'
            p.write_text('defense,steps,doc_id,correct_flexible,correct_strict,audit_score,repaired\n'+row+row)
            with self.assertRaises(ValueError):
                read_rows(p,'vanilla')

    def test_cluster_bootstrap_identical_steps(self):
        pairs=[({'flexible':True,'strict':True},
                {'flexible':i%2==0,'strict':i%2==0,'score':i/250,'repaired':True}) for i in range(250)]
        r=analyze({s:pairs for s in STEPS}, repetitions=25, seed=7)
        for m in ('flexible','strict'):
            rows=r[m]['by_steps']
            self.assertTrue(all(rows[s]==rows[256] for s in STEPS))
            self.assertEqual(r[m]['equal_step_mean_auc_ci95'],rows[256]['audit_auc_loss_ci95'])
        self.assertEqual(r['audit']['identical_across_steps'],250)
        self.assertEqual(interval([0,1]),[.025,.975])


if __name__=='__main__':
    unittest.main()
