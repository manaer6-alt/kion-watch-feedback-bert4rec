"""Small scientific-contract checks: no downloads and no GPU required."""
import ast
import hashlib
import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import kion_watch_feedback as s

ROOT = Path(__file__).resolve().parents[1]


class ScientificContract(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(s.WatchConfig(), technical_mode=True, n_users=4,
                           min_history=2, hidden_size=8, layers=1)

    def test_frozen_code_equals_notebook(self):
        code = (ROOT / 'kion_watch_feedback.py').read_text(encoding='utf-8')
        self.assertEqual(hashlib.sha256(code.encode()).hexdigest(),
                         '5ab07ac369ae3966f3d781ada94a38415eed710ade65541c594c2a4fd3b57509')
        notebook = json.loads((ROOT / 'notebooks/KION_WatchFeedback_5methods_Kaggle.ipynb').read_text(encoding='utf-8'))
        cell = next(c for c in notebook['cells'] if c.get('id') == 'implementation')
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        embedded = next(ast.literal_eval(n.value) for n in ast.parse(source).body
                        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROGRAM_SOURCE' for t in n.targets))
        self.assertEqual(code, embedded)

    def test_analytic_loss_and_ignored_gradient(self):
        logits = torch.tensor([[0., 0.], [0., math.log(3)]], requires_grad=True)
        labels = torch.tensor([0, 1])
        weights = torch.tensor([.24, .92])
        actual = s.weighted_loss(logits, labels, weights)
        self.assertAlmostEqual(actual.item(), (.24*math.log(2)+.92*math.log(4/3))/1.16, places=6)
        torch.testing.assert_close(actual, s.weighted_loss(logits, labels, 3*weights))
        s.weighted_loss(logits, labels, torch.tensor([0., 1.])).backward()
        self.assertTrue(logits.grad[0].eq(0).all())
        self.assertGreater(logits.grad[1].abs().sum().item(), 0)
        self.assertIsNone(s.weighted_loss(logits, labels, torch.zeros(2)))

    def test_thresholds_and_masks(self):
        q = torch.tensor([0., 10., 10.001, 100.])
        torch.testing.assert_close(s.target_weights(q, 'TARGET10', self.cfg), torch.tensor([0., 0., 1., 1.]))
        torch.testing.assert_close(s.target_weights(q, 'STEP10', self.cfg), torch.tensor([1., 1., 3., 3.]))
        torch.testing.assert_close(s.target_weights(q, 'SOFT', self.cfg), torch.tensor([.2, .28, .280008, 1.]))
        clean = torch.zeros((3, 20), dtype=torch.long)
        for i, n in enumerate([2, 10, 20]):
            clean[i, :n] = torch.arange(1, n+1)
        pct = clean.float()*4
        corrupted, mask, labels, observed = s.masked_batch(clean, pct, 99, self.cfg, 17, 9)
        self.assertTrue(mask.any(dim=1).all())
        self.assertTrue(((~mask)&clean.ne(0)).any(dim=1).all())
        self.assertFalse((mask&clean.eq(0)).any())
        self.assertTrue(corrupted[mask].eq(99).all())
        torch.testing.assert_close(labels, clean[mask]-1)
        torch.testing.assert_close(observed, pct[mask])

    def test_calendar_targets_and_future_percentage(self):
        rows = []
        for user, ids, ps in [(1, [1,2], [20.,30.]), (2, [3,4], [10.,90.]),
                              (3, [5,6], [0.,10.]), (4, [7,8], [50.,90.])]:
            rows += [[user, i, '2021-01-02', p] for i, p in zip(ids, ps)]
        rows += [[1,99,'2021-01-06',90.], [1,3,'2021-01-07',90.],
                 [2,1,'2021-01-06',np.nan], [2,2,'2021-01-07',90.],
                 [3,1,'2021-01-06',90.], [4,3,'2021-01-06',90.]]
        films = pd.DataFrame(rows, columns=['user_id','item_id','last_watch_dt','watched_pct']).sort_values(['user_id','last_watch_dt','item_id'])
        cut = ('synthetic','2021-01-05','2021-01-10','2021-01-15')
        a = s.prepare_split(films, cut, self.cfg)
        self.assertEqual([e.user_id for e in a.validation], [3, 4])  # no shifting to later valid day
        example = a.validation[0]
        self.assertFalse(example.delete_history)
        self.assertEqual(len(example.seen), 2)
        altered = films.copy()
        altered.loc[altered.user_id.eq(3)&altered.last_watch_dt.eq('2021-01-06'), 'watched_pct'] = 5.
        b = s.prepare_split(altered, cut, self.cfg)
        np.testing.assert_array_equal(a.users, b.users)
        np.testing.assert_array_equal(a.item_ids, b.item_ids)
        for key in ['all', 'delete']:
            np.testing.assert_array_equal(a.tokens[key], b.tokens[key])
            np.testing.assert_array_equal(a.percentages[key], b.percentages[key])
        self.assertEqual(a.validation[0].targets_any, b.validation[0].targets_any)
        self.assertTrue(a.validation[0].targets_high)
        self.assertFalse(b.validation[0].targets_high)
        model = s.CompactBERT4Rec(len(a.item_ids), self.cfg)
        _, predicted = s.evaluate(model, a.validation, 'DELETE10', self.cfg, torch.device('cpu'))
        record = predicted.loc[predicted.user_id.eq(3)].iloc[0]
        self.assertEqual(record.context_length, 0)
        self.assertFalse(set(map(int, record.recommendation_internal_ids.split())) & set(example.seen))

    def test_metrics_known_ranking(self):
        ndcg, recall = s.metrics([4, 2, 6], [2, 3])
        self.assertAlmostEqual(recall, .5)
        self.assertAlmostEqual(ndcg, (1/math.log2(3))/(1+1/math.log2(3)))


if __name__ == '__main__':
    unittest.main()
