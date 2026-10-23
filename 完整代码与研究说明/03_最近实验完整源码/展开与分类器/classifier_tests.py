#!/usr/bin/env python3
"""Small synthetic-data checks only. These outputs are never experiment evidence."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
import classifier as c


class ClassifierChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='synthetic_classifier_preflight_')
        cls.root = Path(cls.tmp.name)
        cls.x, cls.y = make_classification(n_samples=36, n_features=6, n_informative=4,
                                            n_redundant=0, random_state=7)
        cls.ids = np.arange(900001, 900037)
        cls.splits = []
        for f, (tr, te) in enumerate(StratifiedKFold(3, shuffle=True, random_state=1).split(cls.x, cls.y)):
            inner = [{'train': tr[it].tolist(), 'validation': tr[iv].tolist()}
                     for it, iv in StratifiedKFold(2, shuffle=True, random_state=f + 3).split(cls.x[tr], cls.y[tr])]
            cls.splits.append({'fold': f, 'train': tr.tolist(), 'test': te.tolist(), 'inner': inner})
        data = cls.root / 'source/downstream_data'; data.mkdir(parents=True)
        (cls.root / 'features').mkdir()
        np.save(data / 'Y.npy', cls.y); np.save(data / 'IDs.npy', cls.ids)
        np.save(cls.root / 'features/synthetic.npy', cls.x)
        c.put(cls.root / 'source/downstream_splits.json', cls.splits)
        cls.protocol = cls.root / 'protocol.json'
        c.put(cls.protocol, {'synthetic_test_only': True, 'testing': {'subjects': 36, 'outer': 3, 'inner': 2},
                            'data': {'source_root': 'source'}, 'output_dir': 'SYNTHETIC_ONLY'})

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_split_guard(self):
        c.validate_splits(self.splits, self.y, self.ids, outer=3, inner=2, expected_n=36)
        bad = json.loads(json.dumps(self.splits))
        bad[0]['inner'][0]['train'][0] = bad[0]['test'][0]
        with self.assertRaises(ValueError): c.validate_splits(bad, self.y, self.ids, 3, 2, 36)

    def test_configuration_and_probability_semantics(self):
        for head in ('lr', 'svm', 'mlp'):
            model, diag = c.fit_model(self.x, self.y, self.splits[0]['train'], head, .01, 17)
            scores, predicted, probabilities = c.score_model(model, self.x, head)
            self.assertTrue(np.isfinite(scores).all())
            self.assertTrue(np.allclose(model[0].mean_, self.x[self.splits[0]['train']].mean(0)))
            if head == 'svm':
                self.assertIsNone(probabilities)
                self.assertIsNone(c.metrics(self.y, scores, predicted, probabilities)['BCE'])
            else: self.assertTrue(np.logical_and(probabilities >= 0, probabilities <= 1).all())
            if head == 'mlp':
                self.assertEqual(model[1].hidden_layer_sizes, (32,))
                self.assertEqual(model[1].activation, 'relu')
                self.assertEqual(model[1].out_activation_, 'logistic')
                self.assertFalse(model[1].early_stopping)
                self.assertEqual(len(diag['loss_curve']), model[1].n_iter_)
            if head == 'lr':
                self.assertEqual(model[1].solver, 'liblinear')
                self.assertEqual(model[1].max_iter, 20000)
                self.assertEqual(model[1].tol, 1e-6)

    def test_ties_choose_stronger_regularization(self):
        choices = [{'value': value, 'mean_inner_AUROC': .6} for value in [.001, .003, .01]]
        self.assertEqual(c.choose_candidate(choices, 'lr')['value'], .001)
        self.assertEqual(c.choose_candidate(choices, 'svm')['value'], .001)
        self.assertEqual(c.choose_candidate(choices, 'mlp')['value'], .01)

    def test_production_cannot_accept_synthetic_flag(self):
        with self.assertRaises(ValueError):
            c.run(self.protocol, {'name': 'not_real', 'feature_key': 'synthetic', 'head': 'lr'}, preflight=True)

    def test_full_nested_run_and_resume(self):
        for head in ('lr', 'svm', 'mlp'):
            job = {'name': f'synthetic_{head}', 'feature_key': 'synthetic', 'head': head}
            pre = c.run(self.protocol, job, test_mode=True, preflight=True)
            self.assertEqual(pre['status'], 'PREFLIGHT_OK')
            result = c.run(self.protocol, job, test_mode=True)
            self.assertEqual(result['subjects'], 36)
            self.assertEqual(result['nested_folds'], '3x2')
            before = c.sha(self.root / 'SYNTHETIC_ONLY' / job['name'] / 'fold_00/classifier.joblib')
            repeated = c.run(self.protocol, job, test_mode=True)
            after = c.sha(self.root / 'SYNTHETIC_ONLY' / job['name'] / 'fold_00/classifier.joblib')
            self.assertEqual(before, after)
            self.assertEqual(result['pooled'], repeated['pooled'])
            self.assertEqual(result['probability_available'], head != 'svm')
            if head == 'svm': self.assertIsNone(result['pooled']['BCE'])

    def test_tampered_completed_fold_is_rejected(self):
        job = {'name': 'synthetic_tamper', 'feature_key': 'synthetic', 'head': 'lr'}
        c.run(self.protocol, job, test_mode=True)
        p = self.root / 'SYNTHETIC_ONLY/synthetic_tamper/fold_00/selection.json'
        p.write_text(p.read_text() + '\n')
        with self.assertRaises(ValueError): c.run(self.protocol, job, test_mode=True)


if __name__ == '__main__': unittest.main(verbosity=2)
