"""Scientific contracts: leakage, deterministic selection, coefficient units."""
import unittest
import numpy as np
from position_decoders import DecoderConfig, select_decoder, coefficient_table

class DecoderTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.y = np.tile(np.arange(5), 30)
        self.x = rng.normal(size=(150, 12))
        self.x[:, :5] += np.eye(5)[self.y] * 4
        self.split = np.repeat(['train','validation','test'], [100,25,25])
        self.cfg = DecoderConfig(c_grid=(.01,.1,1.))

    def test_test_values_cannot_affect_selection_or_fit(self):
        for method in ('logistic','knn'):
            a = select_decoder(self.x,self.y,self.split,method,self.cfg)
            x,y=self.x.copy(),self.y.copy()
            x[self.split=='test'] = 1e9
            y[self.split=='test'] = 4
            b = select_decoder(x,y,self.split,method,self.cfg)
            self.assertEqual(a[2],b[2])
            np.testing.assert_allclose(a[3].validation_balanced_accuracy,b[3].validation_balanced_accuracy)
            np.testing.assert_array_equal(a[1].predict(self.x),b[1].predict(self.x))
            np.testing.assert_allclose(a[1].steps[0][1].mean_,self.x[self.split=='train'].mean(axis=0))

    def test_coefficient_export_reconstructs_decision_scores(self):
        _, model, _, _ = select_decoder(self.x,self.y,self.split,'logistic',self.cfg)
        table=coefficient_table(model,[f'f{i}' for i in range(12)])
        w=table.coefficient_input_units.to_numpy().reshape(5,12)
        intercept=table.groupby('position').intercept_input_units.first().to_numpy()
        np.testing.assert_allclose(self.x @ w.T + intercept, model.decision_function(self.x),atol=1e-10)
        self.assertEqual(model.steps[-1][1].l1_ratio,1.0)
        self.assertTrue((table.coefficient_standardized==0).any())

    def test_invalid_method_fails(self):
        with self.assertRaises(ValueError):
            select_decoder(self.x,self.y,self.split,'unknown')

if __name__ == '__main__':
    unittest.main()

