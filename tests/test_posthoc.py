"""Check subgroup boundaries and additive metrics on small known examples."""
import unittest
import numpy as np
import pandas as pd
from analysis.posthoc_analysis import features, pop_group, contributions, interval


class PosthocContract(unittest.TestCase):
    def test_training_frequency_boundaries(self):
        np.testing.assert_array_equal(pop_group([1,10,11,100,101]),[0,0,1,1,2])

    def test_group_features(self):
        rows=[]
        for user,n,weak in [(1,10,0),(2,20,2),(3,50,8)]:
            rows.extend([[user,item,5. if item>n-weak else 90.] for item in range(1,n+1)])
        frame=pd.DataFrame(rows,columns=['user_id','item_id','watched_pct'])
        f=features(frame,[1,2,3])
        np.testing.assert_array_equal(f.history,[0,1,2])
        np.testing.assert_array_equal(f.weak_share,[0,1,2])
        np.testing.assert_array_equal(f.context,[0,2,1])
        np.testing.assert_array_equal(f.delete_context_length,[10,18,19])

    def test_multi_target_contributions_use_full_idcg(self):
        prediction=np.array([np.arange(1,11),np.arange(11,21)])
        relevant=[{1,2,3,20},{11,20}]
        group=np.array([0]+[i%3 for i in range(1,21)])
        ndcg,recall,parts,items,hits=contributions(prediction,relevant,group,20)
        d=1/np.log2(np.arange(2,12))
        expected=[d[:3].sum()/d[:4].sum(),(d[0]+d[9])/d[:2].sum()]
        np.testing.assert_allclose(ndcg,expected)
        np.testing.assert_allclose(recall,[.75,1.])
        np.testing.assert_allclose(parts.sum(axis=1),expected)
        self.assertAlmostEqual(items.sum(),np.mean(expected))
        self.assertEqual(hits.sum(),5)

    def test_constant_bootstrap_interval(self):
        low,high=interval(np.ones((7,2)),reps=100)
        np.testing.assert_array_equal(low,[1.,1.])
        np.testing.assert_array_equal(high,[1.,1.])


if __name__=='__main__':
    unittest.main()
