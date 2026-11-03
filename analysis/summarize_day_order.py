"""Paired descriptive inference-sensitivity analysis from saved evaluations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir',type=Path,required=True)
OUT=parser.parse_args().results_dir
SEEDS=[20261103,20261104,20261105]
METRICS=['ndcg10_high','recall10_high','ndcg10_any','recall10_any']


def main():
    complete=json.loads((OUT/'inference_complete.json').read_text(encoding='utf-8'))
    assert complete['models']==12 and complete['evaluations']==48
    runs=pd.read_csv(OUT/'results_per_run.csv')
    aggregates=[]; intervals=[]
    for split in ['main','replication']:
        mean={}
        for scenario in ['baseline']+[f'permuted_{s}' for s in SEEDS]:
            for method in ['ALL','SOFT']:
                frames=[pd.read_csv(OUT/f'{split}__{method}__seed{seed}'/scenario/'per_user.csv').set_index('user_id')[METRICS] for seed in [17,29,43]]
                assert all(f.index.equals(frames[0].index) for f in frames)
                mean[scenario,method]=sum(frames)/3
        for method in ['ALL','SOFT']:
            mean['permutation_average',method]=sum(mean[f'permuted_{s}',method] for s in SEEDS)/3
        contrasts=[]
        for scenario in ['baseline']+[f'permuted_{s}' for s in SEEDS]+['permutation_average']:
            a,b=mean[scenario,'ALL'],mean[scenario,'SOFT']
            assert a.index.equals(b.index)
            aggregates.append({'split':split,'scenario':scenario,'high_users':int(a.ndcg10_high.notna().sum()),
                **{'ALL_'+m:float(a[m].mean()) for m in METRICS},
                **{'SOFT_'+m:float(b[m].mean()) for m in METRICS},
                'SOFT_ALL_delta':float((b.ndcg10_high-a.ndcg10_high).mean()),
                'SOFT_ALL_relative_pct':float(100*(b.ndcg10_high.mean()/a.ndcg10_high.mean()-1))})
            gap=b.ndcg10_high-a.ndcg10_high
            contrasts.append((scenario,'SOFT-ALL',gap.dropna()))
            if scenario!='baseline':
                for method,frame in [('ALL',a),('SOFT',b)]:
                    contrasts.append((scenario,method+'_permuted-baseline',
                        (frame.ndcg10_high-mean['baseline',method].ndcg10_high).dropna()))
                basegap=mean['baseline','SOFT'].ndcg10_high-mean['baseline','ALL'].ndcg10_high
                contrasts.append((scenario,'change_in_SOFT_ALL_gap',(gap-basegap).dropna()))
        reference=contrasts[0][2].index
        assert all(c[2].index.equals(reference) for c in contrasts)
        values=np.column_stack([c[2].to_numpy() for c in contrasts])
        rng=np.random.default_rng(20261103)
        samples=np.empty((5000,len(contrasts)))
        for start in range(0,5000,64):
            count=min(64,5000-start)
            index=rng.integers(0,len(values),size=(count,len(values)))
            samples[start:start+count]=values[index].mean(axis=1)
        lo,hi=np.quantile(samples,[.025,.975],axis=0)
        for i,(scenario,contrast,series) in enumerate(contrasts):
            intervals.append({'split':split,'scenario':scenario,'contrast':contrast,
                'metric':'ndcg10_high','users':len(values),'mean_difference':float(series.mean()),
                'low':float(lo[i]),'high':float(hi[i]),'level':.95,'bootstrap_reps':5000,
                'scope':'exploratory_conditional_user_intervals_no_multiplicity_correction'})
    pd.DataFrame(aggregates).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    valid={'status':'passed','training_performed':False,'evaluations':len(runs),
        'max_baseline_CPU_vs_saved_mean_NDCG_difference':float(runs.CPU_vs_saved_ndcg_mean_difference.abs().max()),
        'baseline_changed_rank_rows':int(runs.CPU_vs_saved_changed_rank_rows.sum()),
        'scope':'inference stress only; does not establish insensitivity to training-day order',
        'complete_source_plan':complete['plan_hash'],'bootstrap_reps':5000,
        'confidence_level':.95,'conditional_on_training_and_permutation_seeds':True}
    (OUT/'analysis_complete.json').write_text(json.dumps(valid,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(aggregates)[['split','scenario','ALL_ndcg10_high','SOFT_ndcg10_high','SOFT_ALL_delta','SOFT_ALL_relative_pct']].to_string(index=False))
    print(pd.DataFrame(intervals).query("scenario=='permutation_average'")[['split','contrast','mean_difference','low','high']].to_string(index=False))


if __name__=='__main__':main()
