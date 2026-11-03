"""Reuse the original independent evaluator to audit the 20k/40k archives."""
import ast
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--full-archive',type=Path,required=True)
parser.add_argument('--results-archive',type=Path,required=True)
parser.add_argument('--original-results',type=Path,required=True)
parser.add_argument('--data-archive',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,default=Path('outputs/budget_audit'))
args=parser.parse_args()
HERE=Path(__file__).resolve().parents[1]
RAW=args.data_archive
OLD=args.original_results
FULL=args.full_archive
LIGHT=args.results_archive
OUT=args.output_dir
OUT.mkdir(exist_ok=True)
METRICS=['ndcg10_high','recall10_high','ndcg10_any','recall10_any']

# Reuse pure functions from the already audited evaluator. Never execute archive code.
tree=ast.parse((HERE/'analysis/verify_results.py').read_text(encoding='utf-8'))
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
           ['canonical','close','population','verify_user_rows']]
exec(compile(ast.Module(body=functions,type_ignores=[]),'<existing independent evaluator>','exec'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def main():
    checked_rows=0; selected=[]; frames={}; stored_runs=[]; exact20=0; rank_diffs=0
    with zipfile.ZipFile(LIGHT) as z,zipfile.ZipFile(FULL) as f,zipfile.ZipFile(OLD) as old:
        assert z.testzip() is f.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        assert len(f.namelist())==len(set(f.namelist()))
        for name in z.namelist(): assert z.read(name)==f.read(name),name
        J=lambda name:json.loads(z.read(name))
        C=lambda name:pd.read_csv(io.BytesIO(z.read(name)),float_precision='round_trip')
        frozen=J('watch_study_plan.json'); p=frozen['plan']; ph=frozen['plan_hash']
        final=J('FINAL_SUMMARY.json'); lock=J('budget_selection_lock.json')
        assert canonical(p)==ph==final['plan_hash']==lock['plan_hash']
        assert final['status']=='complete' and final['trained_models']==12 and final['evaluated_models']==24
        assert p['methods']==['ALL','SOFT'] and p['seeds']==[17,29,43]
        expected=json.loads(old.read('watch_study_plan.json'))['plan']['config']
        expected['max_steps']=40000
        assert p['config']==expected
        assert sha(RAW)==p['source_archive_sha256']
        assert hashlib.sha256(z.read('implementation.py')).hexdigest()==p['code_sha256']
        assert z.read('implementation.py')==old.read('implementation.py')
        assert hashlib.sha256(z.read('review_cell.py')).hexdigest()==p['adapter_sha256']
        expected_ast=ast.parse((HERE/'analysis/Continue_Article2_40k_Cell.py').read_text(encoding='utf-8')).body
        actual_ast=ast.parse(z.read('review_cell.py').decode()).body
        assert ast.dump(ast.Module(body=actual_ast[-len(expected_ast):],type_ignores=[]))==ast.dump(ast.Module(body=expected_ast,type_ignores=[]))
        with zipfile.ZipFile(RAW) as raw:
            items=pd.read_csv(raw.open('data_en/items_en.csv'),usecols=['item_id','content_type'])
            rows=pd.read_csv(raw.open('data_en/interactions.csv'),dtype={'user_id':'int32','item_id':'int32','watched_pct':'float32'})
        films=rows.loc[rows.item_id.isin(items.loc[items.content_type.eq('film'),'item_id'])].sort_values(['user_id','last_watch_dt','item_id'])
        for split,train_end,val_end,test_end in p['cuts']:
            assert p['data'][split]==json.loads(old.read('data/'+split+'/manifest.json'))
            assert z.read('data/'+split+'/selection_catalog.npz')==old.read('data/'+split+'/selection_catalog.npz')
            with np.load(io.BytesIO(z.read('data/'+split+'/selection_catalog.npz')),allow_pickle=False) as a:
                users,catalog=a['users'],a['item_ids']
            info,manifest=population(films.loc[films.user_id.isin(users)],catalog,val_end,test_end,50)
            assert manifest==J('test_selection_lock.json')['test_manifests'][split]
            for method in p['methods']:
                for seed in p['seeds']:
                    name=f'{split}__{method}__seed{seed}'
                    identity={'plan_hash':ph,'split':split,'method':method,'seed':seed,'data_fingerprint':p['data'][split]['fingerprint']}
                    history=C('runs/'+name+'/history.csv')
                    assert history.step.tolist()==list(range(1000,40001,1000))
                    summary=J('runs/'+name+'/summary.json')
                    assert summary['completed_steps']==40000 and summary['identity']==identity
                    completed=J('runs/'+name+'/completed.json')
                    for file,digest in completed['checksums'].items():
                        assert hashlib.sha256(f.read('runs/'+name+'/'+file)).hexdigest()==digest
                    s20=J('budget20k/'+name+'/selection.json')
                    assert s20['identity']==identity
                    for file,digest in s20['checksums'].items():
                        assert hashlib.sha256(f.read('budget20k/'+name+'/'+file)).hexdigest()==digest
                    for budget in [20000,40000]:
                        h=history.loc[history.step.le(budget)]
                        best=h.loc[h.ndcg10_high.idxmax()]
                        selection=s20 if budget==20000 else summary
                        step=selection['selected_step'] if budget==20000 else selection['best_step']
                        assert step==int(best.step)
                        folder=f'evaluation/{budget}/{name}/'
                        marker=J(folder+'completed.json'); report=J(folder+'summary.json')
                        for file,digest in marker['checksums'].items():
                            assert hashlib.sha256(z.read(folder+file)).hexdigest()==digest
                        weight=f"{'budget20k' if budget==20000 else 'runs'}/{name}/best.pt"
                        assert hashlib.sha256(f.read(weight)).hexdigest()==report['weight_sha256']
                        if budget==20000:
                            assert report['weight_sha256']==lock['budget20k_sha256'][name]
                        else:
                            assert report['weight_sha256']==lock['budget40k']['best_checkpoint_sha256'][name]
                        assert report['selected_step']==step and report['plan_hash']==ph
                        frame=C(folder+'per_user.csv')
                        metrics=verify_user_rows(frame,info,method)
                        checked_rows+=len(frame)
                        for metric,value in metrics.items(): close(value,report[metric])
                        frames[budget,split,method,seed]=frame.set_index('user_id')[METRICS]
                        if budget==20000:
                            old_frame=pd.read_csv(io.BytesIO(old.read('test_runs/'+name+'/per_user.csv')),float_precision='round_trip')
                            pd.testing.assert_frame_equal(frame.drop(columns='recommendation_internal_ids'),old_frame.drop(columns='recommendation_internal_ids'),check_exact=True)
                            rank_diffs+=int(frame.recommendation_internal_ids.ne(old_frame.recommendation_internal_ids).sum())
                            exact20+=1
                        selected.append({'split':split,'method':method,'seed':seed,'budget':budget,
                            'selected_step':step,'validation_ndcg':float(best.ndcg10_high),
                            'at_budget_limit':step==budget,**{m:metrics[m] for m in METRICS}})
                    stored_runs.append(summary)
            print('Targets, predictions, metrics, weights verified:',split,flush=True)
        derived=pd.DataFrame(selected)
        aggregate=derived.groupby(['budget','split','method'])[METRICS].agg(['mean','std'])
        aggregate.columns=['_'.join(c) for c in aggregate.columns]
        stored=C('results_by_method.csv').set_index(['budget','split','method']).sort_index()
        close(aggregate.loc[stored.index].to_numpy(),stored.to_numpy())
        means={(b,s,m):sum(frames[b,s,m,k] for k in [17,29,43])/3
               for b in [20000,40000] for s in ['main','replication'] for m in ['ALL','SOFT']}
        comparisons={}
        for split in ['main','replication']:
            for budget in [20000,40000]:
                comparisons[split,f'SOFT-ALL at {budget}']=means[budget,split,'SOFT']-means[budget,split,'ALL']
            for method in ['ALL','SOFT']:
                comparisons[split,f'40k-20k {method}']=means[40000,split,method]-means[20000,split,method]
        intervals=C('paired_intervals.csv')
        assert len(intervals)==32
        # Batch columns sharing the same user population: same draws as separate bootstraps.
        for split in ['main','replication']:
            for suffix in ['high','any']:
                part=intervals.loc[intervals.split.eq(split)&intervals.metric.str.endswith('_'+suffix)]
                deltas=np.column_stack([comparisons[split,r.contrast][r.metric].dropna().to_numpy() for r in part.itertuples()])
                rng=np.random.default_rng(20261031); draws=np.empty((5000,len(part)))
                for start in range(0,5000,64):
                    count=min(64,5000-start)
                    indices=rng.integers(0,len(deltas),size=(count,len(deltas)))
                    draws[start:start+count]=deltas[indices].mean(axis=1)
                close(deltas.mean(axis=0),part.mean_difference.to_numpy())
                close(np.quantile(draws,.025,axis=0),part.low.to_numpy())
                close(np.quantile(draws,.975,axis=0),part.high.to_numpy())
                assert part.level.eq(.95).all() and part.repetitions.eq(5000).all()
        derived.to_csv(OUT/'verified_runs.csv',index=False)
        aggregate.reset_index().to_csv(OUT/'verified_aggregate.csv',index=False)
        intervals.to_csv(OUT/'verified_intervals.csv',index=False)
        report={'status':'passed','plan_hash':ph,'raw_sha256':sha(RAW),
            'full_zip_sha256':sha(FULL),'results_zip_sha256':sha(LIGHT),
            'source_implementation_unchanged':True,'adapter_matches_reviewed_local_code':True,
            'source_cohorts_catalogs_and_test_targets_match_original':True,
            'models_20k_exactly_reproduce_original_per_user_metrics':exact20,
            'changed_ranked_rows_vs_original_at20k':rank_diffs,
            'user_rows_recomputed':checked_rows,'bootstrap_intervals_recomputed':32,
            'all_recorded_weight_hashes_verified':True,'weight_tensor_contents_loaded':False,
            'best_steps_above_20k':int(derived.query('budget==40000').selected_step.gt(20000).sum()),
            'best_at_40k':int(derived.query('budget==40000').at_budget_limit.sum()),
            'training_hours':sum(x['training_seconds'] for x in stored_runs)/3600,
            'validation_hours':sum(x['evaluation_seconds'] for x in stored_runs)/3600,
            'scope':'numeric_results_and_weight_file_integrity; no retraining or weight inference',
            'inference':'exploratory_conditional_user_intervals; no multiplicity correction',
            'environment':p['environment']}
        (OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__': main()
