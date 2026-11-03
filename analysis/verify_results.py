"""Independently audit the second KION study from raw data and saved recommendations.

The implementation supplied in the results ZIP is hashed, never executed.
No neural-network training and no loading of model weights.
"""
import argparse
import ast
import hashlib
import io
import json
import math
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'reference_results'
METHODS = ['ALL', 'DELETE10', 'TARGET10', 'STEP10', 'SOFT']
SEEDS = [17, 29, 43]
METRICS = ['ndcg10_any', 'recall10_any', 'ndcg10_high', 'recall10_high']
parser = argparse.ArgumentParser()
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--data-archive', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/audit')
args = parser.parse_args()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()

def close(actual, expected, label='', atol=1e-12):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0, equal_nan=True, err_msg=label)

def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def population(selected_rows, catalog, start, end, high_threshold):
    """Rebuild calendar populations via grouping, independently of the training module."""
    index = {int(item): i+1 for i, item in enumerate(catalog)}
    past = selected_rows.loc[selected_rows.last_watch_dt.le(start)].copy()
    mapped = past.loc[past.item_id.isin(index)].copy()
    mapped['internal'] = mapped.item_id.map(index)
    known = mapped.loc[mapped.watched_pct.notna()]
    histories = known.groupby('user_id').internal.agg(list).to_dict()
    deleted = known.loc[known.watched_pct.gt(10)].groupby('user_id').internal.agg(list).to_dict()
    seen = mapped.groupby('user_id').internal.agg(list).to_dict()
    future = selected_rows.loc[selected_rows.last_watch_dt.gt(start) & selected_rows.last_watch_dt.le(end)]
    first_date = future.groupby('user_id').last_watch_dt.transform('min')
    first = future.loc[future.last_watch_dt.eq(first_date)]
    records, info = [], {}
    for uid, group in first.groupby('user_id', sort=True):
        uid = int(uid)
        seen_set = set(seen.get(uid, []))
        valid = group.loc[group.item_id.isin(index) & group.watched_pct.notna()]
        any_internal = sorted({index[int(i)] for i in valid.item_id} - seen_set)
        if not any_internal:
            continue
        high_internal = sorted({index[int(i)] for i in valid.loc[valid.watched_pct.ge(high_threshold), 'item_id']} - seen_set)
        history, dhistory = histories.get(uid, []), deleted.get(uid, [])
        record = {'user_id': uid, 'history': history[-19:], 'delete_history': dhistory[-19:],
                  'seen': sorted(seen_set), 'targets_any': any_internal, 'targets_high': high_internal,
                  'history_length': len(history), 'target_day': str(group.last_watch_dt.iloc[0])}
        records.append(record)
        info[uid] = {'any': set(any_internal), 'high': set(high_internal), 'seen': seen_set,
                     'history': len(history), 'context': min(19, len(history)), 'delete_context': min(19, len(dhistory))}
    manifest = {'start_exclusive': start, 'end_inclusive': end,
                'users_with_raw_future_films': int(future.user_id.nunique()),
                'users_without_known_targets': int(first.user_id.nunique()) - len(info),
                'users_any': len(info), 'users_high': sum(bool(x['high']) for x in info.values()),
                'targets_any': sum(len(x['any']) for x in info.values()),
                'targets_high': sum(len(x['high']) for x in info.values()), 'fingerprint': canonical(records)}
    return info, manifest

def verify_user_rows(frame, info, method):
    assert frame.user_id.tolist() == list(info) and frame.user_id.is_unique
    assert pd.api.types.is_bool_dtype(frame.has_high)
    identifiers = list(info)
    expected_high = np.array([bool(info[u]['high']) for u in identifiers])
    np.testing.assert_array_equal(frame.has_high.to_numpy(), expected_high)
    for field, key in [('history_length', 'history'), ('context_length', 'delete_context' if method=='DELETE10' else 'context')]:
        np.testing.assert_array_equal(frame[field], [info[u][key] for u in identifiers])
    predictions = [list(map(int, s.split())) for s in frame.recommendation_internal_ids]
    assert all(len(p)==10 and len(set(p))==10 for p in predictions)
    assert all(not (set(p) & info[u]['seen']) for u,p in zip(identifiers,predictions))
    discount = 1 / np.log2(np.arange(2, 12))
    ideal = np.r_[0., np.cumsum(discount)]
    for target, suffix in [('any','any'), ('high','high')]:
        counts = np.array([len(info[u][target]) for u in identifiers])
        np.testing.assert_array_equal(frame['targets_'+suffix], counts)
        hits = np.array([[i in info[u][target] for i in p] for u,p in zip(identifiers,predictions)])
        ndcg = np.full(len(frame), np.nan)
        recall = np.full(len(frame), np.nan)
        valid = counts>0
        ndcg[valid] = (hits[valid]*discount).sum(axis=1) / ideal[np.minimum(10,counts[valid])]
        recall[valid] = hits[valid].sum(axis=1) / counts[valid]
        close(frame['ndcg10_'+suffix], ndcg, 'NDCG recomputed from ranked IDs')
        close(frame['recall10_'+suffix], recall, 'Recall recomputed from ranked IDs')
    return {**{m:float(frame[m].mean()) for m in METRICS}, 'users_any':len(frame), 'users_high':int(expected_high.sum())}

def pack_array(frame, users, catalog):
    last = frame.groupby('user_id', sort=False).tail(20)
    tokens = np.zeros((len(users),20), dtype=np.int64)
    pct = np.zeros((len(users),20), dtype=np.float32)
    row = np.searchsorted(users,last.user_id.to_numpy())
    column = last.groupby('user_id',sort=False).cumcount().to_numpy()
    tokens[row,column] = np.searchsorted(catalog,last.item_id.to_numpy())+1
    pct[row,column] = last.watched_pct.to_numpy(dtype=np.float32)
    return tokens,pct

with zipfile.ZipFile(args.archive) as z:
    assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == 263
    assert not any(n.endswith('.pt') for n in z.namelist())
    J = lambda name: json.loads(z.read(name))
    C = lambda name: pd.read_csv(io.BytesIO(z.read(name)), float_precision='round_trip')
    frozen, summary, lock = J('watch_study_plan.json'), J('FINAL_SUMMARY.json'), J('test_selection_lock.json')
    p, plan_hash = frozen['plan'], frozen['plan_hash']
    assert re.fullmatch('[a-f0-9]{64}',plan_hash) and canonical(p)==plan_hash==summary['plan_hash']==lock['plan_hash']
    assert summary['status']=='complete_30_model_study' and p['mode']=='scientific'
    assert p['methods']==summary['methods']==METHODS and p['seeds']==summary['seeds']==SEEDS
    assert p['models']==summary['models']==30 and len(lock['best_checkpoint_sha256'])==30
    assert p['primary_interval_level']==.9875 and len(p['primary_contrasts'])==4
    local_program=(ROOT/'kion_watch_feedback.py').read_text(encoding='utf-8')
    local_code_hash=hashlib.sha256(local_program.encode()).hexdigest()
    assert hashlib.sha256(z.read('implementation.py')).hexdigest()==p['code_sha256']==local_code_hash
    config_class=next(n for n in ast.parse(local_program).body if isinstance(n,ast.ClassDef) and n.name=='WatchConfig')
    expected_config={n.target.id:ast.literal_eval(n.value) for n in config_class.body if isinstance(n,ast.AnnAssign)}
    assert p['config']==summary['config']==expected_config and not p['config']['technical_mode']
    assert p['environment']==summary['environment'] and p['environment']['device']=='cuda'
    assert p['source_archive_sha256']==sha(args.data_archive)
    original_data_audit=json.loads((STUDY/'full_data_validation.json').read_text(encoding='utf-8'))
    assert original_data_audit['runtime_data_sha256']==hashlib.sha256(local_program.split('# %% model_training')[0].encode()).hexdigest()
    for name in ['main','replication']:
        assert p['data'][name]==J('data/'+name+'/manifest.json')==original_data_audit['full_data_manifests'][name]
    for name,digest in J('report_completed.json')['checksums'].items():
        assert hashlib.sha256(z.read(name)).hexdigest()==digest, name
    progress=J('progress.json')
    assert progress['completed_models']==progress['total_models']==30 and progress['status']=='training_complete'
    with zipfile.ZipFile(args.data_archive) as raw:
        with raw.open('data_en/items_en.csv') as f:
            items=pd.read_csv(f,usecols=['item_id','content_type'])
        with raw.open('data_en/interactions.csv') as f:
            all_rows=pd.read_csv(f,dtype={'user_id':'int32','item_id':'int32','watched_pct':'float32'})
    films=all_rows.loc[all_rows.item_id.isin(items.loc[items.content_type.eq('film'),'item_id'])].sort_values(['user_id','last_watch_dt','item_id'])
    assert len(films)==4137134 and not films.duplicated(['user_id','item_id']).any()
    del all_rows
    populations={}
    arrays={}
    raw_checks={}
    for name,train_end,val_end,test_end in p['cuts']:
        with np.load(io.BytesIO(z.read('data/'+name+'/selection_catalog.npz')),allow_pickle=False) as npz:
            users,catalog=npz['users'].copy(),npz['item_ids'].copy()
        training=films.loc[films.last_watch_dt.le(train_end)&films.watched_pct.notna()]
        counts=training.groupby('user_id').size()
        eligible=np.sort(counts.loc[counts.ge(10)].index.to_numpy())
        selected=np.sort(np.random.default_rng(p['config']['sample_seed']).choice(eligible,50000,replace=False))
        np.testing.assert_array_equal(users,selected)
        selected_rows=films.loc[films.user_id.isin(users)]
        training=training.loc[training.user_id.isin(users)]
        np.testing.assert_array_equal(catalog,np.sort(training.item_id.unique()))
        calculated_hashes={'users':array_hash(users),'items':array_hash(catalog)}
        for key,frame in [('all',training),('delete',training.loc[training.watched_pct.gt(10)])]:
            tokens,pct=pack_array(frame,users,catalog)
            calculated_hashes[key+'_tokens']=array_hash(tokens)
            calculated_hashes[key+'_pct']=array_hash(pct)
        assert calculated_hashes==p['data'][name]['array_hashes']
        for period,start,end in [('validation',train_end,val_end),('test',val_end,test_end)]:
            info,manifest=population(selected_rows,catalog,start,end,50)
            reference=p['data'][name]['validation'] if period=='validation' else lock['test_manifests'][name]
            assert manifest==reference,(name,period)
            populations[(name,period)]=info
        arrays[name]=(users,catalog)
        raw_checks[name]={'users':len(users),'catalog_items':len(catalog),'train_pairs':len(training),
                         'validation_users':len(populations[(name,'validation')]),'test_users':len(populations[(name,'test')])}
        print('Raw data, windows and future targets verified:',name,flush=True)

    stored_runs=C('results_per_run.csv').set_index(['split','method','seed']).sort_index()
    assert len(stored_runs)==30 and not stored_runs.index.duplicated().any()
    audit_runs=[]
    test_scores={}
    checked_rows=0
    for name in ['main','replication']:
        for method in METHODS:
            for seed in SEEDS:
                run=f'{name}__{method}__seed{seed}'
                prefix='runs/'+run+'/'
                completed,config,train=J(prefix+'completed.json'),J(prefix+'config.json'),J(prefix+'summary.json')
                identity={'plan_hash':plan_hash,'split':name,'method':method,'seed':seed,
                          'data_fingerprint':p['data'][name]['fingerprint']}
                assert completed['identity']==config['identity']==train['identity']==identity
                assert config['config']==p['config'] and config['environment']==p['environment']
                assert completed['completed_steps']==train['completed_steps']==20000 and train['mode']=='scientific'
                for filename,digest in completed['checksums'].items():
                    if filename.endswith('.pt'):
                        assert re.fullmatch('[a-f0-9]{64}',digest)
                    else:
                        assert hashlib.sha256(z.read(prefix+filename)).hexdigest()==digest,(run,filename)
                assert completed['checksums']['best.pt']==lock['best_checkpoint_sha256'][run]
                history=C(prefix+'history.csv')
                assert history.step.tolist()==list(range(1000,20001,1000))
                assert (history[['training_seconds','evaluation_seconds']].diff().iloc[1:]>0).all().all()
                best=history.loc[history.ndcg10_high.idxmax()]
                assert train['best_step']==best.step
                val_rows=C(prefix+'best_validation_per_user.csv')
                val=verify_user_rows(val_rows,populations[(name,'validation')],method)
                checked_rows+=len(val_rows)
                for metric,value in val.items():
                    close(value,train['best_validation'][metric],run+' validation '+metric)
                    close(value,best[metric],run+' selected history '+metric)
                testprefix='test_runs/'+run+'/'
                test_marker,test=C(testprefix+'per_user.csv'),J(testprefix+'summary.json')
                record=J(testprefix+'completed.json')
                test_identity={**identity,'weight_sha256':completed['checksums']['best.pt'],
                               'test_fingerprint':lock['test_manifests'][name]['fingerprint']}
                assert record['identity']==test['identity']==test_identity
                for filename,digest in record['checksums'].items():
                    assert hashlib.sha256(z.read(testprefix+filename)).hexdigest()==digest
                values=verify_user_rows(test_marker,populations[(name,'test')],method)
                checked_rows+=len(test_marker)
                for metric,value in values.items():
                    close(value,test['metrics'][metric],run+' test '+metric)
                test_scores[(name,method,seed)]=test_marker.set_index('user_id')[METRICS]
                totals=train['totals']
                assert totals['zero_weight_batches']==0
                close(train['actual_mask_fraction'],totals['masked_positions']/totals['real_positions'])
                close(train['effective_weighted_targets'],totals['weight_sum']**2/totals['weight_square_sum'],atol=1e-7)
                expected_users=50000 if method!='DELETE10' else p['data'][name]['stats']['delete_training_users']
                assert train['training_users']==expected_users
                row={'split':name,'method':method,'seed':seed,'best_step':train['best_step'],
                     **{'validation_'+k:v for k,v in val.items()},**{'test_'+k:v for k,v in values.items()},
                     **{k:train[k] for k in ['training_seconds','evaluation_seconds','checkpoint_seconds','actual_mask_fraction','effective_weighted_targets','training_users']},
                     **{'total_'+k:v for k,v in totals.items()}}
                stored=stored_runs.loc[(name,method,seed)]
                for key,value in row.items():
                    if key not in ['split','method','seed']:
                        close(value,stored[key],run+' '+key,atol=1e-7 if key=='effective_weighted_targets' or key.startswith('total_') else 1e-9)
                audit_runs.append(row)
        print('All validation/test recommendations verified:',name,flush=True)
    verified_runs=pd.DataFrame(audit_runs)
    for (split,seed),group in verified_runs.groupby(['split','seed']):
        full=group.loc[group.method.ne('DELETE10')].set_index('method')
        assert full.total_masked_positions.nunique()==full.total_real_positions.nunique()==1
        masks=float(full.loc['ALL','total_masked_positions'])
        strong=float(full.loc['TARGET10','total_positive_weight_targets'])
        close(full.loc['STEP10','total_weight_sum'],masks+2*strong,atol=1e-7)
        close(full.loc['STEP10','total_weight_square_sum'],masks+8*strong,atol=1e-7)
        for method in ['ALL','STEP10','SOFT']:
            assert full.loc[method,'total_positive_weight_targets']==masks
    numeric=[c for c in verified_runs.select_dtypes(include='number') if c!='seed']
    aggregate=verified_runs.groupby(['split','method'])[numeric].agg(['mean','std'])
    aggregate.columns=['_'.join(c) for c in aggregate.columns]
    aggregate['seeds']=3
    stored_aggregate=C('results_by_method.csv').set_index(['split','method']).loc[aggregate.index]
    for c in aggregate.columns:
        close(aggregate[c],stored_aggregate[c],c,atol=1e-7 if c.startswith(('total_','effective_')) else 1e-10)

    intervals=J('paired_intervals.json')
    assert len(intervals)==32 and [r for r in intervals if r['role']=='primary']==summary['primary_comparisons']
    recalculated=[]
    for split in ['main','replication']:
        means={method:sum((test_scores[(split,method,s)] for s in SEEDS))/3 for method in METHODS}
        for population_name in ['high','any']:
            part=[r for r in intervals if r['split']==split and r['metric'].endswith('_'+population_name)]
            columns=[]
            for row in part:
                delta=(means[row['left']][row['metric']]-means[row['right']][row['metric']]).dropna()
                columns.append(delta.to_numpy())
                assert len(delta)==row['users'] and row['repetitions']==5000
                expected_level=.9875 if row['role']=='primary' else .95
                assert row['level']==expected_level
                close(delta.mean(),row['mean_difference'])
            deltas=np.column_stack(columns)
            rng=np.random.default_rng(p['config']['bootstrap_seed'])
            bootstrap_means=np.empty((5000,len(part)))
            for start in range(0,5000,64):
                count=min(64,5000-start)
                index=rng.integers(0,len(deltas),size=(count,len(deltas)))
                bootstrap_means[start:start+count]=deltas[index].mean(axis=1)
            for column,row in enumerate(part):
                alpha=1-row['level']
                low,high=np.quantile(bootstrap_means[:,column],[alpha/2,1-alpha/2])
                close([low,high],[row['low'],row['high']],f'{split} {row["left"]}-{row["right"]} {row["metric"]}')
                recalculated.append({**row,'independently_recomputed_low':float(low),'independently_recomputed_high':float(high)})
            print('Bootstrap independently recomputed:',split,population_name,flush=True)

    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    copy=out/'received_results.zip'
    if not copy.exists():
        shutil.copyfile(args.archive,copy)
    assert sha(copy)==sha(args.archive)
    for filename in ['FINAL_SUMMARY.json','watch_study_plan.json','test_selection_lock.json','results_by_method.csv',
                     'results_per_run.csv','paired_intervals.json','paired_intervals.csv','learning_curves.csv','benchmark.json',
                     'quality_overview.png','quality_overview.svg','cost_overview.png','cost_overview.svg','primary_differences.png','primary_differences.svg']:
        (out/filename).write_bytes(z.read(filename))
    verified_runs.to_csv(out/'independent_results_per_run.csv',index=False)
    aggregate.reset_index().to_csv(out/'independent_results_by_method.csv',index=False)
    pd.DataFrame(recalculated).to_csv(out/'independent_intervals.csv',index=False)
    effects=[]
    for split in ['main','replication']:
        for left,right in [('SOFT','ALL'),('SOFT','STEP10'),('SOFT','DELETE10'),('TARGET10','DELETE10')]:
            a,b=aggregate.loc[(split,left)],aggregate.loc[(split,right)]
            effects.append({'split':split,'left':left,'right':right,
                            **{metric+'_relative_percent':float(100*(a[metric+'_mean']/b[metric+'_mean']-1))
                               for metric in ['test_ndcg10_high','test_recall10_high','test_ndcg10_any','test_recall10_any','training_seconds']}})
    pd.DataFrame(effects).to_csv(out/'relative_effects.csv',index=False)
    total_seconds=verified_runs[['training_seconds','evaluation_seconds','checkpoint_seconds']].sum()
    audit={'status':'passed','archive_sha256':sha(args.archive),'archive_files':len(z.namelist()),
           'plan_hash':plan_hash,'code_sha256':local_code_hash,'models':30,'steps_per_model':20000,
           'raw_cohorts_catalogs_training_windows_validation_and_test_targets_rebuilt':True,
           'validation_and_test_user_rows_recomputed_from_ranked_ids':checked_rows,
           'bootstrap_intervals_independently_recomputed':32,'replications':5000,
           'primary_positive_intervals':sum(r['low']>0 for r in intervals if r['role']=='primary'),
           'primary_comparisons':summary['primary_comparisons'],'raw_data_checks':raw_checks,
           'available_checksums_verified':True,'model_weight_contents_verified':False,
           'training_hours':float(total_seconds.training_seconds/3600),
           'validation_hours':float(total_seconds.evaluation_seconds/3600),
           'recorded_training_validation_checkpoint_hours':float(total_seconds.sum()/3600),
           'scope':'independent_numeric_audit_no_retraining_no_execution_of_archive_code_no_model_weights_in_ZIP'}
    close(audit['training_hours'],summary['training_hours_without_validation'])
    json_write(out/'independent_review.json',audit)
    json_write(out/'latest_results_review.json',{'status':'passed','output_directory':str(out),
                                                'archive_sha256':audit['archive_sha256'],'plan_hash':plan_hash})
    print(json.dumps(audit,ensure_ascii=False,indent=2))
