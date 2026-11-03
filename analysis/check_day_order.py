"""Inference-only stress test; reuse frozen model, evaluator and saved 40k weights."""
import argparse
import dataclasses
import hashlib
import importlib.util
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parents[1]
FULL=None
RAW=None
OUT=None
PERMUTATION_SEEDS=[20261103,20261104,20261105]
METRICS=['ndcg10_high','recall10_high','ndcg10_any','recall10_any']
spec=importlib.util.spec_from_file_location('frozen_watch_feedback',HERE/'kion_watch_feedback.py')
s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)


def permuted_contexts(past, users, seed):
    """Permutation is deterministic per user/item/day and applied BEFORE truncation."""
    f=past.loc[past.user_id.isin(users)].copy()
    # SplitMix64; raw IDs and date only, no watched_pct or future feedback.
    date=pd.to_datetime(f.last_watch_dt).to_numpy(dtype='datetime64[D]').astype(np.uint64)
    with np.errstate(over='ignore'):
        x=(f.item_id.to_numpy(dtype=np.uint64)*np.uint64(0x9e3779b97f4a7c15)
           ^ f.user_id.to_numpy(dtype=np.uint64)*np.uint64(0xbf58476d1ce4e5b9)
           ^ date ^ np.uint64(seed))
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        f['_order']=x^(x>>np.uint64(31))
    f=f.sort_values(['user_id','last_watch_dt','_order','item_id'])
    return f.groupby('user_id',sort=False).tail(19).groupby('user_id',sort=False).internal.agg(list).to_dict()


def main():
    global FULL,RAW,OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--full-archive',type=Path,required=True)
    parser.add_argument('--data-archive',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/day_order_inference'))
    args=parser.parse_args()
    FULL,RAW,OUT=args.full_archive,args.data_archive,args.output_dir
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device('cpu')
    with zipfile.ZipFile(FULL) as z:
        frozen=json.loads(z.read('watch_study_plan.json'));p=frozen['plan']
        assert hashlib.sha256((HERE/'kion_watch_feedback.py').read_bytes()).hexdigest()==p['code_sha256']
        assert s.file_hash(RAW)==p['source_archive_sha256']
        cfg=s.WatchConfig(**p['config'])
        # Config kept intact. Inference does not call scientific training validation.
        first='runs/main__ALL__seed17/best.pt'
        state=torch.load(io.BytesIO(z.read(first)),map_location='cpu',weights_only=True)
        with np.load(io.BytesIO(z.read('data/main/selection_catalog.npz')),allow_pickle=False) as a:
            pilot_model=s.CompactBERT4Rec(len(a['item_ids']),cfg).to(device)
        pilot_model.load_state_dict(state['model']);pilot_model.eval()
        dummy=[s.EvalExample(i,list(range(1,20)),list(range(1,20)),list(range(1,20)),[20],[20],19,'2021-08-09') for i in range(256)]
        s.evaluate(pilot_model,dummy[:16],'ALL',cfg,device)
        started=time.perf_counter();s.evaluate(pilot_model,dummy,'ALL',cfg,device)
        elapsed=time.perf_counter()-started
        print('PILOT seconds/256:',round(elapsed,3),'estimated 48 full evaluations minutes:',round(elapsed/256*14200*48/60,1),flush=True)
        if args.pilot: return
        del pilot_model,state
        plan={'source_plan_hash':frozen['plan_hash'],'source_archive':str(FULL),
            'script_sha256':s.file_hash(__file__),'source_code_sha256':p['code_sha256'],
            'methods':['ALL','SOFT'],'training_seeds':[17,29,43],
            'permutation_seeds':PERMUTATION_SEEDS,'budget':40000,
            'environment':s.environment(device),'training_performed':False,
            'context':'reorder full known past within same recorded day, then take last19',
            'targets_and_seen':'original fixed test targets and exclusions',
            'scope':'exploratory inference perturbation on ID-order-trained weights; not training-order robustness'}
        path=OUT/'plan.json'
        if path.exists(): assert json.loads(path.read_text(encoding='utf-8'))==plan
        else: s.save_json(path,plan)
        films=s.read_films(RAW)
        results=[];diagnostics=[]
        for split,train_end,val_end,test_end in p['cuts']:
            with np.load(io.BytesIO(z.read('data/'+split+'/selection_catalog.npz')),allow_pickle=False) as a:
                users,catalog=a['users'],a['item_ids']
            rows=films.loc[films.user_id.isin(users)].copy()
            baseline,manifest=s.make_examples(rows,users,catalog,val_end,test_end,cfg)
            assert manifest==json.loads(z.read('test_selection_lock.json'))['test_manifests'][split]
            mapping={int(item):i+1 for i,item in enumerate(catalog)}
            past=rows.loc[rows.last_watch_dt.le(val_end)&rows.watched_pct.notna()&rows.item_id.isin(catalog)].copy()
            past['internal']=past.item_id.map(mapping)
            scenarios={'baseline':baseline}
            for seed in PERMUTATION_SEEDS:
                contexts=permuted_contexts(past,[x.user_id for x in baseline],seed)
                changed=[dataclasses.replace(x,history=contexts.get(x.user_id,[])) for x in baseline]
                assert all((a.user_id,a.targets_any,a.targets_high,a.seen,a.history_length)==(b.user_id,b.targets_any,b.targets_high,b.seen,b.history_length) for a,b in zip(baseline,changed))
                assert all(len(a.history)==len(b.history) for a,b in zip(baseline,changed))
                diagnostics.append({'split':split,'permutation_seed':seed,'users':len(baseline),
                    'contexts_order_changed':sum(a.history!=b.history for a,b in zip(baseline,changed)),
                    'contexts_item_set_changed':sum(set(a.history)!=set(b.history) for a,b in zip(baseline,changed)),
                    'high_users':sum(bool(a.targets_high) for a in baseline)})
                scenarios[f'permuted_{seed}']=changed
            print('Contexts ready:',split,len(baseline),'users',flush=True)
            for method in ['ALL','SOFT']:
                for seed in [17,29,43]:
                    name=s.run_name(split,method,seed)
                    weight_name='runs/'+name+'/best.pt'
                    digest=hashlib.sha256(z.read(weight_name)).hexdigest()
                    marker=json.loads(z.read('runs/'+name+'/completed.json'))
                    assert digest==marker['checksums']['best.pt']
                    state=torch.load(io.BytesIO(z.read(weight_name)),map_location='cpu',weights_only=True)
                    assert state['identity']==s.expected_identity(type('Split',(),{'name':split,'manifest':p['data'][split]})(),method,seed,frozen)
                    model=s.CompactBERT4Rec(len(catalog),cfg).to(device)
                    model.load_state_dict(state['model']);del state
                    for scenario,examples in scenarios.items():
                        folder=OUT/name/scenario;completion=folder/'completed.json'
                        identity={'plan_hash':s.canonical(plan),'weight_sha256':digest,
                                  'test_fingerprint':manifest['fingerprint'],'scenario':scenario}
                        if completion.exists():
                            cached=json.loads(completion.read_text(encoding='utf-8'));assert cached['identity']==identity
                            assert all(s.file_hash(folder/n)==h for n,h in cached['checksums'].items())
                            result=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
                        else:
                            started=time.perf_counter()
                            metrics,frame=s.evaluate(model,examples,method,cfg,device)
                            result={'split':split,'method':method,'seed':seed,'scenario':scenario,
                                    'seconds':time.perf_counter()-started,**metrics}
                            if scenario=='baseline':
                                saved=pd.read_csv(io.BytesIO(z.read(f'evaluation/40000/{name}/per_user.csv'))).set_index('user_id')
                                frame=frame.set_index('user_id');assert frame.index.equals(saved.index)
                                result['CPU_vs_saved_max_metric_difference']=float(np.nanmax(np.abs(frame[METRICS].to_numpy()-saved[METRICS].to_numpy())))
                                result['CPU_vs_saved_changed_rank_rows']=int(frame.recommendation_internal_ids.ne(saved.recommendation_internal_ids).sum())
                                result['CPU_vs_saved_ndcg_mean_difference']=float(frame.ndcg10_high.mean()-saved.ndcg10_high.mean())
                                frame=frame.reset_index()
                                if abs(result['CPU_vs_saved_ndcg_mean_difference'])>1e-4:
                                    raise ValueError('CPU baseline differs materially from saved GPU evaluation; inspect before perturbing')
                            s.save_csv(folder/'per_user.csv',frame);s.save_json(folder/'summary.json',result)
                            s.save_json(completion,{'identity':identity,'checksums':{n:s.file_hash(folder/n) for n in ['summary.json','per_user.csv']}})
                        results.append(result)
                        print(name,scenario,'NDCG',round(result['ndcg10_high'],6),'seconds',round(result['seconds'],1),flush=True)
                    del model
        s.save_csv(OUT/'results_per_run.csv',results)
        s.save_csv(OUT/'context_diagnostics.csv',diagnostics)
        s.save_json(OUT/'inference_complete.json',{'models':12,'evaluations':len(results),
            'training_performed':False,'plan_hash':s.canonical(plan)})


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
