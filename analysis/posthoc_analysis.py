"""Exploratory SOFT vs DELETE10 analysis. No training, no model-weight loading.

All subgroup features precede the test period. Saved predictions are evaluated
against rebuilt targets. Public outputs contain aggregates, never user rows.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ['ALL', 'DELETE10', 'SOFT']
SEEDS = [17, 29, 43]
AXES = {
    'history': ['10–19', '20–49', '50+'],
    'weak_share': ['0', '(0; 1/3]', '(1/3; 1]'],
    'context': ['no_weak_recent', 'weak_refilled_19', 'weak_shorter_19'],
}
POP_GROUPS = ['1–10', '11–100', '101+']
DESIGN = {'status': 'exploratory_after_aggregate_results', 'contrast': 'SOFT-DELETE10',
          'outcome': 'ndcg10_high', 'axes': AXES, 'film_training_frequency': POP_GROUPS,
          'bootstrap_repetitions': 5000, 'bootstrap_seed': 20261031, 'level': .95,
          'interval_scope': 'descriptive_conditional_on_trained_models_no_multiplicity_adjustment',
          'group_features': 'observed_before_test_only', 'new_training': False}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def interval(values, reps=5000, seed=20261031):
    a = np.asarray(values, dtype=float)
    if a.ndim == 1: a = a[:, None]
    assert len(a) and np.isfinite(a).all()
    rng = np.random.default_rng(seed)
    samples = np.empty((reps, a.shape[1]))
    for start in range(0, reps, 64):
        count = min(64, reps-start)
        index = rng.integers(0, len(a), size=(count, len(a)))
        samples[start:start+count] = a[index].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975], axis=0)
    return low, high


def pop_group(count):
    return np.select([np.asarray(count) <= 10, np.asarray(count) <= 100], [0, 1], default=2)


def target_population(rows, catalog, start, end):
    index = {int(i): j+1 for j, i in enumerate(catalog)}
    past_raw = rows.loc[rows.last_watch_dt.le(start)]
    known_past = past_raw.loc[past_raw.item_id.isin(index) & past_raw.watched_pct.notna()]
    histories, deleted, seen = {}, {}, {}
    for uid, g in known_past.groupby('user_id', sort=False):
        histories[int(uid)] = [index[int(i)] for i in g.item_id]
        deleted[int(uid)] = [index[int(i)] for i in g.loc[g.watched_pct.gt(10), 'item_id']]
    for uid, g in past_raw.groupby('user_id', sort=False):
        seen[int(uid)] = sorted({index[int(i)] for i in g.item_id if int(i) in index})
    future = rows.loc[rows.last_watch_dt.gt(start) & rows.last_watch_dt.le(end)]
    day = future.groupby('user_id').last_watch_dt.transform('min')
    records, information = [], {}
    for uid, g in future.loc[future.last_watch_dt.eq(day)].groupby('user_id', sort=True):
        uid = int(uid)
        valid = g.loc[g.item_id.isin(index) & g.watched_pct.notna()]
        already = set(seen.get(uid, []))
        any_targets = sorted({index[int(i)] for i in valid.item_id} - already)
        if not any_targets: continue
        high = sorted({index[int(i)] for i in valid.loc[valid.watched_pct.ge(50), 'item_id']} - already)
        h, d = histories.get(uid, []), deleted.get(uid, [])
        records.append({'user_id':uid, 'history':h[-19:], 'delete_history':d[-19:],
                        'seen':seen.get(uid, []), 'targets_any':any_targets, 'targets_high':high,
                        'history_length':len(h), 'target_day':str(g.last_watch_dt.iloc[0])})
        information[uid] = {'high':set(high), 'any':set(any_targets), 'history_length':len(h),
                            'context':h[-19:], 'delete_context':d[-19:], 'seen':already}
    return information, canonical(records), known_past


def features(past, users):
    recent = past.groupby('user_id', sort=False).tail(19).copy()
    recent['weak'] = recent.watched_pct.le(10).astype(int)
    out = pd.DataFrame(index=pd.Index(users, name='user_id'))
    out['history_length'] = past.groupby('user_id').size().reindex(users)
    out['context_length'] = recent.groupby('user_id').size().reindex(users)
    out['weak_count'] = recent.groupby('user_id').weak.sum().reindex(users)
    retained = past.loc[past.watched_pct.gt(10)].groupby('user_id').size()
    out['delete_context_length'] = retained.reindex(users, fill_value=0).clip(upper=19)
    assert out.notna().all().all() and out.history_length.ge(10).all()
    out['history'] = np.select([out.history_length.lt(20), out.history_length.lt(50)], [0,1], default=2)
    out['weak_share'] = np.select([out.weak_count.eq(0), out.weak_count.mul(3).le(out.context_length)], [0,1], default=2)
    out['context'] = np.select([out.weak_count.eq(0), out.delete_context_length.eq(19)], [0,1], default=2)
    return out


def contributions(predictions, relevant, item_groups, catalog_size):
    """Keep the full user IDCG: item-group contributions add exactly to NDCG."""
    discount = 1/np.log2(np.arange(2, 12))
    ideal = np.r_[0., discount.cumsum()]
    counts = np.array([len(r) for r in relevant])
    assert (counts > 0).all() and predictions.shape == (len(relevant), 10)
    hits = np.array([[int(i) in r for i in p] for r,p in zip(relevant,predictions)])
    normalized = hits*discount / ideal[np.minimum(counts,10), None]
    recall_hits = hits/counts[:, None]
    ndcg = normalized.sum(axis=1)
    recall = recall_hits.sum(axis=1)
    groups = item_groups[predictions]
    part = np.column_stack([(normalized*(groups==g)).sum(axis=1) for g in range(3)])
    np.testing.assert_allclose(part.sum(axis=1), ndcg, atol=1e-15)
    item = np.bincount(predictions.ravel(), weights=normalized.ravel(), minlength=catalog_size+1)/len(relevant)
    np.testing.assert_allclose(item.sum(), ndcg.mean(), atol=1e-15)
    item_hits = np.bincount(predictions.ravel(), weights=hits.ravel(), minlength=catalog_size+1)
    return ndcg, recall, part, item, item_hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--data-archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {**DESIGN, 'design_hash':canonical(DESIGN)}
    write_json(args.output_dir/'exploratory_plan.json', plan)
    subgroup_rows, film_rows, item_rows, recommendation_rows, overall_rows = [], [], [], [], []
    overlap, features_by_cut = {}, {}
    with zipfile.ZipFile(args.archive) as z, zipfile.ZipFile(args.data_archive) as raw:
        assert z.testzip() is None
        J = lambda name: json.loads(z.read(name))
        p = J('watch_study_plan.json')['plan']
        if p['mode'] != 'scientific' or p['config']['technical_mode'] or p['config']['n_users'] != 50000:
            raise ValueError('This analysis requires the complete scientific protocol; CPU smoke results are not accepted.')
        assert canonical(p) == J('FINAL_SUMMARY.json')['plan_hash']
        assert sha(args.data_archive) == p['source_archive_sha256']
        assert hashlib.sha256(z.read('implementation.py')).hexdigest() == p['code_sha256']
        assert p['code_sha256'] == '5ab07ac369ae3966f3d781ada94a38415eed710ade65541c594c2a4fd3b57509'
        for name,digest in J('report_completed.json')['checksums'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
        items = pd.read_csv(raw.open('data_en/items_en.csv'), usecols=['item_id','content_type','title_orig','release_year'])
        rows = pd.read_csv(raw.open('data_en/interactions.csv'), dtype={'user_id':'int32','item_id':'int32','watched_pct':'float32'})
        films = rows.loc[rows.item_id.isin(items.loc[items.content_type.eq('film'),'item_id'])].sort_values(['user_id','last_watch_dt','item_id'])
        del rows
        for split, train_end, val_end, test_end in p['cuts']:
            with np.load(io.BytesIO(z.read('data/'+split+'/selection_catalog.npz')), allow_pickle=False) as selection:
                users, catalog = selection['users'].copy(), selection['item_ids'].copy()
            selected = films.loc[films.user_id.isin(users)]
            info, fingerprint, past = target_population(selected, catalog, val_end, test_end)
            assert fingerprint == J('test_selection_lock.json')['test_manifests'][split]['fingerprint']
            all_users = list(info)
            high_users = [u for u in all_users if info[u]['high']]
            overlap[split] = {'cohort':set(map(int,users)), 'test_high':set(high_users)}
            feature = features(past, high_users)
            features_by_cut[split] = feature
            relevant = [info[u]['high'] for u in high_users]
            training = selected.loc[selected.last_watch_dt.le(train_end) & selected.watched_pct.notna()]
            counts = training.groupby('item_id').size().reindex(catalog).to_numpy()
            assert (counts > 0).all()
            frequency = np.r_[0, counts]
            groups = pop_group(frequency)
            target_counts = np.bincount(np.array([i for r in relevant for i in r]), minlength=len(catalog)+1)
            mean, components, item_components, item_hit_counts = {}, {}, {}, {}
            # The rest of the calculations are in the same frozen group design.
            for method in METHODS:
                values, parts, per_item, per_item_hits = [], [], [], []
                for seed in SEEDS:
                    folder = f'test_runs/{split}__{method}__seed{seed}/'
                    frame = pd.read_csv(io.BytesIO(z.read(folder+'per_user.csv')), float_precision='round_trip')
                    assert frame.user_id.tolist() == all_users and frame.user_id.is_unique
                    marker = J(folder+'completed.json')
                    # The full original audit already checks these markers; compare its CSV hash again.
                    assert hashlib.sha256(z.read(folder+'per_user.csv')).hexdigest() == marker['checksums']['per_user.csv']
                    high = frame.loc[frame.has_high].set_index('user_id').loc[high_users]
                    predictions = np.array([list(map(int,s.split())) for s in high.recommendation_internal_ids])
                    assert all(len(set(r))==10 and not (set(r)&info[u]['seen']) for u,r in zip(high_users,predictions))
                    metric, recall, part, item_part, hits = contributions(predictions, relevant, groups, len(catalog))
                    np.testing.assert_allclose(metric, high.ndcg10_high, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(recall, high.recall10_high, atol=1e-12, rtol=0)
                    np.testing.assert_array_equal(high.history_length, feature.history_length)
                    np.testing.assert_array_equal(high.context_length, feature['delete_context_length' if method=='DELETE10' else 'context_length'])
                    values.append(np.column_stack([metric,recall])); parts.append(part)
                    per_item.append(item_part); per_item_hits.append(hits)
                    recommendation_rows.append({'split':split,'method':method,'seed':seed,
                        'users':len(high_users),'catalog_size':len(catalog),
                        'coverage_items':int(len(np.unique(predictions))),
                        'catalog_coverage':float(len(np.unique(predictions))/len(catalog)),
                        'mean_log1p_training_frequency':float(np.log1p(frequency[predictions]).mean()),
                        **{'recommendation_share_'+POP_GROUPS[g]:float((groups[predictions]==g).mean()) for g in range(3)}})
                mean[method] = np.mean(values, axis=0)
                components[method] = np.mean(parts, axis=0)
                item_components[method] = np.mean(per_item, axis=0)
                item_hit_counts[method] = np.mean(per_item_hits, axis=0)
                mean[method+'_seeds'] = np.stack(values,axis=1)
            # Filled below: subgroup contrasts, film contributions and composition checks.
            summarize_split(split, high_users, feature, mean, components, item_components,
                            item_hit_counts, catalog, groups, target_counts, training, items,
                            subgroup_rows, film_rows, item_rows, overall_rows)
            print('Verified and analysed:',split,len(high_users),'users',flush=True)
    save_outputs(args, plan, overlap, subgroup_rows, film_rows, item_rows,
                 recommendation_rows, overall_rows)


def summarize_split(split, users, feature, mean, components, item_components, item_hits,
                    catalog, groups, target_counts, training, items,
                    subgroup_rows, film_rows, item_rows, overall_rows):
    delta = mean['SOFT']-mean['DELETE10']
    low, high = interval(delta)
    overall = {'split':split,'users':len(users)}
    for k,metric in enumerate(['ndcg','recall']):
        overall.update({metric+'_difference':float(delta[:,k].mean()),
                        metric+'_low':float(low[k]),metric+'_high':float(high[k])})
        for method in METHODS: overall[method+'_'+metric] = float(mean[method][:,k].mean())
    overall_rows.append(overall)
    for axis,labels in AXES.items():
        recomposed = 0.
        for g,label in enumerate(labels):
            mask = feature[axis].eq(g).to_numpy()
            if not mask.any():
                subgroup_rows.append({'split':split,'axis':axis,'group_index':g,'group':label,'users':0,'share':0.})
                continue
            lo, hi = interval(delta[mask])
            row = {'split':split,'axis':axis,'group_index':g,'group':label,
                   'users':int(mask.sum()),'share':float(mask.mean()),'small_group':bool(mask.sum()<100),
                   'mean_full_history':float(feature.loc[mask,'history_length'].mean()),
                   'mean_weak_share':float((feature.loc[mask,'weak_count']/feature.loc[mask,'context_length']).mean()),
                   'mean_delete_context_length':float(feature.loc[mask,'delete_context_length'].mean())}
            for k,metric in enumerate(['ndcg','recall']):
                row.update({metric+'_difference':float(delta[mask,k].mean()),
                            metric+'_low':float(lo[k]),metric+'_high':float(hi[k])})
                for method in METHODS: row[method+'_'+metric] = float(mean[method][mask,k].mean())
            seed_delta = (mean['SOFT_seeds'][mask,:,0]-mean['DELETE10_seeds'][mask,:,0]).mean(axis=0)
            row.update({'ndcg_difference_seed'+str(seed):float(v) for seed,v in zip(SEEDS,seed_delta)})
            row['seed_difference_std'] = float(seed_delta.std(ddof=1))
            row['positive_seed_differences'] = int((seed_delta>0).sum())
            subgroup_rows.append(row)
            recomposed += mask.mean()*delta[mask,0].mean()
        np.testing.assert_allclose(recomposed, delta[:,0].mean(), atol=1e-15)
    difference_components = components['SOFT']-components['DELETE10']
    lo, hi = interval(difference_components)
    for g,label in enumerate(POP_GROUPS):
        selected = groups[1:]==g
        row = {'split':split,'group_index':g,'group':label,'users_in_common_population':len(users),
               'catalog_films':int(selected.sum()),
               'future_targets':int(target_counts[1:][selected].sum()),
               'future_target_share':float(target_counts[1:][selected].sum()/target_counts.sum()),
               'ndcg_contribution_difference':float(difference_components[:,g].mean()),
               'low':float(lo[g]),'high':float(hi[g])}
        for method in METHODS:
            row[method+'_ndcg_contribution'] = float(components[method][:,g].mean())
            row[method+'_mean_hit_count'] = float(item_hits[method][1:][selected].sum())
        film_rows.append(row)
    np.testing.assert_allclose(difference_components.sum(axis=1),delta[:,0],atol=1e-15)
    windows = training.groupby('user_id',sort=False).tail(20).copy()
    windows['weight'] = .2+.8*windows.watched_pct/100
    deleted = training.loc[training.watched_pct.gt(10)]
    retained = deleted.groupby('user_id').size()
    deleted = deleted.loc[deleted.user_id.isin(retained.loc[retained.ge(2)].index)].groupby('user_id',sort=False).tail(20)
    item_frame = pd.DataFrame({'item_id':catalog,'training_frequency':training.groupby('item_id').size().reindex(catalog).to_numpy(),
                              'future_high_target_users':target_counts[1:],
                              'ALL_window_positions':windows.groupby('item_id').size().reindex(catalog,fill_value=0).to_numpy(),
                              'DELETE10_window_positions':deleted.groupby('item_id').size().reindex(catalog,fill_value=0).to_numpy(),
                              'SOFT_window_weight_sum':windows.groupby('item_id').weight.sum().reindex(catalog,fill_value=0).to_numpy()})
    item_frame = item_frame.merge(items[['item_id','title_orig','release_year']],on='item_id',validate='one_to_one')
    item_frame['split'] = split
    for method in METHODS:
        item_frame[method+'_ndcg_contribution'] = item_components[method][1:]
        item_frame[method+'_mean_hit_count'] = item_hits[method][1:]
    item_frame['ndcg_contribution_difference'] = item_frame.SOFT_ndcg_contribution-item_frame.DELETE10_ndcg_contribution
    np.testing.assert_allclose(item_frame.ndcg_contribution_difference.sum(),delta[:,0].mean(),atol=1e-15)
    item_rows.extend(item_frame.to_dict('records'))


def save_outputs(args, plan, overlap, subgroup_rows, film_rows, item_rows, recommendation_rows, overall_rows):
    out = args.output_dir
    for name, rows in [('user_subgroups',subgroup_rows),('film_popularity_contributions',film_rows),
                       ('film_contributions',item_rows),('recommendation_properties',recommendation_rows),
                       ('overall_SOFT_DELETE10',overall_rows)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    subgroup = pd.DataFrame(subgroup_rows)
    weak = subgroup.loc[subgroup.axis.eq('weak_share')].set_index(['split','group_index'])
    a,b = weak.loc['main'].sort_index(),weak.loc['replication'].sort_index()
    weight_a,weight_b = a.share.to_numpy(),b.share.to_numpy()
    effect_a,effect_b = a.ndcg_difference.to_numpy(),b.ndcg_difference.to_numpy()
    composition = ((weight_a-weight_b)*(effect_a+effect_b)/2).sum()
    within = ((effect_a-effect_b)*(weight_a+weight_b)/2).sum()
    overall = pd.DataFrame(overall_rows).set_index('split')
    shift = overall.loc['main','ndcg_difference']-overall.loc['replication','ndcg_difference']
    np.testing.assert_allclose(composition+within,shift,atol=1e-15)
    pd.DataFrame([{'axis':'weak_share','change_main_minus_replication':shift,
                   'composition_component':composition,'within_group_component':within,
                   'scope':'arithmetic_decomposition_not_causal'}]).to_csv(out/'composition_decomposition.csv',index=False)
    items = pd.DataFrame(item_rows)
    tops, concentration = [], {}
    for split in ['main','replication']:
        part = items.loc[items.split.eq(split)].copy()
        part['absolute_contribution_difference'] = part.ndcg_contribution_difference.abs()
        top = part.nlargest(5,'absolute_contribution_difference')
        tops.append(top)
        concentration[split] = {'top5_absolute_share':float(top.absolute_contribution_difference.sum()/part.absolute_contribution_difference.sum()),
                                'top5_signed_sum':float(top.ndcg_contribution_difference.sum()),
                                'films_with_high_targets':int(part.future_high_target_users.gt(0).sum())}
    pd.concat(tops,ignore_index=True).to_csv(out/'top5_absolute_film_contributions.csv',index=False)
    summary = {'status':'passed','scope':'exploratory_subgroup_and_additive_item_analysis',
               'archive_sha256':sha(args.archive),'source_archive_sha256':sha(args.data_archive),
               'design_hash':plan['design_hash'],'new_training':False,
               'methods':METHODS,'seeds':SEEDS,'user_subgroup_rows':len(subgroup_rows),
               'all_group_partitions_reconstruct_overall_difference':True,
               'item_groups_and_individual_items_reconstruct_NDCG':True,
               'saved_user_metrics_recomputed_from_ranked_ids_and_raw_targets':True,
               'features_precede_test':True,'intervals':'descriptive_conditional_95_percent_5000_repetitions_no_multiplicity_correction',
               'cohort_overlap':len(overlap['main']['cohort'] & overlap['replication']['cohort']),
               'high_test_population_overlap':len(overlap['main']['test_high'] & overlap['replication']['test_high']),
               'film_concentration':concentration,
               'composition_decomposition':{'main_minus_replication':float(shift),'composition':float(composition),'within_group':float(within)}}
    write_json(out/'posthoc_validation.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
