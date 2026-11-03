"""Standalone KION study 2: film watch-percentage handling, five policies, two cuts.

The notebook embeds this file verbatim. No imports from article-1 implementations.
"""
# %% runtime
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')
import hashlib
import io
import json
import math
import platform
import random
import shutil
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'kion-watch-feedback-1.0'
ARCHIVE_SHA256 = 'fddc5b2215bfff49bcfa21e464441f1fc6dc30f8f4afe41d3f23bf3fc45f3542'
SOURCE_URL = 'https://raw.githubusercontent.com/irsafilo/KION_DATASET/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_en.zip'
METHODS = ['ALL', 'DELETE10', 'TARGET10', 'STEP10', 'SOFT']
SEEDS = [17, 29, 43]
CUTS = [('main', '2021-08-01', '2021-08-08', '2021-08-15'),
        ('replication', '2021-07-18', '2021-07-25', '2021-08-01')]
EXPECTED = {
    'main': {'eligible_users': 81545, 'train_pairs': 1085046, 'catalog_items': 10131,
             'input_positions': 783385, 'validation_users_any': 15523, 'validation_users_high': 8471},
    'replication': {'eligible_users': 66620, 'train_pairs': 1069327, 'catalog_items': 10298,
                    'input_positions': 780135, 'validation_users_any': 16058, 'validation_users_high': 9650},
}

@dataclass(frozen=True)
class WatchConfig:
    n_users: int = 50000
    min_history: int = 10
    sample_seed: int = 20261031
    sequence_length: int = 20
    hidden_size: int = 64
    layers: int = 2
    heads: int = 2
    dropout: float = .1
    mask_probability: float = .35
    weak_threshold: float = 10.0
    high_threshold: float = 50.0
    soft_floor: float = .2
    batch_size: int = 128
    eval_batch_size: int = 128
    learning_rate: float = .001
    weight_decay: float = .01
    gradient_clip: float = 5.0
    max_steps: int = 20000
    eval_every: int = 1000
    checkpoint_every: int = 1000
    bootstrap_reps: int = 5000
    bootstrap_seed: int = 20261031
    technical_mode: bool = False

    def validate(self):
        if not self.technical_mode and self != WatchConfig():
            raise ValueError('Научные настройки отличаются от согласованного протокола. Не меняйте их для продолжения серии.')
        if self.hidden_size % self.heads or not 0 < self.mask_probability < 1:
            raise ValueError('Неверные размерности или вероятность маскирования.')
        if min(self.n_users, self.batch_size, self.eval_batch_size, self.max_steps,
               self.eval_every, self.checkpoint_every, self.bootstrap_reps) < 1 or self.sequence_length < 2:
            raise ValueError('Неверный бюджет или длина входа.')
        return self

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def array_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    (rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)).to_csv(temporary, index=False)
    temporary.replace(path)

def save_torch(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)

def load_torch(path):
    # All stored states contain only tensors and primitive containers. Never unpickle arbitrary objects.
    return torch.load(path, map_location='cpu', weights_only=True)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

def select_device(cfg):
    cfg.validate()
    if not torch.cuda.is_available() and not cfg.technical_mode:
        raise RuntimeError('GPU не обнаружена. В Kaggle включите Accelerator → GPU и перезапустите сессию.')
    return torch.device('cuda:0' if torch.cuda.is_available() and not cfg.technical_mode else 'cpu')

def environment(device):
    return {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
            'pandas': pd.__version__, 'cuda': torch.version.cuda, 'device': device.type,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'precision': 'float32', 'torch_cpu_threads': torch.get_num_threads(), 'implementation': VERSION}

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def acquire_archive(cache, explicit=''):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = Path(explicit) if explicit else cache / 'data_en.zip'
    if not path.exists():
        if explicit:
            raise FileNotFoundError('Указанный архив KION не найден: ' + str(path))
        temporary = path.with_suffix('.download')
        print('Скачивание закреплённого архива KION…', flush=True)
        urllib.request.urlretrieve(SOURCE_URL, temporary)
        if file_hash(temporary) != ARCHIVE_SHA256:
            raise ValueError('Контрольная сумма скачанного KION не совпала.')
        temporary.replace(path)
    if file_hash(path) != ARCHIVE_SHA256:
        raise ValueError('Нужен официальный архив data_en.zip закреплённой ревизии; этот файл отличается.')
    return path

# %% data
@dataclass
class EvalExample:
    user_id: int
    history: list
    delete_history: list
    seen: list
    targets_any: list
    targets_high: list
    history_length: int
    target_day: str

@dataclass
class SplitData:
    name: str
    train_end: str
    validation_end: str
    test_end: str
    users: np.ndarray
    item_ids: np.ndarray
    tokens: dict
    percentages: dict
    training_indices: dict
    validation: list
    manifest: dict
    rows: pd.DataFrame = field(repr=False)

def read_films(archive):
    with zipfile.ZipFile(archive) as z:
        with z.open('data_en/items_en.csv') as f:
            items = pd.read_csv(f, usecols=['item_id', 'content_type'])
        with z.open('data_en/interactions.csv') as f:
            rows = pd.read_csv(f, dtype={'user_id': 'int32', 'item_id': 'int32', 'watched_pct': 'float32'})
    if not items.item_id.is_unique or rows.duplicated(['user_id', 'item_id']).any():
        raise ValueError('Нарушена уникальность объектов или агрегированных пар пользователь–объект.')
    film_ids = items.loc[items.content_type.eq('film'), 'item_id']
    films = rows.loc[rows.item_id.isin(film_ids)].copy()
    if (films.watched_pct.dropna().lt(0) | films.watched_pct.dropna().gt(100)).any():
        raise ValueError('Доля просмотра вне 0–100: требуется проверка исходных данных.')
    return films.sort_values(['user_id', 'last_watch_dt', 'item_id']).reset_index(drop=True)

def pack_histories(frame, users, item_index, sequence_length):
    tokens = np.zeros((len(users), sequence_length), dtype=np.int64)
    percentages = np.zeros((len(users), sequence_length), dtype=np.float32)
    user_index = {int(u): i for i, u in enumerate(users)}
    latest = frame.groupby('user_id', sort=False).tail(sequence_length)
    for user_id, group in latest.groupby('user_id', sort=False):
        i, n = user_index[int(user_id)], len(group)
        tokens[i, :n] = [item_index[int(item)] for item in group.item_id]
        percentages[i, :n] = group.watched_pct.to_numpy(dtype=np.float32)
    return tokens, percentages

def make_examples(rows, users, item_ids, start, end, cfg):
    item_index = {int(item): i + 1 for i, item in enumerate(item_ids)}
    past_raw = rows.loc[rows.last_watch_dt.le(start)]
    past = past_raw.loc[past_raw.watched_pct.notna() & past_raw.item_id.isin(item_index)]
    histories, deleted, seen = {}, {}, {}
    for user, group in past.groupby('user_id', sort=False):
        mapped = [item_index[int(i)] for i in group.item_id]
        histories[int(user)] = mapped
        deleted[int(user)] = [item_index[int(i)] for i in group.loc[group.watched_pct.gt(cfg.weak_threshold), 'item_id']]
    for user, group in past_raw.groupby('user_id', sort=False):
        seen[int(user)] = sorted({item_index[int(i)] for i in group.item_id if int(i) in item_index})
    future = rows.loc[rows.last_watch_dt.gt(start) & rows.last_watch_dt.le(end)]
    first_day = future.groupby('user_id').last_watch_dt.transform('min')
    first = future.loc[future.last_watch_dt.eq(first_day)]
    result, lost = [], 0
    for user, group in first.groupby('user_id', sort=True):
        user = int(user)
        known = group.loc[group.item_id.isin(item_index) & group.watched_pct.notna()]
        already = set(seen.get(user, []))
        any_targets = sorted({item_index[int(i)] for i in known.item_id} - already)
        high_targets = sorted({item_index[int(i)] for i in known.loc[known.watched_pct.ge(cfg.high_threshold), 'item_id']} - already)
        if not any_targets:
            lost += 1
            continue
        hist, dhist = histories.get(user, []), deleted.get(user, [])
        result.append(EvalExample(user, hist[-(cfg.sequence_length - 1):], dhist[-(cfg.sequence_length - 1):],
                                  seen.get(user, []), any_targets, high_targets, len(hist), str(group.last_watch_dt.iloc[0])))
    if not result or not any(x.targets_high for x in result):
        raise ValueError('Нет общей оценочной когорты или целей высокой доли просмотра.')
    manifest = {'start_exclusive': start, 'end_inclusive': end,
                'users_with_raw_future_films': int(future.user_id.nunique()), 'users_without_known_targets': lost,
                'users_any': len(result), 'users_high': sum(bool(x.targets_high) for x in result),
                'targets_any': sum(len(x.targets_any) for x in result),
                'targets_high': sum(len(x.targets_high) for x in result),
                'fingerprint': canonical([asdict(x) for x in result])}
    return result, manifest

def prepare_split(films, cut, cfg):
    cfg.validate()
    name, train_end, validation_end, test_end = cut
    valid_train = films.loc[films.last_watch_dt.le(train_end) & films.watched_pct.notna()]
    counts = valid_train.groupby('user_id').size()
    eligible = np.sort(counts.loc[counts.ge(cfg.min_history)].index.to_numpy())
    if len(eligible) < cfg.n_users:
        raise ValueError('Недостаточно пользователей с принятой историей; минимум автоматически не понижается.')
    users = np.sort(np.random.default_rng(cfg.sample_seed).choice(eligible, cfg.n_users, replace=False))
    rows = films.loc[films.user_id.isin(users)].copy()
    training = valid_train.loc[valid_train.user_id.isin(users)]
    item_ids = np.sort(training.item_id.unique())
    item_index = {int(item): i + 1 for i, item in enumerate(item_ids)}
    tokens, percentages, indices = {}, {}, {}
    for key, subset in [('all', training), ('delete', training.loc[training.watched_pct.gt(cfg.weak_threshold)])]:
        tokens[key], percentages[key] = pack_histories(subset, users, item_index, cfg.sequence_length)
        lengths = np.count_nonzero(tokens[key], axis=1)
        indices[key] = np.flatnonzero(lengths >= 2)
        if not len(indices[key]):
            raise ValueError('После фильтрации нет обучающих историй с целью и видимым объектом.')
    validation, val_manifest = make_examples(rows, users, item_ids, train_end, validation_end, cfg)
    stats = {'eligible_users': len(eligible), 'selected_users': len(users), 'train_pairs': len(training),
             'catalog_items': len(item_ids), 'input_positions': int(np.count_nonzero(tokens['all'])),
             'validation_users_any': len(validation), 'validation_users_high': sum(bool(x.targets_high) for x in validation),
             'delete_training_users': len(indices['delete']), 'delete_removed_training_users': len(users) - len(indices['delete']),
             'weak_input_fraction': float((percentages['all'][tokens['all'] > 0] <= cfg.weak_threshold).mean())}
    if not cfg.technical_mode:
        for key, expected in EXPECTED[name].items():
            if stats[key] != expected:
                raise ValueError(f'Срез {name}, {key}: {stats[key]} вместо {expected}. Обучение не начато.')
    manifest = {'split': name, 'train_end': train_end, 'validation_end': validation_end, 'test_end': test_end,
                'stats': stats, 'validation': val_manifest,
                'array_hashes': {'users': array_hash(users), 'items': array_hash(item_ids),
                     **{k + '_tokens': array_hash(a) for k, a in tokens.items()},
                     **{k + '_pct': array_hash(a) for k, a in percentages.items()}},
                'source_archive_sha256': ARCHIVE_SHA256}
    manifest['fingerprint'] = canonical(manifest)
    return SplitData(name, train_end, validation_end, test_end, users, item_ids, tokens, percentages,
                     indices, validation, manifest, rows)

def make_plan(data, cfg, device, code_hash):
    plan = {'version': VERSION, 'mode': 'technical_smoke_not_scientific' if cfg.technical_mode else 'scientific',
            'config': asdict(cfg), 'methods': METHODS, 'seeds': SEEDS, 'cuts': [list(c) for c in CUTS],
            'code_sha256': code_hash, 'source_archive_sha256': ARCHIVE_SHA256,
            'environment': environment(device), 'data': {k: v.manifest for k, v in data.items()},
            'models': len(data) * len(METHODS) * len(SEEDS),
            'checkpoint_selection': 'validation_ndcg10_high; first maximum; no test selection',
            'primary_contrasts': [[s, 'SOFT', right, 'ndcg10_high'] for s in data for right in ['ALL', 'STEP10']],
            'primary_interval_level': .9875, 'other_intervals': 'descriptive_95_percent',
            'test_use': 'fixed_protocol; KION previously used for another study; no claim of independent dataset',
            'loss': 'sum(w*CE)/sum(w); mask p=.35; ZERO and missing pct are different',
            'limits': 'aggregated last-date film interactions; conditional user bootstrap; three training seeds'}
    return {'plan_hash': canonical(plan), 'plan': plan}

def queue(data):
    return [(split, method, seed) for seed in SEEDS for method in METHODS for split in data]

def run_name(split, method, seed):
    return f'{split}__{method}__seed{seed}'

def freeze_plan(root, frozen, data):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'watch_study_plan.json'
    if target.exists():
        old = json.loads(target.read_text(encoding='utf-8'))
        if old != frozen:
            differences = [k for k in frozen['plan'] if old.get('plan', {}).get(k) != frozen['plan'][k]]
            raise ValueError('Сохранённая серия отличается: ' + ', '.join(differences) + '. Проверку не отключайте.')
    else:
        if any((root / folder).exists() and any((root / folder).iterdir()) for folder in ['runs', 'test_runs']):
            raise ValueError('Найдены результаты без корневого плана; автоматическое смешивание запрещено.')
        save_json(target, frozen)
    for name, split in data.items():
        folder = root / 'data' / name
        folder.mkdir(parents=True, exist_ok=True)
        save_json(folder / 'manifest.json', split.manifest)
        np.savez_compressed(folder / 'selection_catalog.npz', users=split.users, item_ids=split.item_ids)
    return root

def restore_folder(destination, frozen, source='', search_root='/kaggle/input'):
    destination = Path(destination)
    if (destination / 'watch_study_plan.json').exists():
        return None
    if source:
        candidates = [Path(source) / 'watch_study_plan.json']
        if not candidates[0].is_file():
            raise FileNotFoundError('RESUME_FROM: укажите папку серии, содержащую watch_study_plan.json и runs.')
    else:
        candidates = list(Path(search_root).rglob('watch_study_plan.json')) if Path(search_root).exists() else []
    matching = []
    for path in candidates:
        try:
            saved = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if saved == frozen:
            matching.append(path.parent)
        elif source:
            raise ValueError('Указанная папка восстановления имеет другую среду, данные, код или параметры.')
    if not matching:
        return None
    # Prefer the snapshot with the most complete models; never merge incompatible snapshots.
    chosen = max(matching, key=lambda p: (len(list((p / 'runs').glob('*/completed.json'))),
                                         len(list((p / 'test_runs').glob('*/completed.json'))), str(p)))
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Папка назначения не пуста и не имеет плана: восстановление не перезаписывает файлы.')
    shutil.copytree(chosen, destination, dirs_exist_ok=True)
    print('Восстановлены сохранённые выходы:', chosen, flush=True)
    return chosen

# %% model_training
def check_execution(cfg, frozen, device):
    if asdict(cfg) != frozen['plan']['config'] or environment(device) != frozen['plan']['environment']:
        raise ValueError('Во время серии изменены настройки или программная среда.')
    if METHODS != frozen['plan']['methods'] or SEEDS != frozen['plan']['seeds']:
        raise ValueError('Изменены согласованные методы или seed.')

class CompactBERT4Rec(nn.Module):
    """Same compact ID-only architecture as article 1; no pretrained language model."""
    def __init__(self, n_items, cfg):
        super().__init__()
        self.n_items, self.mask_id = n_items, n_items + 1
        self.item_embedding = nn.Embedding(n_items + 2, cfg.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(cfg.sequence_length, cfg.hidden_size)
        self.input_norm = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        block = nn.TransformerEncoderLayer(cfg.hidden_size, cfg.heads, 4 * cfg.hidden_size, cfg.dropout,
                    activation='gelu', batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(block, cfg.layers, enable_nested_tensor=False)
        self.output_dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.output_norm = nn.LayerNorm(cfg.hidden_size)
        self.item_bias = nn.Parameter(torch.zeros(n_items))
        for name, parameter in self.named_parameters():
            if parameter.ndim >= 2:
                nn.init.trunc_normal_(parameter, std=.02, a=-.04, b=.04)
            elif name.endswith('weight'):
                nn.init.ones_(parameter)
            else:
                nn.init.zeros_(parameter)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        h = self.dropout(self.input_norm(self.item_embedding(tokens) + self.position_embedding(positions)[None]))
        return self.encoder(h, src_key_padding_mask=tokens.eq(0), is_causal=False)

    def scores(self, h):
        h = self.output_norm(F.gelu(self.output_dense(h)))
        return F.linear(h, self.item_embedding.weight[1:self.n_items + 1], self.item_bias)

    def masked_logits(self, tokens, mask):
        return self.scores(self.encode(tokens)[mask])

    def next_scores(self, tokens, positions):
        h = self.encode(tokens)
        return self.scores(h[torch.arange(len(tokens), device=tokens.device), positions])

def target_weights(pct, method, cfg):
    if method in ['ALL', 'DELETE10']:
        return torch.ones_like(pct)
    if method == 'TARGET10':
        return pct.gt(cfg.weak_threshold).to(pct.dtype)
    if method == 'STEP10':
        return torch.where(pct.gt(cfg.weak_threshold), 3.0, 1.0)
    if method == 'SOFT':
        return cfg.soft_floor + (1 - cfg.soft_floor) * pct / 100
    raise ValueError('Неизвестный метод: ' + method)

def masked_batch(clean, pct, mask_id, cfg, seed, draw):
    generator = torch.Generator(device='cpu').manual_seed(seed * 1000003 + draw)
    real = clean.ne(0)
    lengths = real.sum(dim=1)
    if lengths.min() < 2 or not torch.isfinite(pct[real]).all() or ((pct[real] < 0) | (pct[real] > 100)).any():
        raise ValueError('Неверная история или доля просмотра в обучающем пакете.')
    selected = torch.rand(clean.shape, generator=generator).lt(cfg.mask_probability) & real
    for row, count in enumerate(selected.sum(dim=1).tolist()):
        n = int(lengths[row])
        if count == 0:
            selected[row, int(torch.randint(n, (1,), generator=generator))] = True
        elif count == n:
            selected[row, int(torch.randint(n, (1,), generator=generator))] = False
    corrupted = clean.clone()
    corrupted[selected] = mask_id
    return corrupted, selected, clean[selected] - 1, pct[selected]

def weighted_loss(logits, labels, weights):
    if weights.sum().item() <= 0:
        return None
    return (F.cross_entropy(logits, labels, reduction='none') * weights).sum() / weights.sum()

def metrics(recommended, relevant):
    if not relevant:
        return None, None
    rel = set(relevant)
    hits = [int(i in rel) for i in recommended[:10]]
    dcg = sum(x / math.log2(i + 2) for i, x in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(rel))))
    return dcg / ideal, sum(hits) / len(rel)

def aggregate_users(rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    return {'users_any': len(frame), 'users_high': int(frame.has_high.sum()),
            **{m: float(frame[m].mean()) for m in ['ndcg10_any', 'recall10_any', 'ndcg10_high', 'recall10_high']}}

@torch.inference_mode()
def evaluate(model, examples, method, cfg, device):
    model.eval()
    rows = []
    for offset in range(0, len(examples), cfg.eval_batch_size):
        batch = examples[offset:offset + cfg.eval_batch_size]
        tokens = torch.zeros((len(batch), cfg.sequence_length), dtype=torch.long)
        positions = []
        for row, ex in enumerate(batch):
            h = ex.delete_history if method == 'DELETE10' else ex.history
            tokens[row, :len(h) + 1] = torch.tensor(h + [model.mask_id])
            positions.append(len(h))
        scores = model.next_scores(tokens.to(device), torch.tensor(positions, device=device))
        for row, ex in enumerate(batch):
            if ex.seen:
                scores[row, torch.tensor(ex.seen, device=device) - 1] = -torch.inf
        order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :min(10, model.n_items)]
        finite = scores.gather(1, order).isfinite().cpu().tolist()
        for ex, ids, allowed, context_length in zip(batch, order.cpu().tolist(), finite, positions):
            rec = [i + 1 for i, ok in zip(ids, allowed) if ok]
            na, ra = metrics(rec, ex.targets_any)
            nh, rh = metrics(rec, ex.targets_high)
            rows.append({'user_id': ex.user_id, 'history_length': ex.history_length, 'context_length': context_length,
                         'has_high': bool(ex.targets_high), 'targets_any': len(ex.targets_any), 'targets_high': len(ex.targets_high),
                         'ndcg10_any': na, 'recall10_any': ra, 'ndcg10_high': nh, 'recall10_high': rh,
                         'recommendation_internal_ids': ' '.join(map(str, rec))})
    return aggregate_users(rows), pd.DataFrame(rows)

def expected_identity(split, method, seed, frozen):
    return {'plan_hash': frozen['plan_hash'], 'split': split.name, 'method': method, 'seed': seed,
            'data_fingerprint': split.manifest['fingerprint']}

def verify_completed(folder, identity):
    folder = Path(folder)
    marker = folder / 'completed.json'
    if not marker.exists():
        return None
    record = json.loads(marker.read_text(encoding='utf-8'))
    if record['identity'] != identity:
        raise ValueError('Несовместимый завершённый запуск: ' + str(folder))
    for name, digest in record['checksums'].items():
        if not (folder / name).is_file() or file_hash(folder / name) != digest:
            raise ValueError('Файл завершённого запуска отсутствует или изменён: ' + str(folder / name))
    return record

def train_run(split, method, seed, cfg, root, frozen, device, stop_after_updates=0):
    check_execution(cfg, frozen, device)
    folder = Path(root) / 'runs' / run_name(split.name, method, seed)
    identity = expected_identity(split, method, seed, frozen)
    if verify_completed(folder, identity):
        return True
    if (Path(root) / 'test_selection_lock.json').exists():
        raise ValueError('Тест уже зафиксирован: отсутствующую модель нельзя заменить новым обучением.')
    folder.mkdir(parents=True, exist_ok=True)
    config_record = {'identity': identity, 'config': asdict(cfg), 'environment': environment(device)}
    config_path = folder / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config_record:
        raise ValueError('Изменена конфигурация незавершённого запуска.')
    save_json(config_path, config_record)
    seed_everything(seed)
    model = CompactBERT4Rec(len(split.item_ids), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    step, draw, best_score, best_step = 0, 0, None, 0
    training_seconds, evaluation_seconds, checkpoint_seconds = 0.0, 0.0, 0.0
    history, best_metrics = [], None
    totals = {'real_positions': 0, 'masked_positions': 0, 'positive_weight_targets': 0,
              'weight_sum': 0.0, 'weight_square_sum': 0.0, 'zero_weight_batches': 0, 'weighted_loss_sum': 0.0}
    if (folder / 'last.pt').exists():
        state = load_torch(folder / 'last.pt')
        if state['identity'] != identity:
            raise ValueError('last.pt относится к другой серии.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        step, draw, best_score, best_step = [state[k] for k in ['step', 'draw', 'best_score', 'best_step']]
        training_seconds, evaluation_seconds, checkpoint_seconds = [state[k] for k in ['training_seconds', 'evaluation_seconds', 'checkpoint_seconds']]
        history, best_metrics, totals = state['history'], state['best_metrics'], state['totals']
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        best_candidate = folder / f'candidate_step_{best_step}.pt'
        if best_score is not None and (not best_candidate.exists() or file_hash(best_candidate) != state['best_sha256']):
            raise ValueError('Не совпадает лучший checkpoint незавершённого запуска.')
        print(f'Продолжение {folder.name}: шаг {step}/{cfg.max_steps}', flush=True)
    else:
        print('Обучение:', folder.name, flush=True)

    def checkpoint():
        nonlocal checkpoint_seconds
        t = time.perf_counter()
        save_torch(folder / 'last.pt', {'identity': identity, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'step': step, 'draw': draw, 'best_score': best_score,
            'best_step': best_step, 'history': history, 'best_metrics': best_metrics, 'totals': totals,
            'training_seconds': training_seconds, 'evaluation_seconds': evaluation_seconds,
            'checkpoint_seconds': checkpoint_seconds, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'best_sha256': file_hash(folder / f'candidate_step_{best_step}.pt') if best_score is not None else None})
        save_csv(folder / 'history.csv', history)
        # Old best snapshots are removed only AFTER last.pt atomically commits the new pointer.
        for pattern, keep in [('candidate_step_*.pt', f'candidate_step_{best_step}.pt'),
                              ('candidate_users_*.csv', f'candidate_users_{best_step}.csv')]:
            for candidate in folder.glob(pattern):
                if candidate.name != keep:
                    candidate.unlink()
        checkpoint_seconds += time.perf_counter() - t

    key = 'delete' if method == 'DELETE10' else 'all'
    valid_indices = split.training_indices[key]
    tokens, pct = split.tokens[key], split.percentages[key]
    start_step = step
    if not (folder / 'last.pt').exists():
        checkpoint()
    try:
        while step < cfg.max_steps:
            model.train()
            sync(device)
            t = time.perf_counter()
            draw += 1
            chosen = np.random.default_rng(seed * 1000003 + draw).choice(valid_indices, cfg.batch_size, replace=True)
            clean, watch = torch.from_numpy(tokens[chosen]), torch.from_numpy(pct[chosen])
            corrupted, selected, labels, target_pct = masked_batch(clean, watch, model.mask_id, cfg, seed, draw)
            weights = target_weights(target_pct, method, cfg)
            if weights.sum().item() == 0:
                totals['zero_weight_batches'] += 1
                training_seconds += time.perf_counter() - t
                if totals['zero_weight_batches'] > 1000:
                    raise ValueError('Слишком много пакетов без ненулевых целей; проверьте данные.')
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(model.masked_logits(corrupted.to(device), selected.to(device)), labels.to(device), weights.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Функция потерь не конечна.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            sync(device)
            training_seconds += time.perf_counter() - t
            step += 1
            totals['real_positions'] += int(clean.ne(0).sum())
            totals['masked_positions'] += int(selected.sum())
            totals['positive_weight_targets'] += int(weights.gt(0).sum())
            totals['weight_sum'] += float(weights.sum())
            totals['weight_square_sum'] += float(weights.square().sum())
            totals['weighted_loss_sum'] += float(loss.detach())
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                sync(device)
                t = time.perf_counter()
                val, users = evaluate(model, split.validation, method, cfg, device)
                sync(device)
                evaluation_seconds += time.perf_counter() - t
                history.append({'step': step, 'training_seconds': training_seconds, 'evaluation_seconds': evaluation_seconds,
                                'mean_training_loss': totals['weighted_loss_sum'] / step, **val})
                if best_score is None or val['ndcg10_high'] > best_score:
                    best_score, best_step, best_metrics = val['ndcg10_high'], step, val
                    t = time.perf_counter()
                    save_torch(folder / f'candidate_step_{step}.pt', {'identity': identity, 'model': model.state_dict(), 'step': step,
                                                  'validation': val})
                    save_csv(folder / f'candidate_users_{step}.csv', users)
                    checkpoint_seconds += time.perf_counter() - t
                print(f'{folder.name}: {step}/{cfg.max_steps}; val NDCG высокой доли={val["ndcg10_high"]:.5f}', flush=True)
            if step % cfg.checkpoint_every == 0 or step == cfg.max_steps:
                checkpoint()
            if stop_after_updates and step - start_step >= stop_after_updates and step < cfg.max_steps:
                checkpoint()
                return False
    except KeyboardInterrupt:
        # A signal can arrive between optimizer.step and counters; use the last completed atomic checkpoint.
        print('Остановка. Сохранённый last.pt позволит повторить не более одного интервала шагов.', flush=True)
        return False
    if best_metrics is None:
        raise ValueError('Не выполнена валидация; завершение не подтверждено.')
    shutil.copyfile(folder / f'candidate_step_{best_step}.pt', folder / 'best.pt')
    shutil.copyfile(folder / f'candidate_users_{best_step}.csv', folder / 'best_validation_per_user.csv')
    result = {'identity': identity, 'completed_steps': step, 'best_step': best_step,
              'training_seconds': training_seconds, 'evaluation_seconds': evaluation_seconds,
              'checkpoint_seconds': checkpoint_seconds, 'best_validation': best_metrics,
              'actual_mask_fraction': totals['masked_positions'] / totals['real_positions'],
              'effective_weighted_targets': totals['weight_sum'] ** 2 / totals['weight_square_sum'],
              'training_users': int(len(valid_indices)), 'totals': totals, 'mode': frozen['plan']['mode']}
    save_json(folder / 'summary.json', result)
    required = ['best.pt', 'last.pt', 'summary.json', 'config.json', 'history.csv', 'best_validation_per_user.csv']
    save_json(folder / 'completed.json', {'identity': identity, 'completed_steps': step,
                                        'checksums': {name: file_hash(folder / name) for name in required}})
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return True

def run_study(data, cfg, root, frozen, device, max_runs=0):
    check_execution(cfg, frozen, device)
    done, trained = 0, 0
    jobs = queue(data)
    for split_name, method, seed in jobs:
        split = data[split_name]
        folder = Path(root) / 'runs' / run_name(split_name, method, seed)
        identity = expected_identity(split, method, seed, frozen)
        if verify_completed(folder, identity):
            done += 1
            continue
        if (Path(root) / 'test_selection_lock.json').exists():
            raise ValueError('После фиксации теста отсутствует завершённая модель. Нужны исходные сохранённые выходы.')
        if max_runs and trained >= max_runs:
            break
        if not train_run(split, method, seed, cfg, root, frozen, device):
            save_json(Path(root) / 'progress.json', {'plan_hash': frozen['plan_hash'], 'completed_models': done,
                       'total_models': len(jobs), 'next_run': folder.name, 'status': 'training_incomplete'})
            return False
        trained += 1
        done += 1
        save_json(Path(root) / 'progress.json', {'plan_hash': frozen['plan_hash'], 'completed_models': done,
                   'total_models': len(jobs), 'last_completed': folder.name, 'status': 'training_complete' if done == len(jobs) else 'training'})
    return done == len(jobs)

def benchmark(data, cfg, root, frozen, device, steps=200):
    check_execution(cfg, frozen, device)
    path = Path(root) / 'benchmark.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old['plan_hash'] == frozen['plan_hash']:
            return old
    split = next(iter(data.values()))
    seed_everything(2718)
    model = CompactBERT4Rec(len(split.item_ids), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sync(device)
    start = time.perf_counter()
    for step in range(steps):
        chosen = np.random.default_rng(2718 + step).choice(split.training_indices['all'], cfg.batch_size)
        clean, pct = torch.from_numpy(split.tokens['all'][chosen]), torch.from_numpy(split.percentages['all'][chosen])
        x, mask, y, q = masked_batch(clean, pct, model.mask_id, cfg, 2718, step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_loss(model.masked_logits(x.to(device), mask.to(device)), y.to(device), torch.ones_like(q).to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()
    sync(device)
    train_sec = time.perf_counter() - start
    start = time.perf_counter()
    evaluate(model, split.validation, 'ALL', cfg, device)
    sync(device)
    val_sec = time.perf_counter() - start
    estimated = len(queue(data)) * (cfg.max_steps * train_sec / steps + math.ceil(cfg.max_steps / cfg.eval_every) * val_sec)
    result = {'plan_hash': frozen['plan_hash'], 'scope': 'timing_only_not_a_scientific_run', 'steps': steps,
              'training_seconds_per_step': train_sec / steps, 'one_validation_seconds': val_sec,
              'rough_hours_all_models': estimated / 3600,
              'estimate_excludes_setup_checkpointing_test_and_bootstrap': True}
    save_json(path, result)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result

# %% reports
def freeze_test(data, cfg, root, frozen):
    root = Path(root)
    hashes = {}
    for split_name, method, seed in queue(data):
        folder = root / 'runs' / run_name(split_name, method, seed)
        record = verify_completed(folder, expected_identity(data[split_name], method, seed, frozen))
        if not record or record['completed_steps'] != cfg.max_steps:
            raise ValueError('Тест доступен только после завершения всех согласованных моделей.')
        hashes[folder.name] = record['checksums']['best.pt']
    examples, manifests = {}, {}
    for name, split in data.items():
        examples[name], manifests[name] = make_examples(split.rows, split.users, split.item_ids,
                                                       split.validation_end, split.test_end, cfg)
    lock = {'plan_hash': frozen['plan_hash'], 'best_checkpoint_sha256': hashes, 'test_manifests': manifests,
            'selection': 'all_30_planned_models; validation_checkpoint_only'}
    path = root / 'test_selection_lock.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != lock:
        raise ValueError('После фиксации теста изменены модели или оценочная когорта.')
    save_json(path, lock)
    return examples, lock

def test_models(data, examples, cfg, root, frozen, device, lock):
    root = Path(root)
    runs = []
    for split_name, method, seed in queue(data):
        name = run_name(split_name, method, seed)
        train_folder, folder = root / 'runs' / name, root / 'test_runs' / name
        identity = expected_identity(data[split_name], method, seed, frozen)
        test_identity = {**identity, 'weight_sha256': lock['best_checkpoint_sha256'][name],
                         'test_fingerprint': lock['test_manifests'][split_name]['fingerprint']}
        marker = folder / 'completed.json'
        if marker.exists():
            cached = json.loads(marker.read_text(encoding='utf-8'))
            if cached['identity'] != test_identity:
                raise ValueError('Изменён сохранённый тест: ' + name)
            for file, digest in cached['checksums'].items():
                if not (folder / file).exists() or file_hash(folder / file) != digest:
                    raise ValueError('Повреждён тестовый файл: ' + name + '/' + file)
            test = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        else:
            seed_everything(seed)
            model = CompactBERT4Rec(len(data[split_name].item_ids), cfg).to(device)
            state = load_torch(train_folder / 'best.pt')
            if state['identity'] != identity:
                raise ValueError('Несовместимый лучший checkpoint.')
            model.load_state_dict(state['model'])
            sync(device)
            t = time.perf_counter()
            result, users = evaluate(model, examples[split_name], method, cfg, device)
            sync(device)
            test = {'identity': test_identity, 'metrics': result, 'test_seconds': time.perf_counter() - t}
            save_json(folder / 'summary.json', test)
            save_csv(folder / 'per_user.csv', users)
            save_json(marker, {'identity': test_identity, 'checksums': {f: file_hash(folder / f) for f in ['summary.json', 'per_user.csv']}})
            del model
        train = json.loads((train_folder / 'summary.json').read_text(encoding='utf-8'))
        runs.append({'split': split_name, 'method': method, 'seed': seed, 'best_step': train['best_step'],
                     **{'validation_' + k: v for k, v in train['best_validation'].items()},
                     **{'test_' + k: v for k, v in test['metrics'].items()},
                     **{k: train[k] for k in ['training_seconds', 'evaluation_seconds', 'checkpoint_seconds',
                                              'actual_mask_fraction', 'effective_weighted_targets', 'training_users']},
                     **{'total_' + k: v for k, v in train['totals'].items()}})
    result = pd.DataFrame(runs)
    save_csv(root / 'results_per_run.csv', result)
    return result

def bootstrap_interval(values, level, cfg):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Некорректные значения для парного bootstrap.')
    rng = np.random.default_rng(cfg.bootstrap_seed)
    means = np.empty(cfg.bootstrap_reps)
    for start in range(0, cfg.bootstrap_reps, 64):
        count = min(64, cfg.bootstrap_reps - start)
        draws = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[draws].mean(axis=1)
    alpha = 1 - level
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {'users': len(values), 'mean_difference': float(values.mean()), 'level': level,
            'low': float(low), 'high': float(high), 'repetitions': cfg.bootstrap_reps}

def make_intervals(data, cfg, root):
    intervals = []
    comparisons = [('SOFT', 'ALL', 'primary'), ('SOFT', 'STEP10', 'primary'),
                   ('TARGET10', 'DELETE10', 'secondary'), ('SOFT', 'DELETE10', 'secondary')]
    metrics_list = ['ndcg10_high', 'recall10_high', 'ndcg10_any', 'recall10_any']
    for split in data:
        per_user = {}
        for method in METHODS:
            frames = [pd.read_csv(Path(root) / 'test_runs' / run_name(split, method, seed) / 'per_user.csv').set_index('user_id') for seed in SEEDS]
            reference = frames[0].index
            if any(not f.index.equals(reference) for f in frames) or not reference.is_unique:
                raise ValueError('Различается порядок или состав пользователей между seed.')
            per_user[method] = sum((f[metrics_list] for f in frames)) / len(SEEDS)
        for left, right, role in comparisons:
            if not per_user[left].index.equals(per_user[right].index):
                raise ValueError('Нарушено парное сравнение пользователей.')
            for metric in metrics_list:
                delta = (per_user[left][metric] - per_user[right][metric]).dropna()
                level = .9875 if role == 'primary' and metric == 'ndcg10_high' else .95
                intervals.append({'split': split, 'left': left, 'right': right, 'metric': metric,
                                  'role': 'primary' if role == 'primary' and metric == 'ndcg10_high' else 'descriptive',
                                  **bootstrap_interval(delta.to_numpy(), level, cfg)})
    save_json(Path(root) / 'paired_intervals.json', intervals)
    save_csv(Path(root) / 'paired_intervals.csv', intervals)
    return intervals

def plot_results(table, intervals, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'path',
                         'savefig.facecolor': 'white'})
    def decimal_axis(ax):
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ticks = ax.get_xticks()
        step = abs(float(ticks[1] - ticks[0])) if len(ticks) > 1 else 1.0
        digits = max(3, int(math.ceil(-math.log10(step) - 1e-9))) if step else 3
        formatter = (lambda x, _: f'{x:.{digits}f}'.replace('.', ',')) if digits <= 6 else (lambda x, _: f'{x:.2g}'.replace('.', ','))
        ax.xaxis.set_major_formatter(FuncFormatter(formatter))
    colors = ['#245B8A', '#84919C', '#B46B30', '#7060A2', '#267151']
    for metric, stem, title in [('test_ndcg10_high', 'quality_overview', 'NDCG@10: доля просмотра ≥50%'),
                                 ('training_seconds', 'cost_overview', 'Обучение, с')]:
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), layout='constrained')
        for ax, split in zip(axes, ['main', 'replication']):
            part = table.loc[table.split.eq(split)].set_index('method').loc[METHODS]
            y = np.arange(len(METHODS))[::-1]
            for position, method, color in zip(y, METHODS, colors):
                ax.errorbar(part.loc[method, metric + '_mean'], position,
                            xerr=part.loc[method, metric + '_std'], color=color, marker='o', capsize=3, lw=1.1)
            ax.set(yticks=y, yticklabels=METHODS, ylim=(-.6, len(METHODS)-.4), xlabel=title)
            ax.set_title('Основной срез' if split == 'main' else 'Повторение во времени', loc='left', pad=8)
            ax.grid(axis='x', color='#E1E5E8', lw=.6)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            if metric.startswith('test_'):
                decimal_axis(ax)
        for ext in ['png', 'svg']:
            fig.savefig(Path(root) / (stem + '.' + ext), dpi=320, bbox_inches='tight', pad_inches=.12)
        plt.close(fig)
    # A coherent separate figure for the four prespecified primary contrasts.
    fig, ax = plt.subplots(figsize=(7.2, 3.3), layout='constrained')
    primary = [r for r in intervals if r['role'] == 'primary']
    for i, row in enumerate(primary[::-1]):
        mean = row['mean_difference']
        ax.plot([row['low'], row['high']], [i, i], color='#267151', lw=1.2)
        ax.vlines([row['low'], row['high']], i-.06, i+.06, color='#267151', lw=1.2)
        ax.plot(mean, i, color='#267151', marker='o')
    ax.set_yticks(range(len(primary)), [('Основной' if r['split']=='main' else 'Повторение') + ': ' + r['left']+' − '+r['right'] for r in primary[::-1]])
    ax.axvline(0, color='#657078', lw=.8, ls='--')
    ax.grid(axis='x', color='#E1E5E8', lw=.6)
    ax.set_xlabel('Разность NDCG@10 высокой доли; пользовательский интервал 98,75%')
    decimal_axis(ax)
    for ext in ['png', 'svg']:
        fig.savefig(Path(root) / ('primary_differences.' + ext), dpi=320, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)

def report_study(data, cfg, root, frozen, device):
    check_execution(cfg, frozen, device)
    root = Path(root)
    examples, lock = freeze_test(data, cfg, root, frozen)
    runs = test_models(data, examples, cfg, root, frozen, device, lock)
    marker = root / 'report_completed.json'
    if marker.exists():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved['plan_hash'] != frozen['plan_hash']:
            raise ValueError('Итоговый отчёт относится к другой серии.')
        for name, digest in saved['checksums'].items():
            if not (root / name).exists() or file_hash(root / name) != digest:
                raise ValueError('Изменён файл итогового отчёта: ' + name)
        return pd.read_csv(root / 'results_by_method.csv')
    numeric = [c for c in runs.select_dtypes(include='number').columns if c != 'seed']
    table = runs.groupby(['split', 'method'])[numeric].agg(['mean', 'std'])
    table.columns = ['_'.join(c) for c in table.columns]
    table = table.reset_index()
    table['seeds'] = len(SEEDS)
    save_csv(root / 'results_by_method.csv', table)
    intervals = make_intervals(data, cfg, root)
    plot_results(table, intervals, root)
    curves = []
    for split, method, seed in queue(data):
        f = pd.read_csv(root / 'runs' / run_name(split, method, seed) / 'history.csv')
        f = f.assign(split=split, method=method, seed=seed)
        curves.append(f)
    save_csv(root / 'learning_curves.csv', pd.concat(curves, ignore_index=True))
    summary = {'status': 'complete_30_model_study' if not cfg.technical_mode else 'technical_smoke_complete',
               'plan_hash': frozen['plan_hash'], 'models': len(runs), 'methods': METHODS, 'seeds': SEEDS,
               'splits': list(data), 'config': asdict(cfg), 'environment': frozen['plan']['environment'],
               'primary_comparisons': [r for r in intervals if r['role'] == 'primary'],
               'test_manifests': lock['test_manifests'], 'checkpoint_selection': 'validation_ndcg10_high',
               'interpretation': 'conditional_user_intervals; not satisfaction, causal debiasing or online revenue',
               'training_hours_without_validation': float(runs.training_seconds.sum() / 3600)}
    save_json(root / 'FINAL_SUMMARY.json', summary)
    (root / 'RESULTS_RU.md').write_text('''# Результаты исследования доли просмотра KION

Полная таблица — results_by_method.csv, отдельные модели — results_per_run.csv.
Основная метрика — test_ndcg10_high: цели с долей просмотра ≥50%.
Метрики с суффиксом _any используют все допустимые просмотры; размеры популяций различаются.
SD отражает три seed. Четыре основных пользовательских интервала имеют уровень 98,75%;
остальные 95%-интервалы описательные. Интервалы условны на обученных моделях.
SOFT — заранее заданная эвристика, не доказанная оценка истинного интереса.
Выбранные по валидации checkpoints всех 30 моделей оценены; тест не выбирает метод.
Сырые ID рекомендаций в per_user.csv внутренние, отображение — data/<split>/selection_catalog.npz.
В лёгком архиве нет весов. Сохраните исходные выходы Kaggle с папкой runs.
''', encoding='utf-8')
    required = ['results_per_run.csv', 'results_by_method.csv', 'paired_intervals.json', 'paired_intervals.csv',
                'learning_curves.csv', 'FINAL_SUMMARY.json', 'RESULTS_RU.md']
    required += [stem + '.' + ext for stem in ['quality_overview', 'cost_overview', 'primary_differences'] for ext in ['png', 'svg']]
    save_json(marker, {'plan_hash': frozen['plan_hash'], 'checksums': {name: file_hash(root / name) for name in required}})
    return table

def export_results(root):
    root = Path(root)
    target = root.parent / (root.name + '_results.zip')
    temporary = target.with_suffix('.tmp')
    allowed = {'.json', '.csv', '.npz', '.md', '.png', '.svg', '.py'}
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(root.rglob('*')):
            if (path.is_file() and path.suffix in allowed and not path.name.startswith('candidate_')
                    and not any(p.startswith('.') for p in path.relative_to(root).parts)):
                z.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise ValueError('Ошибка проверки экспортированного архива.')
    temporary.replace(target)
    return target
