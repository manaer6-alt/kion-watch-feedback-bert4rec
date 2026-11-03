# Заменяет настройки и все последующие ячейки старого ноутбука.
# Сначала выполните только исходную ячейку с PROGRAM_SOURCE.
from dataclasses import dataclass, asdict
import shutil

assert hashlib.sha256(PROGRAM_SOURCE.encode()).hexdigest() == PROGRAM_SHA256
assert PROGRAM_SHA256 == "5ab07ac369ae3966f3d781ada94a38415eed710ade65541c594c2a4fd3b57509"

TOTAL_STEPS = 40_000
FIRST_BUDGET = 20_000
METHODS = ["ALL", "SOFT"]
SEEDS = [17, 29, 43]
RESUME_FROM = ""  # Папка серии из сохранённых выходов Kaggle Input; иначе пусто.
DATA_ARCHIVE = ""  # Иначе исходный KION скачивается автоматически.
MAX_NEW_MODELS = 0  # 0 = все 12 моделей; 1 = закончить одну и сохранить выходы.
assert MAX_NEW_MODELS >= 0


@dataclass(frozen=True)
class ReviewConfig(WatchConfig):
    max_steps: int = TOTAL_STEPS

    def validate(self):
        expected = asdict(WatchConfig())
        expected["max_steps"] = TOTAL_STEPS
        if asdict(self) != expected:
            raise ValueError("Параметры отличаются от согласованной проверки бюджета.")
        return self


CFG = ReviewConfig().validate()
DEVICE = select_device(CFG)
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
RESULTS_DIR = WORK_ROOT / "kion_article2_budget40k_v1"
# Сохраняем также код этой адаптации. Это обеспечивает проверку при продолжении.
_cell_source = globals().get("_REVIEW_CELL_SOURCE")
if _cell_source is None:
    _cell_source = get_ipython().history_manager.input_hist_raw[-1]
if not _cell_source.strip():
    raise ValueError("Не удалось зафиксировать код текущей ячейки.")
_cell_hash = hashlib.sha256(_cell_source.encode()).hexdigest()
ARCHIVE = acquire_archive(WORK_ROOT / "kion_data_cache", DATA_ARCHIVE)
FILMS = read_films(ARCHIVE)
DATA = {cut[0]: prepare_split(FILMS, cut, CFG) for cut in CUTS}
FROZEN = make_plan(DATA, CFG, DEVICE, PROGRAM_SHA256)
FROZEN["plan"].update(
    version="article2-budget-sensitivity-v1",
    adapter_sha256=_cell_hash,
    evaluation_budgets=[FIRST_BUDGET, TOTAL_STEPS],
    primary_contrasts=[],
    planned_contrasts=[[split, "SOFT", "ALL", budget] for split in DATA
                       for budget in [FIRST_BUDGET, TOTAL_STEPS]],
    interval_level=0.95,
    interpretation="Exploratory review-driven sensitivity on previously inspected test periods",
    checkpoint_selection="first validation maximum separately within each budget",
)
FROZEN["plan_hash"] = canonical(FROZEN["plan"])
restore_folder(RESULTS_DIR, FROZEN, RESUME_FROM)
freeze_plan(RESULTS_DIR, FROZEN, DATA)
for _name, _text in [("implementation.py", PROGRAM_SOURCE), ("review_cell.py", _cell_source)]:
    _path = RESULTS_DIR / _name
    if _path.exists() and _path.read_text(encoding="utf-8") != _text:
        raise ValueError("Изменён сохранённый код: " + _name)
    _path.write_text(_text, encoding="utf-8", newline="\n")
print("План:", FROZEN["plan_hash"], "| Моделей:", len(queue(DATA)))
print("Папка:", RESULTS_DIR)


def preserve_first_budget(split, method, seed):
    name = run_name(split.name, method, seed)
    source = RESULTS_DIR / "runs" / name
    target = RESULTS_DIR / "budget20k" / name
    identity = expected_identity(split, method, seed, FROZEN)
    marker = target / "selection.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("Несовместимое состояние 20k: " + name)
        for file, digest in saved["checksums"].items():
            if file_hash(target / file) != digest:
                raise ValueError("Повреждён файл 20k: " + name + "/" + file)
        return
    state = load_torch(source / "last.pt")
    if state["identity"] != identity or state["step"] != FIRST_BUDGET:
        raise ValueError("Нельзя восстановить выбор на границе 20k: " + name)
    best = source / f"candidate_step_{state['best_step']}.pt"
    if file_hash(best) != state["best_sha256"]:
        raise ValueError("Повреждён выбранный checkpoint: " + name)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, target / "best.pt")
    save_csv(target / "history.csv", state["history"])
    save_json(marker, {"identity": identity, "budget": FIRST_BUDGET,
        "selected_step": state["best_step"], "validation": state["best_metrics"],
        "checksums": {f: file_hash(target / f) for f in ["best.pt", "history.csv"]}})


def run_budget_check():
    new_models = 0
    for split_name, method, seed in queue(DATA):
        split = DATA[split_name]
        name = run_name(split_name, method, seed)
        folder = RESULTS_DIR / "runs" / name
        identity = expected_identity(split, method, seed, FROZEN)
        if verify_completed(folder, identity):
            preserve_first_budget(split, method, seed)
            continue
        if (RESULTS_DIR / "test_selection_lock.json").exists():
            raise ValueError("После фиксации теста отсутствует обученная модель.")
        if MAX_NEW_MODELS and new_models >= MAX_NEW_MODELS:
            return False
        if not (RESULTS_DIR / "budget20k" / name / "selection.json").exists():
            step = load_torch(folder / "last.pt")["step"] if (folder / "last.pt").exists() else 0
            if step < FIRST_BUDGET:
                train_run(split, method, seed, CFG, RESULTS_DIR, FROZEN, DEVICE,
                          stop_after_updates=FIRST_BUDGET - step)
                if load_torch(folder / "last.pt")["step"] < FIRST_BUDGET:
                    return False
            preserve_first_budget(split, method, seed)
        if not train_run(split, method, seed, CFG, RESULTS_DIR, FROZEN, DEVICE):
            return False
        new_models += 1
        save_json(RESULTS_DIR / "progress.json", {
            "plan_hash": FROZEN["plan_hash"], "status": "training", "last_completed": name,
            "completed_models": len(list((RESULTS_DIR / "runs").glob("*/completed.json"))),
            "total_models": len(queue(DATA))})
    # Тест открывается после завершения всех 12 моделей и фиксации обоих наборов весов.
    examples, lock40 = freeze_test(DATA, CFG, RESULTS_DIR, FROZEN)
    lock = {"plan_hash": FROZEN["plan_hash"], "budget40k": lock40,
            "budget20k_sha256": {run_name(a, b, c): file_hash(
                RESULTS_DIR / "budget20k" / run_name(a, b, c) / "best.pt")
                for a, b, c in queue(DATA)}}
    lock_path = RESULTS_DIR / "budget_selection_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise ValueError("Изменились выбранные модели.")
    save_json(lock_path, lock)
    results = []
    for split_name, method, seed in queue(DATA):
        name = run_name(split_name, method, seed)
        for budget, subdir in [(FIRST_BUDGET, "budget20k"), (TOTAL_STEPS, "runs")]:
            weight = RESULTS_DIR / subdir / name / "best.pt"
            folder = RESULTS_DIR / "evaluation" / str(budget) / name
            identity = {"plan_hash": FROZEN["plan_hash"], "split": split_name,
                "method": method, "seed": seed, "budget": budget,
                "weight_sha256": file_hash(weight),
                "test_fingerprint": lock40["test_manifests"][split_name]["fingerprint"]}
            marker = folder / "completed.json"
            if marker.exists():
                cached = json.loads(marker.read_text(encoding="utf-8"))
                if cached["identity"] != identity:
                    raise ValueError("Несовместимая оценка: " + name)
                for file, digest in cached["checksums"].items():
                    if file_hash(folder / file) != digest:
                        raise ValueError("Повреждён результат: " + name)
                record = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            else:
                model = CompactBERT4Rec(len(DATA[split_name].item_ids), CFG).to(DEVICE)
                state = load_torch(weight)
                if state["identity"] != expected_identity(DATA[split_name], method, seed, FROZEN):
                    raise ValueError("Вес относится к другой модели.")
                model.load_state_dict(state["model"])
                values, users = evaluate(model, examples[split_name], method, CFG, DEVICE)
                record = {**identity, "selected_step": state["step"], **values}
                save_json(folder / "summary.json", record)
                save_csv(folder / "per_user.csv", users)
                save_json(marker, {"identity": identity, "checksums": {
                    f: file_hash(folder / f) for f in ["summary.json", "per_user.csv"]}})
                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
            results.append(record)
    table = pd.DataFrame(results)
    save_csv(RESULTS_DIR / "results_per_run.csv", table)
    metrics = ["ndcg10_high", "recall10_high", "ndcg10_any", "recall10_any"]
    aggregate = table.groupby(["budget", "split", "method"])[metrics].agg(["mean", "std"])
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    save_csv(RESULTS_DIR / "results_by_method.csv", aggregate.reset_index())
    paired = {}
    for budget in [FIRST_BUDGET, TOTAL_STEPS]:
        for split_name in DATA:
            for method in METHODS:
                frames = [pd.read_csv(RESULTS_DIR / "evaluation" / str(budget) /
                    run_name(split_name, method, seed) / "per_user.csv").set_index("user_id") for seed in SEEDS]
                assert all(f.index.equals(frames[0].index) for f in frames)
                paired[budget, split_name, method] = sum(f[metrics] for f in frames) / len(SEEDS)
    intervals = []
    for split_name in DATA:
        contrasts = [(f"SOFT-ALL at {b}", paired[b, split_name, "SOFT"],
                      paired[b, split_name, "ALL"]) for b in [FIRST_BUDGET, TOTAL_STEPS]]
        contrasts += [(f"40k-20k {m}", paired[TOTAL_STEPS, split_name, m],
                       paired[FIRST_BUDGET, split_name, m]) for m in METHODS]
        for contrast, left, right in contrasts:
            assert left.index.equals(right.index)
            for metric in metrics:
                delta = (left[metric] - right[metric]).dropna().to_numpy()
                intervals.append({"split": split_name, "contrast": contrast, "metric": metric,
                    "scope": "exploratory_conditional_95_no_multiplicity_correction",
                    **bootstrap_interval(delta, 0.95, CFG)})
    save_csv(RESULTS_DIR / "paired_intervals.csv", intervals)
    save_json(RESULTS_DIR / "FINAL_SUMMARY.json", {
        "status": "complete", "trained_models": len(queue(DATA)), "evaluated_models": len(table),
        "plan_hash": FROZEN["plan_hash"], "interpretation": "exploratory_budget_sensitivity"})
    print("ГОТОВО: 12 моделей, 24 оценки двух бюджетов.")
    display(aggregate)
    return True


try:
    COMPLETE = run_budget_check()
    if not COMPLETE:
        print("Серия не завершена. Сохраните полные выходы и продолжите тем же кодом.")
finally:
    LIGHT_ZIP = export_results(RESULTS_DIR)
    # Полный архив включает best.pt и last.pt. Он находится ВНЕ папки серии.
    FULL_ZIP = Path(shutil.make_archive(str(RESULTS_DIR) + "_FULL", "zip", root_dir=RESULTS_DIR))
    print("Отчёт:", LIGHT_ZIP)
    print("ВЕСА И ПРОГРЕСС — обязательно сохраните:", FULL_ZIP)
    display(FileLink(os.path.relpath(LIGHT_ZIP, Path.cwd())))
    display(FileLink(os.path.relpath(FULL_ZIP, Path.cwd())))
