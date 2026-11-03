"""Run the frozen experiment; scientific settings remain in WatchConfig.

The CPU smoke option is a technical check, never a substitute for the paper run.
"""
import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import kion_watch_feedback as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/kion_watch_feedback_50k_S20_v1'))
    parser.add_argument('--data-archive', default='', help='Optional official data_en.zip; otherwise downloaded and SHA checked.')
    parser.add_argument('--resume-from', default='', help='Saved run directory with checkpoints, not the light results ZIP.')
    parser.add_argument('--max-runs', type=int, default=0, help='0 = entire queue; positive = new models this invocation.')
    parser.add_argument('--pilot-steps', type=int, default=200)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='CPU: 80 users/cut, 2 updates/model; separate output required.')
    args = parser.parse_args()
    if args.max_runs < 0 or args.pilot_steps < 1:
        parser.error('max-runs must be nonnegative and pilot-steps must be positive')
    cfg = study.WatchConfig()
    if args.smoke:
        if args.output == parser.get_default('output'):
            parser.error('--smoke requires an explicit --output, e.g. outputs/cpu-smoke')
        cfg = replace(cfg, n_users=80, hidden_size=8, layers=1, batch_size=8,
                      eval_batch_size=32, max_steps=2, eval_every=1,
                      checkpoint_every=1, bootstrap_reps=24, technical_mode=True)
    study.torch.set_num_threads(2)
    cfg.validate()
    device = study.select_device(cfg)
    code = Path(study.__file__).read_text(encoding='utf-8')
    digest = hashlib.sha256(code.encode()).hexdigest()
    if digest != '5ab07ac369ae3966f3d781ada94a38415eed710ade65541c594c2a4fd3b57509':
        raise ValueError('Frozen implementation changed. Use the original release for this protocol.')
    root = args.output.resolve()
    archive = study.acquire_archive(root.parent / 'kion_data_cache', args.data_archive)
    films = study.read_films(archive)
    data = {cut[0]: study.prepare_split(films, cut, cfg) for cut in study.CUTS}
    frozen = study.make_plan(data, cfg, device, digest)
    study.restore_folder(root, frozen, args.resume_from)
    study.freeze_plan(root, frozen, data)
    source = root / 'implementation.py'
    if source.exists() and source.read_text(encoding='utf-8') != code:
        raise ValueError('Saved implementation differs.')
    source.write_text(code, encoding='utf-8', newline='\n')
    print('Plan:', frozen['plan_hash'], 'mode:', frozen['plan']['mode'], flush=True)
    print('Output:', root, flush=True)
    if args.prepare_only:
        return
    study.benchmark(data, cfg, root, frozen, device, min(args.pilot_steps, 2) if args.smoke else args.pilot_steps)
    complete = False
    try:
        complete = study.run_study(data, cfg, root, frozen, device, args.max_runs)
        if complete:
            study.report_study(data, cfg, root, frozen, device)
    finally:
        print('Results ZIP:', study.export_results(root), flush=True)
    print('All planned models complete:', complete, flush=True)


if __name__ == '__main__':
    main()
