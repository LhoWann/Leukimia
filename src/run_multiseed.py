import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from main import EXPERIMENTS, run_experiment

KEY_EXPERIMENTS = ['focusmix_stain', 'no_mix', 'focusmix', 'coatnet_0', 'coatnet_0_stain']

SEEDS = [42, 123, 2025, 7, 314, 1001, 555, 8888]  # dari 3 -> 8 seed

CKPT_ROOT = 'checkpoints_multiseed'
LOG_ROOT = 'logs_multiseed'
RESULTS_ROOT = 'results_multiseed'
EVAL_SCRIPT = SRC / 'evaluate_external.py'


def find_best_ckpt(ckpt_dir: Path):
    cands = [p for p in ckpt_dir.glob('*.ckpt') if p.name != 'last.ckpt']
    if not cands:
        return None

    def _val_f1(p: Path) -> float:
        m = re.search(r'val_f1=([0-9]+(?:\.[0-9]+)?)', p.stem)
        return float(m.group(1)) if m else 0.0

    return max(cands, key=_val_f1)


def train_one(exp: str, seed: int, data_dir: str, skip_existing: bool,
              ckpt_root: str = CKPT_ROOT, log_root: str = LOG_ROOT):
    run_name = f'{exp}_seed{seed}'
    ckpt_dir = Path(ckpt_root) / run_name
    if skip_existing and find_best_ckpt(ckpt_dir) is not None:
        print(f'[skip-train] {run_name} (checkpoint sudah ada)')
        return find_best_ckpt(ckpt_dir)

    best = run_experiment(
        EXPERIMENTS[exp], data_dir=data_dir, seed=seed,
        ckpt_root=ckpt_root, log_root=log_root, run_name=run_name,
    )
    return Path(best) if best else find_best_ckpt(ckpt_dir)


def eval_one(exp: str, seed: int, ckpt_path: Path, cnmc_dir: str,
             data_dir: str, tta_n: int, skip_existing: bool,
             results_root: str = RESULTS_ROOT):
    run_name = f'{exp}_seed{seed}'
    out_json = Path(results_root) / f'{run_name}.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and out_json.exists():
        print(f'[skip-eval] {run_name} (JSON sudah ada)')
    else:
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            '--ckpt', str(ckpt_path),
            '--cnmc-dir', cnmc_dir,
            '--data-dir', data_dir,
            '--tta-n', str(tta_n),
            '--output-json', str(out_json),
        ]
        print('  $', ' '.join(cmd))
        subprocess.run(cmd, check=True)

    with open(out_json) as f:
        data = json.load(f)
    data['seed'] = seed
    data['exp'] = exp
    with open(out_json, 'w') as f:
        json.dump(data, f, indent=2)
    return out_json


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exps', nargs='+', default=KEY_EXPERIMENTS,
                    choices=list(EXPERIMENTS), metavar='EXP')
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--data-dir', default='dataset')
    ap.add_argument('--cnmc-dir', default='PKG_C_NMC 2019/C-NMC_train_merged')
    ap.add_argument('--tta-n', type=int, default=1,
                    help='1 = no-TTA (headline). Pakai 8 untuk ablation TTA.')
    ap.add_argument('--results-root', default=RESULTS_ROOT,
                    help='folder output JSON. Pakai folder lain (mis. '
                         'results_multiseed_tta8 / results_coatnet) agar tidak menimpa run lain.')
    ap.add_argument('--ckpt-root', default=CKPT_ROOT,
                    help='folder checkpoint per-run. Pisahkan (mis. checkpoints_coatnet) '
                         'agar artefak baseline tidak tercampur dengan ConvNeXtV2.')
    ap.add_argument('--log-root', default=LOG_ROOT,
                    help='folder CSV log per-run. Pisahkan (mis. logs_coatnet) bila perlu.')
    ap.add_argument('--no-train', action='store_true', help='lewati training')
    ap.add_argument('--no-eval', action='store_true', help='lewati evaluasi')
    ap.add_argument('--skip-existing', action='store_true',
                    help='lewati run yang checkpoint/JSON-nya sudah ada')
    args = ap.parse_args()

    print(f'Root proyek : {ROOT}')
    print(f'Eksperimen  : {args.exps}')
    print(f'Seeds       : {args.seeds}')
    print(f'TTA         : {args.tta_n} ({"no-TTA" if args.tta_n == 1 else "ablation"})')

    manifest = {}
    failures = []

    for exp in args.exps:
        for seed in args.seeds:
            run_name = f'{exp}_seed{seed}'
            print(f'\n{"#" * 70}\n#  {run_name}\n{"#" * 70}')

            ckpt = None
            if not args.no_train:
                ckpt = train_one(exp, seed, args.data_dir, args.skip_existing,
                                 ckpt_root=args.ckpt_root, log_root=args.log_root)
            if ckpt is None:
                ckpt = find_best_ckpt(Path(args.ckpt_root) / run_name)
            if ckpt is None:
                print(f'[ERROR] tidak ada checkpoint untuk {run_name} — lewati evaluasi')
                failures.append(run_name)
                continue

            if not args.no_eval:
                try:
                    eval_one(exp, seed, ckpt, args.cnmc_dir,
                             args.data_dir, args.tta_n, args.skip_existing,
                             results_root=args.results_root)
                except subprocess.CalledProcessError as e:
                    print(f'[ERROR] evaluasi {run_name} gagal: {e}')
                    failures.append(run_name)

            manifest.setdefault(exp, {})[str(seed)] = str(ckpt)

    Path(args.results_root).mkdir(parents=True, exist_ok=True)
    with open(Path(args.results_root) / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\n{"=" * 70}')
    print(f'Manifest  -> {Path(args.results_root) / "manifest.json"}')
    if failures:
        print(f'Gagal     : {failures}')
    print('Langkah berikutnya: python src/aggregate_seeds.py')


if __name__ == '__main__':
    main()
