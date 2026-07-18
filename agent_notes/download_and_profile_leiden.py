"""Download a realistic ENCODE contribution-score dataset and compare
TF-MoDISco's default (serial) Leiden clustering against the parallelized
Leiden-seed backend (``n_leiden_jobs`` in ``cluster.LeidenCluster`` /
``tfmodisco.TFMoDISco``), reporting both wall time and peak memory.

DO NOT RUN THIS ON A RESOURCE-CONSTRAINED MACHINE AT THE DEFAULTS BELOW.

The dataset is real production scale (30,534 peaks x 1000bp x 4) and the
production default settings (``max_seqlets_per_metacluster=20000``,
``n_leiden_runs=50``) are exactly what makes Leiden-seed parallelization
worth measuring in the first place -- but that also means this script is
meant for a beefier machine (many cores, tens of GB of RAM), not a laptop.
Each ``--n-leiden-jobs`` value spawns that many OS processes that each build
their OWN copy of the clustering graph (``leidenalg`` does not release the
GIL, so thread-based sharing isn't an option -- verified empirically before
writing this script), so memory scales close to linearly with the job count,
not just CPU. Start with small ``--n-peaks``/``--max-seqlets-per-metacluster``
to sanity check the script runs correctly, then scale up deliberately while
watching the peak-RSS column this script reports for each configuration.

Usage (small smoke check -- confirm the script works, ~1-2 min):

    python agent_notes/download_and_profile_leiden.py \\
        --n-peaks 500 --max-seqlets-per-metacluster 500 --n-leiden-runs 10 \\
        --n-leiden-jobs 1,2

Usage (representative comparison -- run on a machine with room to spare):

    python agent_notes/download_and_profile_leiden.py \\
        --n-peaks 20000 --window 400 --max-seqlets-per-metacluster 20000 \\
        --n-leiden-runs 50 --n-leiden-jobs 1,4,8,16 \\
        --report-path agent_notes/leiden_parallel_report.txt

Dataset: ENCFF407PRC (ENCODE PRO-cap peaks + one-hot sequences + hypothetical
contribution scores, 30,534 peaks x 1000bp x 4), ~445MB compressed / ~610MB
for the two arrays used here. Downloaded once and cached under --data-dir.
"""

import argparse
import json
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

TAR_URL = "https://www.encodeproject.org/files/ENCFF407PRC/@@download/ENCFF407PRC.tar.gz"
SEQS_MEMBER = "profile/ENCSR261KBX.input.seqs.npy"
CONTRIBS_MEMBER = "profile/ENCSR261KBX.input.hyp.contrib.scores.npy"


def download_and_extract(data_dir: Path) -> tuple:
    data_dir.mkdir(parents=True, exist_ok=True)
    seqs_path = data_dir / "seqs.npy"
    contribs_path = data_dir / "hyp_contribs.npy"

    if seqs_path.exists() and contribs_path.exists():
        return seqs_path, contribs_path

    tar_path = data_dir / "ENCFF407PRC.tar.gz"
    if not tar_path.exists():
        print(f"Downloading {TAR_URL} (~445MB) to {tar_path} ...")
        urllib.request.urlretrieve(TAR_URL, tar_path)

    print(f"Extracting {SEQS_MEMBER} and {CONTRIBS_MEMBER} ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        for member_name, out_path in [(SEQS_MEMBER, seqs_path),
                (CONTRIBS_MEMBER, contribs_path)]:
            src = tf.extractfile(member_name)
            with open(out_path, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)

    # The tar.gz is large and only needed once; drop it once both arrays are
    # extracted so re-runs don't need the disk space.
    tar_path.unlink(missing_ok=True)

    return seqs_path, contribs_path


def load_arrays(seqs_path: Path, contribs_path: Path, n_peaks: int,
        window: int, seed: int) -> tuple:
    # mmap so subsampling/windowing doesn't require materializing the full
    # ~610MB of both arrays in RAM before we've even cropped/subsampled.
    seqs = np.load(seqs_path, mmap_mode="r")
    contribs = np.load(contribs_path, mmap_mode="r")

    # Arrays are already (n_examples, length, 4) -- no transpose needed,
    # unlike the CLI's npy convention of (n_examples, 4, length). Verified
    # directly against the npy header of this specific dataset.
    n_total, length, _ = seqs.shape

    rng = np.random.RandomState(seed)
    if n_peaks < n_total:
        idx = np.sort(rng.choice(n_total, size=n_peaks, replace=False))
    else:
        idx = np.arange(n_total)

    center = length // 2
    from modiscolite.util import calculate_window_offsets
    start, end = calculate_window_offsets(center, window)

    one_hot = np.asarray(seqs[idx][:, start:end, :], dtype="float32")
    hyp_contribs = np.asarray(contribs[idx][:, start:end, :], dtype="float32")
    return one_hot, hyp_contribs


def _measure_subprocess(cmd) -> dict:
    # resource.getrusage(RUSAGE_CHILDREN) does NOT work for this: it's a
    # running high-water mark that never resets between calls (so later
    # configs inherit earlier configs' peak), and it only accounts for
    # direct children -- it can't see the `loky` worker processes joblib
    # spawns as grandchildren when n_leiden_jobs > 1, which is exactly the
    # memory this script exists to measure. Poll the actual process tree
    # (subprocess + all descendants) instead.
    try:
        import psutil
    except ImportError:
        psutil = None
        print("  (psutil not installed -- peak_rss_mb will read 0.0; "
              "`pip install psutil` for real per-config memory measurement)")

    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)

    peak_rss_bytes = 0
    if psutil is not None:
        try:
            ps_proc = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            ps_proc = None

        while proc.poll() is None:
            if ps_proc is not None:
                try:
                    tree = [ps_proc] + ps_proc.children(recursive=True)
                    total = sum(p.memory_info().rss for p in tree if p.is_running())
                    peak_rss_bytes = max(peak_rss_bytes, total)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            time.sleep(0.05)

    stdout, stderr = proc.communicate()
    wall_seconds = time.perf_counter() - start

    if proc.returncode != 0:
        raise RuntimeError(f"worker failed (exit {proc.returncode}):\n{stderr}")

    peak_rss_mb = peak_rss_bytes / 1e6
    payload = json.loads(stdout.strip().splitlines()[-1])
    return {"wall_seconds": wall_seconds, "peak_rss_mb": peak_rss_mb,
        **payload}


def _run_worker(args):
    # Executed inside the measured subprocess: run TFMoDISco once with a
    # single n_leiden_jobs setting and print a one-line JSON result.
    from modiscolite import tfmodisco, util

    seqs_path = Path(args.data_dir) / "seqs.npy"
    contribs_path = Path(args.data_dir) / "hyp_contribs.npy"
    one_hot, hyp_contribs = load_arrays(seqs_path, contribs_path,
        args.n_peaks, args.window, args.seed)

    profiler = util.ProfileRecorder()
    pos_patterns, neg_patterns = tfmodisco.TFMoDISco(
        one_hot=one_hot,
        hypothetical_contribs=hyp_contribs,
        max_seqlets_per_metacluster=args.max_seqlets_per_metacluster,
        n_leiden_runs=args.n_leiden_runs,
        n_leiden_jobs=args.n_leiden_jobs,
        profile=profiler)

    summary = profiler.summary()
    result = {
        "n_pos_patterns": len(pos_patterns) if pos_patterns else 0,
        "n_neg_patterns": len(neg_patterns) if neg_patterns else 0,
        "profile_summary": summary,
    }
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="agent_notes/data/encode_ENCFF407PRC")
    parser.add_argument("--n-peaks", type=int, default=3000,
        help="Subsample to this many peaks (of 30,534 available). Keep small "
             "for a first smoke check.")
    parser.add_argument("--window", type=int, default=400)
    parser.add_argument("--max-seqlets-per-metacluster", type=int, default=2000)
    parser.add_argument("--n-leiden-runs", type=int, default=50)
    parser.add_argument("--n-leiden-jobs", default="1,4",
        help="Comma-separated list of n_leiden_jobs values to compare, e.g. "
             "'1,4,8'. The first value should normally be 1 -- that is the "
             "unmodified modisco default/baseline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", default=None,
        help="Optional path to also write a plain-text report to.")
    # Internal: re-exec entry point for the measured subprocess.
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        _run_worker(args)
        return

    data_dir = Path(args.data_dir)
    download_and_extract(data_dir)

    n_jobs_values = [int(v) for v in args.n_leiden_jobs.split(",")]

    lines = []
    lines.append("git commit: <fill in via `git rev-parse HEAD`>")
    lines.append(f"dataset: ENCFF407PRC, n_peaks={args.n_peaks}, window={args.window}")
    lines.append(f"max_seqlets_per_metacluster={args.max_seqlets_per_metacluster}, "
        f"n_leiden_runs={args.n_leiden_runs}")
    lines.append("")

    for n_jobs in n_jobs_values:
        label = "modisco defaults (n_leiden_jobs=1)" if n_jobs == 1 else \
            f"parallel Leiden (n_leiden_jobs={n_jobs})"
        print(f"Running: {label} ...")

        cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker",
            "--data-dir", str(data_dir),
            "--n-peaks", str(args.n_peaks),
            "--window", str(args.window),
            "--max-seqlets-per-metacluster", str(args.max_seqlets_per_metacluster),
            "--n-leiden-runs", str(args.n_leiden_runs),
            "--n-leiden-jobs", str(n_jobs),
            "--seed", str(args.seed)]

        measured = _measure_subprocess(cmd)

        lines.append(f"=== {label} ===")
        lines.append(f"wall_seconds: {measured['wall_seconds']:.2f}")
        lines.append(f"peak_rss_mb: {measured['peak_rss_mb']:.1f}")
        lines.append(f"n_pos_patterns: {measured['n_pos_patterns']}")
        lines.append(f"n_neg_patterns: {measured['n_neg_patterns']}")
        for stage, stats in measured["profile_summary"].items():
            lines.append(f"  {stage}: {stats['seconds']:.4f}s over {stats['count']} call(s)")
        lines.append("")

        print(f"  wall={measured['wall_seconds']:.2f}s peak_rss={measured['peak_rss_mb']:.1f}MB")

    report = "\n".join(lines)
    print()
    print(report)

    if args.report_path:
        Path(args.report_path).write_text(report)
        print(f"\nReport written to {args.report_path}")


if __name__ == "__main__":
    main()
