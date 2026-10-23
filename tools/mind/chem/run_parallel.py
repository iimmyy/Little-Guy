"""Run the section F suite split across fresh processes (no cache shared between them), then write one results.json and
report.md in the usual TESTS order. Same answers as test_foundations.main: every test seeds its own randomness.
usage: run_parallel.py <out_dir> [workers]"""
import multiprocessing as mp
import os
import sys


def _worker(args):
    indices, = args
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import engine as E
    import test_foundations as T
    E.calibrate()
    return T.run_tests(set(indices))


def main(out_dir, workers):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import engine as E
    import test_foundations as T
    os.makedirs(out_dir, exist_ok=True)
    n = len(T.TESTS)
    slices = [list(range(i, n, workers)) for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_worker, [(s,) for s in slices if s])
    by_name = {r["name"]: r for part in parts for r in part}
    E.calibrate()                                                     # the rest knobs written with the results
    T.write_results(out_dir, [by_name[t["name"]] for t in T.TESTS])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
