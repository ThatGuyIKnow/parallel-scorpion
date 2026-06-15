#! /usr/bin/env python
"""HTG GBFS-FF parallel experiment for parallel-scorpion (peager), built to be
directly comparable to distributed-tyr's full-suite-experiment.

Search config matches tyr: GBFS with the FF heuristic, preferred operators on,
boosted alternating dual queue (boost = 1000). peager is MPI-only, so a "worker"
is an MPI rank (driver flag -np N); tyr's WxT reduces here to 1x1 and 16x1.

REMOTE (Tetralith): the real comparison -- 32 GB / 10 min, 1 vs 16 workers, full
HTG suite, headline config grazhda (GRAZHDA*/sparsity) plus a zobrist baseline.
Local: parse-validation only -- a tiny HTG subset, reduced limits, small np
(this box has 12 cores / 14 GB).

Build the planner first:  ./build.py release
Run, e.g.:
  DOWNWARD_BENCHMARKS=/home/workbox/Projects/downward-projects/benchmarks \
      uv run experiments/2024-12-parallel/2026-06-15-htg-gbfs-ff.py build start parse fetch
"""
import os
import sys

import htg_parser
import project

from downward.suites import build_suite
from lab.experiment import Experiment

REPO = project.get_repo_base()
FAST_DOWNWARD = str(REPO / "fast-downward.py")
BENCHMARKS_DIR = os.environ.get(
    "DOWNWARD_BENCHMARKS", "/home/workbox/Projects/downward-projects/benchmarks")

# GBFS + FF, preferred operators with boost=1000 (matches tyr). FF is defined as
# the evaluator `h`; the alternation open list pairs a standard and a
# preferred-only queue, boosted -- the eager analogue of lazy_greedy([ff()],
# preferred=[ff()], boost=1000).
EVALUATORS = ["--evaluator", "h=ff()"]


def search_string(hash_fn):
    return (
        "peager(alt([single(h), single(h, pref_only=true)], boost=1000), "
        "preferred=[h], reopen_closed=false, f_eval=h, hash=" + hash_fn + ")"
    )


# All HTG domains (subdirectories of htg-domains), used for the full remote run.
HTG_DOMAINS = sorted(
    d for d in os.listdir(os.path.join(BENCHMARKS_DIR, "htg-domains"))
    if os.path.isdir(os.path.join(BENCHMARKS_DIR, "htg-domains", d))
) if os.path.isdir(os.path.join(BENCHMARKS_DIR, "htg-domains")) else []

if project.REMOTE:
    # 16 cores on one node, 2 GB/cpu => 32 GB total; SLURM enforces the memory.
    ENV = project.TetralithEnvironment(
        email="olijo92@liu.se",
        extra_options="#SBATCH -A naiss2025-22-1329\n#SBATCH --nodes=1\n"
                      "#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=16",
        memory_per_cpu="2G",
    )
    SUITE = build_suite(os.path.join(BENCHMARKS_DIR, "htg-domains"), HTG_DOMAINS)
    SEARCH_TIME_LIMIT = "10m"
    MEMORY_LIMIT_MB = 32000
    WALL_TIME_LIMIT = 12 * 60  # 10 min search + headroom for MPI setup/teardown
    CONFIGS = [
        # nick, driver -np, hash. At np=1 the hash is irrelevant (single rank);
        # this is the sequential baseline for search overhead.
        ("gbfs-ff-1", ["-np", "1"], search_string("grazhda()")),
        ("gbfs-ff-16-grazhda", ["-np", "16"], search_string("grazhda()")),
        ("gbfs-ff-16-zobrist", ["-np", "16"], search_string("zobrist()")),
    ]
else:
    ENV = project.LocalEnvironment(processes=1)
    # Tiny, fast HTG-test instances; just enough to validate parsing end to end.
    SUITE = build_suite(
        os.path.join(BENCHMARKS_DIR, "htg-test"),
        [
            "visitall-multidimensional-3-dim-visitall-CLOSE-g1:p0.pddl",
            "childsnack-contents-parsize1-cham3:contentam1-p0.pddl",
            "genome-edit-distance:d-2-1.pddl",
        ],
    )
    SEARCH_TIME_LIMIT = "60s"
    MEMORY_LIMIT_MB = 8000
    WALL_TIME_LIMIT = 90
    CONFIGS = [
        ("gbfs-ff-1", ["-np", "1"], search_string("grazhda()")),
        ("gbfs-ff-2-grazhda", ["-np", "2"], search_string("grazhda()")),
        ("gbfs-ff-4-zobrist", ["-np", "4"], search_string("zobrist()")),
    ]

ATTRIBUTES = [
    "error",
    "coverage",
    "oom",
    "oot",
    "total_expansions",
    "off_parent_transfer_rate",
    "num_off_parent_transfers",
    "num_generated_co",
    "total_peak_memory",
    "memory_mb",
    "score_peak_memory",
    "search_time",
]

exp = Experiment(environment=ENV)
exp.add_parser(htg_parser.get_parser())

for nick, driver_np, search_str in CONFIGS:
    driver_options = [
        "--overall-time-limit", SEARCH_TIME_LIMIT,
        "--overall-memory-limit", f"{MEMORY_LIMIT_MB}M",
    ] + driver_np
    for task in SUITE:
        run = exp.add_run()
        run.add_command(
            "solve",
            [sys.executable, FAST_DOWNWARD, *driver_options,
             task.domain_file, task.problem_file, *EVALUATORS,
             "--search", search_str],
            time_limit=WALL_TIME_LIMIT,
            memory_limit=MEMORY_LIMIT_MB,
        )
        run.set_property("domain", task.domain)
        run.set_property("problem", task.problem)
        run.set_property("algorithm", nick)
        run.set_property("time_limit", SEARCH_TIME_LIMIT)
        run.set_property("memory_limit", MEMORY_LIMIT_MB)
        run.set_property("id", [nick, task.domain, task.problem])

exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("parse", exp.parse)
exp.add_fetcher(name="fetch")
project.add_absolute_report(exp, attributes=ATTRIBUTES)

exp.run_steps()
