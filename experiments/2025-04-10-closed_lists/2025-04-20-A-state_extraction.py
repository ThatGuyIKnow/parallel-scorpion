#! /usr/bin/env python

import os
from pathlib import Path

from downward.experiment import FastDownwardExperiment, FastDownwardAlgorithm, FastDownwardRun
from downward.suites import build_suite
from downward.cached_revision import CachedFastDownwardRevision
from lab.experiment import Experiment
from lab.environments import TetralithEnvironment, LocalEnvironment

import project

REVISION_CACHE = (
        os.environ.get("DOWNWARD_REVISION_CACHE") or project.DIR / "data" / "revision-cache"
)
BUILD_OPTIONS = []
if project.REMOTE:
    ENV = TetralithEnvironment(
        setup=TetralithEnvironment.DEFAULT_SETUP,
        email="olijo92@liu.se",
        extra_options="#SBATCH -A naiss2024-5-421",
        memory_per_cpu="9G",
    )
    TIME_LIMIT = 30 * 60
    MEMORY_LIMIT = "8G"
    SUITE = project.SUITE_OPTIMAL_STRIPS
else:
    ENV = LocalEnvironment(processes=3)
    MEMORY_LIMIT = "4G"
    TIME_LIMIT = 5 * 60
    SUITE = build_suite(
         os.environ.get("DOWNWARD_BENCHMARKS"),
        ["depot:p01.pddl", "grid:prob01.pddl", "gripper:prob01.pddl"]
    )


DRIVER_OPTIONS = [
    "--overall-time-limit",
    f"{TIME_LIMIT}s",
    "--overall-memory-limit",
    MEMORY_LIMIT,
    ]
CONFIGS = [
    (f"state_analytics", ["--search", f"astar(blind())"]),
]
REV_NICKS = [("state_analytics", "")]

exp = Experiment(environment=ENV)
for rev, rev_nick in REV_NICKS:
    cached_rev = CachedFastDownwardRevision(REVISION_CACHE, project.get_repo_base(), rev, BUILD_OPTIONS)
    cached_rev.cache()
    exp.add_resource("", cached_rev.path, cached_rev.get_relative_exp_path())
    for config_nick, config in CONFIGS:
        algo_name = f"{rev_nick}-{config_nick}" if rev_nick else config_nick

        bounds = {}
        for task in SUITE:
            algo = FastDownwardAlgorithm(
                algo_name,
                cached_rev,
                DRIVER_OPTIONS,
                config,
            )
            run = FastDownwardRun(exp, algo, task)
            exp.add_run(run)

exp.add_parser(FastDownwardExperiment.EXITCODE_PARSER)
exp.add_parser(FastDownwardExperiment.TRANSLATOR_PARSER)
exp.add_parser(FastDownwardExperiment.SINGLE_SEARCH_PARSER)
exp.add_parser(FastDownwardExperiment.PLANNER_PARSER)

exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)

# Parse the commandline and run the given steps.
exp.run_steps()
