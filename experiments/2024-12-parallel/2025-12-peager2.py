#! /usr/bin/env python

from collections import namedtuple
import os
import sys

import custom_parser
import project
from downward.reports.absolute import AbsoluteReport
from downward.suites import build_suite
from lab.environments import BaselSlurmEnvironment, LocalEnvironment
from lab.experiment import Experiment
from lab.parser import Parser
from lab.reports import Attribute

SEED = 2018
TIME_LIMIT = 1800
MEMORY_LIMIT = 2048

USER = project.oliver_dfsplan

REPO = project.get_repo_base()
BENCHMARKS_DIR = os.environ["DOWNWARD_BENCHMARKS"]
# If REVISION_CACHE is None, the default "./data/revision-cache/" is used.
REVISION_CACHE = os.environ.get("DOWNWARD_REVISION_CACHE")
if project.REMOTE:
    SUITE = project.SUITE_SATISFICING
    ENV = project.TetralithEnvironment(
        email="olijo92@liu.se",
        extra_options="#SBATCH -A naiss2024-5-421",
        memory_per_cpu="9G",
    )
else:
    SUITE = ["depot:p01.pddl", "grid:prob01.pddl", "gripper:prob01.pddl"]
    ENV = project.LocalEnvironment(processes=1)
SUITE = build_suite(BENCHMARKS_DIR, SUITE)

CONFIGS = [
    ("001-hda", ["--evaluator", "h=ff()" ,"--search", "peager(tiebreaking([sum([g(), h]), h], unsafe_pruning=false), reopen_closed=true, f_eval=sum([g(), h]), hash=zobrist())"])
]
BUILD_OPTIONS = []
DRIVER_OPTIONS = ["--overall-time-limit", "5m", "-np", "4"  ]
REV_NICKS = [
    ("hda", ""),
]
ATTRIBUTES = [
    "error",
    "run_dir",
    "search_start_time",
    "search_start_memory",
    "total_time",
    "h_values",
    "coverage",
    "expansions",
    "peak_memory",
    "total_peak_memory",
    project.EVALUATIONS_PER_TIME,
]

exp = project.FastDownwardExperiment(environment=ENV, revision_cache=REVISION_CACHE)
for config_nick, config in CONFIGS:
    for rev, rev_nick in REV_NICKS:
        algo_name = f"{rev_nick}:{config_nick}" if rev_nick else config_nick
        exp.add_algorithm(
            algo_name,
            REPO,
            rev,
            config,
            build_options=BUILD_OPTIONS,
            driver_options=DRIVER_OPTIONS,
        )

        exp._algorithms[algo_name].driver_options = [v for v in exp._algorithms[algo_name].driver_options if v != "--validate"]
exp.add_suite(BENCHMARKS_DIR, SUITE)
# exp.add_parser(exp.EXITCODE_PARSER)
# exp.add_parser(exp.TRANSLATOR_PARSER)
# exp.add_parser(exp.SINGLE_SEARCH_PARSER)
exp.add_parser(custom_parser.get_parser())
# exp.add_parser(exp.PLANNER_PARSER)

# exp.add_parser(project.FastDownwardExperiment.EXITCODE_PARSER)
# exp.add_parser(project.FastDownwardExperiment.TRANSLATOR_PARSER)
# exp.add_parser(project.FastDownwardExperiment.SINGLE_SEARCH_PARSER)
# exp.add_parser(custom_parser.get_parser())
# exp.add_parser(project.FastDownwardExperiment.PLANNER_PARSER)

exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("parse", exp.parse)
exp.add_fetcher(name="fetch")

# if not project.REMOTE:
#     project.add_scp_step(exp, SCP_LOGIN, REMOTE_REPOS_DIR)

project.add_absolute_report(
    exp,
    attributes=ATTRIBUTES,
    filter=[project.add_evaluations_per_time, project.group_domains],
)
# Parse the commandline and run the given steps.
exp.run_steps()
