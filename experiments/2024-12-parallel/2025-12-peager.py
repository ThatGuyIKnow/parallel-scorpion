#! /usr/bin/env python

import os

import custom_parser
import project
from downward.reports.absolute import AbsoluteReport
from lab.environments import BaselSlurmEnvironment, LocalEnvironment
from lab.experiment import Experiment
from lab.parser import Parser
from lab.reports import Attribute

SEED = 2018
TIME_LIMIT = 1800
MEMORY_LIMIT = 2048

REPO = project.get_repo_base()
BENCHMARKS_DIR = os.environ["DOWNWARD_BENCHMARKS"]
SCP_LOGIN = "myname@myserver.com"
REMOTE_REPOS_DIR = "/infai/seipp/projects"
# If REVISION_CACHE is None, the default "./data/revision-cache/" is used.
REVISION_CACHE = os.environ.get("DOWNWARD_REVISION_CACHE")
if project.REMOTE:
    SUITE = project.SUITE_SATISFICING
    ENV = project.BaselSlurmEnvironment(email="my.name@myhost.ch")
else:
    SUITE = ["depot:p01.pddl", "grid:prob01.pddl", "gripper:prob01.pddl"]
    ENV = project.LocalEnvironment(processes=1)

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
    "memory",
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

exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("parse", exp.parse)
exp.add_fetcher(name="fetch")

# if not project.REMOTE:
#     project.add_scp_step(exp, SCP_LOGIN, REMOTE_REPOS_DIR)



# Create a new experiment.
exp = Experiment(environment=ENV)
# Add solver to experiment and make it available to all runs.
exp.add_resource("parallel-scorpion", os.path.join(REPO, "fast-downward.py"))
# Add custom parser.
exp.add_parser(custom_parser.get_parser())


for algo in ALGORITHMS:
    for task in SUITE:
        run = exp.add_run()
        # Create a symbolic link and an alias. This is optional. We
        # could also use absolute paths in add_command().
        run.add_resource("task", task, symlink=True)
        run.add_command(
            "solve",
            ["{solver}", "--seed", str(SEED), "{task}", algo],
            time_limit=TIME_LIMIT,
            memory_limit=MEMORY_LIMIT,
        )
        # AbsoluteReport needs the following attributes:
        # 'domain', 'problem' and 'algorithm'.
        domain = os.path.basename(os.path.dirname(task))
        task_name = os.path.basename(task)
        run.set_property("domain", domain)
        run.set_property("problem", task_name)
        run.set_property("algorithm", algo)
        # BaseReport needs the following properties:
        # 'time_limit', 'memory_limit', 'seed'.
        run.set_property("time_limit", TIME_LIMIT)
        run.set_property("memory_limit", MEMORY_LIMIT)
        run.set_property("seed", SEED)
        # Every run has to have a unique id in the form of a list.
        run.set_property("id", [algo, domain, task_name])

# Add step that writes experiment files to disk.
exp.add_step("build", exp.build)

# Add step that executes all runs.
exp.add_step("start", exp.start_runs)

# Add step that parses the logs.
exp.add_step("parse", exp.parse)

# Add step that collects properties from run directories and
# writes them to *-eval/properties.
exp.add_fetcher(name="fetch")

# Make a report.
project.add_absolute_report(
    exp, attributes=ATTRIBUTES, filter=[project.add_evaluations_per_time]
)
# Parse the commandline and run the given steps.
exp.run_steps()