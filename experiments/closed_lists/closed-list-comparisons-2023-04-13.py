#! /usr/bin/env python

import os
import shutil

import project
from parser import CommonParser

REPO = project.get_repo_base()
BENCHMARKS_DIR = os.environ["DOWNWARD_BENCHMARKS"]
# If REVISION_CACHE is None, the default ./data/revision-cache is used.
REVISION_CACHE = os.environ.get("DOWNWARD_REVISION_CACHE")

if project.REMOTE:
    ENV = project.TetralithEnvironment(extra_options="#SBATCH -A snic2022-5-341",memory_per_cpu="5G")
    SUITE = project.SUITE_OPTIMAL_STRIPS
else:
    ENV = project.LocalEnvironment(processes=2)
    SUITE = ["depot:p01.pddl","depot:p02.pddl", "grid:prob01.pddl", "gripper:prob01.pddl", "gripper:prob02.pddl", "gripper:prob03.pddl", "gripper:prob04.pddl", "gripper:prob05.pddl"]

CLOSED_LISTS = [
    ("no closed", "astar", ""),
    ("loes", "astarmod", "loes(samples=1000)"),
    ("cloes", "astarmod", "cloes(samples=1000)"),
    ("bdd", "astarmod", "bdd"),
]
HEURISTICS = [
    ("blind", ""),
    # ("pdb", ""),
    # ("hmax", ""),
    # ("cegar", ""),
    # ("merge_and_shrink", ", shrink_strategy=shrink_bisimulation(greedy=false)," 
    # + "merge_strategy=merge_sccs(order_of_sccs=topological,"
    # + "merge_selector=score_based_filtering(scoring_functions=[goal_relevance,dfp,total_order])),"
    # + "label_reduction=exact(before_shrinking=true,before_merging=false),max_states=50k,threshold_before_merge=1"),
    #("Potential Heuristics????????", ""),
]
CONFIGS = [
    (f"{index:02d}-{nick}-{heu}", ["--search", f"{algo}({heu}(cache_estimates=false{h_opts}),{closed})"])
    for index, ((nick, algo, closed), (heu , h_opts))  in enumerate(
        [(c, h) for c in CLOSED_LISTS for h in HEURISTICS],
        start=1,
    )
]

BUILD_OPTIONS = [] #debug is for some reason needed here because LOES is 2x as slow without it 
DRIVER_OPTIONS = ["--overall-time-limit", "30m", "--overall-memory-limit", "4G"] 
REVS = [
    ("symk-loes", "symk-loes"),
]

ATTRIBUTES =  [ 
    'cost',
    'coverage',
    'dead_ends',
    "search_start_time",
    "search_start_memory",
    'evaluations',
    'expansions',
    'generated',
    'initial_h_value',
    'plan_length',
    'planner_time',
    'score_*',
    'search_time',
    'total_time',
    'unsolvable',
    'error',
    'planner_wall_clock_time',
    'raw_memory',
    "memory",
    "closed_states",
    "open_states",
    "max_open_states",
    project.EVALUATIONS_PER_TIME,
    project.PROPORTION_CLOSED,
]

exp = project.FastDownwardExperiment(environment=ENV, revision_cache=REVISION_CACHE)
for config_nick, config in CONFIGS:
    for rev, rev_nick in REVS:
        algo_name = f"{rev_nick}:{config_nick}" if rev_nick else config_nick
        exp.add_algorithm(
            algo_name,
            REPO,
            rev,
            config,
            build_options=BUILD_OPTIONS,
            driver_options=DRIVER_OPTIONS,
        )

exp.add_suite(BENCHMARKS_DIR, SUITE)

exp.add_parser(exp.EXITCODE_PARSER)
exp.add_parser(exp.TRANSLATOR_PARSER)
exp.add_parser(exp.SINGLE_SEARCH_PARSER)
exp.add_parser(CommonParser())
exp.add_parser(exp.PLANNER_PARSER)

exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)

exp.add_fetcher(name="fetch")

def algos_with_name_filter(name):
    def filt(run):
        return name in run["algorithm"]
    return filt
exp.add_report(project.AbsoluteReport(attributes=ATTRIBUTES, filter=[project.add_proportion_closed, project.add_evaluations_per_time]), name="report", outfile="report.html")
exp.add_report(project.AbsoluteReport(attributes=ATTRIBUTES, filter=[project.add_proportion_closed, project.add_evaluations_per_time, algos_with_name_filter("cloes")]), name="report2", outfile="report2.html")
#exp.add_step("open-report", project.subprocess.call, ["xdg-open", "/home/hugo/symk/symk-loes/experiments/cg-vs-ff/data/closed-list-comparisons-eval/report.html"])



exp.run_steps()