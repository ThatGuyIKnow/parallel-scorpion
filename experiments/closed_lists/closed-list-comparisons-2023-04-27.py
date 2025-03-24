#! /usr/bin/env python

import os
import shutil

import project

REPO = project.get_repo_base()
BENCHMARKS_DIR = os.environ["DOWNWARD_BENCHMARKS"]
# If REVISION_CACHE is None, the default ./data/revision-cache is used.
REVISION_CACHE = os.environ.get("DOWNWARD_REVISION_CACHE")

if project.REMOTE:
    ENV = project.TetralithEnvironment(email="hugax961@student.liu.se", extra_options="#SBATCH -A snic2022-5-341",memory_per_cpu="2G", time_limit_per_task="30:00:00")
    SUITE = project.SUITE_OPTIMAL_STRIPS
else:
    ENV = project.LocalEnvironment(processes=2)
    SUITE = ["depot:p01.pddl","depot:p02.pddl", "grid:prob01.pddl", "gripper:prob01.pddl", "gripper:prob02.pddl", "gripper:prob03.pddl", "gripper:prob04.pddl", "gripper:prob05.pddl",
            "airport:p01-airport1-p1.pddl", "airport:p04-airport2-p1.pddl", "airport:p03-airport1-p2.pddl", "airport:p05-airport2-p1.pddl", "airport:p06-airport2-p2.pddl", "airport:p07-airport2-p2.pddl", "airport:p08-airport2-p3.pddl", "airport:p10-airport3-p1.pddl", "airport:p11-airport3-p1.pddl", "airport:p12-airport3-p2.pddl", "airport:p13-airport3-p2.pddl", "airport:p14-airport3-p3.pddl", "airport:p15-airport3-p3.pddl"]

CLOSED_LISTS = [
    ("no-closed", "astar", ""),
    ("loes", "astarmod", "loes(samples=1000)"),
    ("cloes", "astarmod", "cloes(samples=1000)"),
    ("bdd", "astarmod", "bdd(cache_size=16384)"),
]

HEURISTICS = [
    ("hmax", ""),
    ("cegar", ""),
    ("merge_and_shrink", ", shrink_strategy=shrink_bisimulation(greedy=false)," 
    + "merge_strategy=merge_sccs(order_of_sccs=topological,"
    + "merge_selector=score_based_filtering(scoring_functions=[goal_relevance,dfp,total_order])),"
    + "label_reduction=exact(before_shrinking=true,before_merging=false),max_states=50k,threshold_before_merge=1"),
]
CONFIGS = [
    (f"{index:02d}-{nick}-{heu}", ["--search", f"{algo}({heu}(cache_estimates=false{h_opts}),{closed})"])
    for index, ((nick, algo, closed), (heu , h_opts))  in enumerate(
        [(c, h) for c in CLOSED_LISTS for h in HEURISTICS],
        start=1,
    )
]

BUILD_OPTIONS = []
DRIVER_OPTIONS = ["--overall-time-limit", "2h", "--overall-memory-limit", "1G"] 
REVS = [
    ("origin/master", None),
]

ATTRIBUTES =  [ 
    'cost',
    'coverage',
    'dead_ends',
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
    project.PROPORTION_MAX_CLOSED,
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
exp.add_parser(project.DIR / "parser.py")
exp.add_parser(exp.PLANNER_PARSER)

exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)

exp.add_fetcher(name="fetch")

def algos_with_name_filter(name):
    def filt(run):
        return name in run["algorithm"]
    return filt
def algos_with_names_filter(name1, name2):
    def filt(run):
        return name1 in run["algorithm"] or name2 in run["algorithm"]
    return filt
exp.add_report(project.AbsoluteReport(attributes=ATTRIBUTES, filter=[project.add_proportion_closed, project.add_evaluations_per_time]), name="full_report", outfile="full_report.html")



HEURISTICS
for closed, unused1, unused2 in CLOSED_LISTS:
    exp.add_report(project.AbsoluteReport(attributes=ATTRIBUTES, filter=[project.add_proportion_closed, project.add_evaluations_per_time, algos_with_name_filter(f"-{closed}-")]), name=f"report-{closed}", outfile=f"report-{closed}.html")
for h, unused1 in HEURISTICS:
    exp.add_report(project.AbsoluteReport(attributes=ATTRIBUTES, filter=[project.add_proportion_closed, project.add_evaluations_per_time, algos_with_name_filter(f"-{h}")]), name=f"report-{h}", outfile=f"report-{h}.html")


attributes = ["memory"]

pairs = [
    (f"-no-closed-{h}", f"-loes-{h}") for h, unused1 in HEURISTICS
    #(f"-no-closed-{h}", f"-cloes-{h}") for h, unused1 in HEURISTICS,
    #(f"-no-closed-{h}", f"-bdd-{h}") for h, unused1 in HEURISTICS
]
print(pairs)
suffix = "-rel" if project.RELATIVE else ""
for algo1, algo2 in pairs:
    for attr in attributes:
        for form in ["tex", "png"]:
            exp.add_report(
                project.ScatterPlotReport(
                    relative=project.RELATIVE,
                    get_category=lambda run1, run2: run1["domain"],
                    attributes=[attr],
                    filter=[project.add_proportion_closed, project.add_evaluations_per_time, algos_with_names_filter(algo1, algo2)],
                    format=form,
                ),
                name=f"{exp.name}{algo1}vs{algo2}{attr}{suffix}{form}",
            )




def attr_comparison_filter(name1, name2, attr1, attr2, temp_attr = "temp-attr"):
    def filt(run):
        if name1 in run["algorithm"]:
            run[temp_attr] = run.get(attr1) 
            return run
        if name2 in run["algorithm"]:
            run[temp_attr] = run.get(attr2)
            return run
        return False
    return filt
        

exp.run_steps()
