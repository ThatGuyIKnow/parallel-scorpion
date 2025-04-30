#! /usr/bin/env python

import os
import sys
import csv
from pathlib import Path, PosixPath
from typing import List
from downward.experiment import FastDownwardExperiment, FastDownwardAlgorithm, FastDownwardRun
from downward.suites import build_suite
from downward.cached_revision import CachedFastDownwardRevision
from lab.experiment import Experiment
from lab.environments import TetralithEnvironment, LocalEnvironment
import state_analytics

import project

def combine_csvs(input_files: List[str], output_file: str, strict_headers: bool = False) -> None:
    """
    Appends multiple CSV files into one output CSV file efficiently.
    Only the header from the first input file is written.
    
    Args:
        input_files (List[str]): List of paths to CSV files to combine
        output_file (str): Path to the combined CSV file (created or appended)
        strict_headers (bool): If True, raise error on header mismatch; If False, warn but proceed
    """
    if not input_files:
        raise ValueError("No input files provided.")

    output_file_exists = os.path.isfile(output_file)
    wrote_header = output_file_exists and os.path.getsize(output_file) > 0
    expected_header = None

    with open(output_file, 'a', newline='', encoding='utf-8') as fout:
        writer = None
        for idx, infile in enumerate(input_files):
            with open(infile, 'r', newline='', encoding='utf-8') as fin:
                reader = csv.reader(fin)
                try:
                    header = next(reader)
                except StopIteration:
                    print(f"Warning: File '{infile}' is empty, skipping.")
                    continue
                # On the first input file, or if output_file didn't exist, set header
                if expected_header is None:
                    expected_header = header
                    if not wrote_header:
                        writer = csv.writer(fout)
                        writer.writerow(header)
                        wrote_header = True
                else:
                    # Check header consistency
                    if header != expected_header:
                        msg = f"Header mismatch: Input file '{infile}' header {header} != expected {expected_header}"
                        if strict_headers:
                            raise ValueError(msg)
                        else:
                            print("Warning:", msg)
                if writer is None:
                    writer = csv.writer(fout)
                # Write the data rows (skip header)
                for row in reader:
                    writer.writerow(row)

def combine_results(exp_data: PosixPath):
    search_glob = map(Path, glob(str(exp_data / "runs-*" / "statistics_results.csv")))
    
    combine_csvs(search_glob, 'experiment_state_statistics.csv')
    return dfs

REVISION_CACHE = (
        os.environ.get("DOWNWARD_REVISION_CACHE") or project.DIR / "data" / "revision-cache"
)
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
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

exp.add_resource("state_analytics_exec", os.path.join(SCRIPT_DIR, "state_analytics.py"))

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
            run.add_command("state_analytics", [sys.executable, "{state_analytics_exec}"])
            
            exp.add_run(run)

exp.add_parser(FastDownwardExperiment.EXITCODE_PARSER)
exp.add_parser(FastDownwardExperiment.TRANSLATOR_PARSER)
exp.add_parser(FastDownwardExperiment.SINGLE_SEARCH_PARSER)
exp.add_parser(FastDownwardExperiment.PLANNER_PARSER)

exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("fetch_analytics", combine_results)

# Parse the commandline and run the given steps.
exp.run_steps()
