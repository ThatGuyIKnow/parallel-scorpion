

import logging
import re

from lab.parser import Parser
from lab import tools


def add_planner_memory_score(content, props):
    success = props["coverage"] or props["unsolvable"]
    memory_limit_kb = 8388608  # 8 GB in KB
    props["score_planner_memory"] = tools.compute_log_score(
        success,
        props.get("planner_memory"),
        lower_bound=2000,
        upper_bound=memory_limit_kb,
        )

class CommonParser(Parser):
    def add_repeated_pattern(
        self, name, regex, file="run.log", required=False, type=int
    ):
        def find_all_occurences(content, props):
            matches = re.findall(regex, content)
            if required and not matches:
                logging.error(f"Pattern {regex} not found in file {file}")
            props[name] = [type(m) for m in matches]

        self.add_function(find_all_occurences, file=file)

    def add_bottom_up_pattern(
        self, name, regex, file="run.log", required=False, type=int
    ):
        def search_from_bottom(content, props):
            reversed_content = "\n".join(reversed(content.splitlines()))
            match = re.search(regex, reversed_content)
            if required and not match:
                logging.error(f"Pattern {regex} not found in file {file}")
            if match:
                props[name] = type(match.group(1))

        self.add_function(search_from_bottom, file=file)

def assign_memory_errors(content, props):
    mem_err_codes = ["250", ]
    if "planner_exit_code" in mem_err_codes:
        props["error"] = "search-out-of-memory"
        props["unexplained_errors"] = []


def get_parser():
    parser = CommonParser()
    parser.add_function(add_planner_memory_score)
    parser.add_pattern(
        "entries_in_state_set",
        r"\[t=.+s, \d+ KB\] Entries in state set: (\d+)",
        type=int)
    parser.add_pattern(
        "bins_per_entry",
        r"\[t=.+s, \d+ KB\] Bins per entry: (\d+)",
        type=int)
    parser.add_pattern(
        "avg_bins_per_state",
        r"\[t=.+s, \d+ KB\] Average bins per state: (\d+\.\d+)",
        type=float)
    parser.add_pattern(
        "state_set_size",
        r"\[t=.+s, \d+ KB\] State set size: (\d+) B",
        type=int)
    parser.add_pattern(
        "lookup_structure_size",
        r"\[t=.+s, \d+ KB\] Lookup structure size: (\d+) B",
        type=int)
    parser.add_pattern(
        "state_registry_size",
        r"\[t=.+s, \d+ KB\] State registry size: (\d+) B",
        type=int)
    parser.add_pattern(
        "bins_per_state",
        r"\[t=.+s, \d+ KB\] Number of bins in state: (\d+)",
        type=int)
    parser.add_pattern(
        "num_operators",
        r"\[t=.+s, \d+ KB\] Number of operators: (\d+)",
        type=int)
    parser.add_pattern(
        "num_fluents",
        r"\[t=.+s, \d+ KB\] Number of fluents: (\d+)",
        type=int)
    parser.add_pattern(
        "num_derived",
        r"\[t=.+s, \d+ KB\] Number of derived: (\d+)",
        type=int)
    parser.add_pattern(
        "operator_touches_full",
        r"\[t=.+s, \d+ KB\] Number of operator touches nodes: (\[(?:\d+, )*\])",
        type=str)
    parser.add_pattern(
        "operator_touches_sum",
        r"\[t=.+s, \d+ KB\] Total number of operator touches: (\d+)",
        type=int)
    parser.add_pattern(
        "operator_touches_avg",
        r"\[t=.+s, \d+ KB\] Average number of operator touches: (\d+\.\d+)",
        type=float)
    
    parser.add_pattern(
        "num_atoms",
        r"Translator variables: (\d+)",
        type=int)
    parser.add_pattern(
        "registered_states",
        r"\[t=.+s, \d+ KB\] Number of registered states: (\d+)",
        type=int)
    parser.add_pattern(
        "memory_error",
        r"(Failed to allocate memory)",
        type=bool,
    )

    parser.add_function(assign_memory_errors)

    return parser

