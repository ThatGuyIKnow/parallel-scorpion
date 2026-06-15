"""Parser for the parallel-scorpion HTG experiment.

Extends the parallel custom parser with the metrics distributed-tyr reports, so
the two planners can be compared on the same attributes (no overapproximation
ratio, which tyr-specific). Emitted per run:

  coverage                    1 iff a plan was found
  oom / oot                   out-of-memory / out-of-time classification
  total_peak_memory           sum of per-rank "Peak memory: N KB"
  memory_mb                   total_peak_memory in MB
  score_peak_memory           IPC-style log score in [0,1], bounds [2 MB, 32 GB]
  off_parent_transfer_rate    communication overhead (CO), parsed from the
                              "Off-parent transfer rate: a/b = r" line
  num_off_parent_transfers    a   (generated nodes sent to another rank)
  num_generated_co            b   (generated nodes, for CO)
  total_expansions            summed expansions across ranks
  search_time                 max per-rank "Actual search time"

"additional_expansions" (= search overhead, expansions@Nw / expansions@1w - 1)
is a cross-run quantity and is computed in the report by pairing each
multi-worker run with its single-worker baseline.
"""
import logging
import math
import re

from lab.parser import Parser
from downward import outcomes

# Memory-score bounds, matching distributed-tyr (2 MB best .. 32 GB worst).
MEM_SCORE_LOWER_BYTES = 2_000_000
MEM_SCORE_UPPER_BYTES = 32_000_000_000


def compute_log_score(success, value, lower, upper):
    """IPC-style log score in [0, 1] (higher is better); 0 on failure."""
    if not success or value is None:
        return 0.0
    if value <= lower:
        return 1.0
    if value >= upper:
        return 0.0
    raw = (math.log(value) - math.log(lower)) / (math.log(upper) - math.log(lower))
    return 1.0 - raw


def parse_exit_code_and_outcome(content, props):
    """Human-readable error + coverage / oom / oot from the search exit code."""
    use_legacy_exit_codes = True
    for line in content.splitlines():
        if line.startswith("translate exit code:") or line.startswith("search exit code:"):
            use_legacy_exit_codes = False
            break

    exitcode = props.get("planner_exit_code")
    outcome = outcomes.get_outcome(exitcode, use_legacy_exit_codes)
    props["error"] = outcome.msg
    msg = outcome.msg

    props["coverage"] = int(msg == "success")
    props["unsolvable"] = int(msg in ["unsolvable", "translate-unsolvable", "search-unsolvable"])
    props["oom"] = int("out-of-memory" in msg)
    props["oot"] = int("out-of-time" in msg or "timeout" in msg)
    if not outcome.explained:
        props.add_unexplained_error(msg)


def derive_metrics(content, props):
    # Total peak memory across ranks (KB -> derived bytes/MB + score).
    peaks = props.get("peak_memory_per_rank", [])
    if peaks:
        total_kb = sum(peaks)
        props["total_peak_memory"] = total_kb
        props["memory_mb"] = total_kb / 1000.0
        props["score_peak_memory"] = compute_log_score(
            props.get("coverage", 0) == 1,
            total_kb * 1000,  # KB -> bytes
            MEM_SCORE_LOWER_BYTES,
            MEM_SCORE_UPPER_BYTES,
        )

    # Max per-rank actual search time (wall time of the expansion loop).
    times = props.get("actual_search_time_per_rank", [])
    if times:
        props["search_time"] = max(times)


class CommonParser(Parser):
    def add_repeated_pattern(self, name, regex, file="run.log", required=False, type=int):
        def find_all_occurences(content, props):
            matches = re.findall(regex, content)
            if required and not matches:
                logging.error(f"Pattern {regex} not found in file {file}")
            props[name] = [type(m) for m in matches]

        self.add_function(find_all_occurences, file=file)


def get_parser():
    parser = CommonParser()
    parser.add_pattern(
        "planner_exit_code",
        r"(?:search exit code: (\d+)\n|Exit code:\s*(\d+)\n)",
        type=int,
    )
    # Communication overhead (paper CO) printed by rank 0 at termination.
    parser.add_pattern(
        "num_off_parent_transfers",
        r"Off-parent transfer rate: (\d+)/\d+ = [\d.]+",
        type=int,
    )
    parser.add_pattern(
        "num_generated_co",
        r"Off-parent transfer rate: \d+/(\d+) = [\d.]+",
        type=int,
    )
    parser.add_pattern(
        "off_parent_transfer_rate",
        r"Off-parent transfer rate: \d+/\d+ = ([\d.]+)",
        type=float,
    )
    parser.add_pattern(
        "total_expansions",
        r"Total expansions: (\d+)",
        type=int,
    )
    parser.add_repeated_pattern(
        "peak_memory_per_rank",
        r"Peak memory: (\d+) KB",
        type=int,
    )
    parser.add_repeated_pattern(
        "actual_search_time_per_rank",
        r"Actual search time: (\d+\.\d+)s",
        type=float,
    )
    parser.add_function(parse_exit_code_and_outcome)
    parser.add_function(derive_metrics)
    return parser
