import logging
import re

from lab.parser import Parser
from downward import outcomes

def parse_exit_code(content, props):
    """
    Convert the exitcode of the planner to a human-readable message and store
    it in props['error']. Additionally, if there was an unexplained error, add
    its source to the list at props['unexplained_errors'].

    For unexplained errors please check the files run.log, run.err,
    driver.log and driver.err to find the reason for the error.

    """
    assert "error" not in props

    # Check if Fast Downward uses the latest exit codes.
    use_legacy_exit_codes = True
    for line in content.splitlines():
        if line.startswith("translate exit code:") or line.startswith(
            "search exit code:"
        ):
            use_legacy_exit_codes = False
            break

    exitcode = props.get("planner_exit_code")
    outcome = outcomes.get_outcome(exitcode, use_legacy_exit_codes)
    props["error"] = outcome.msg
    if use_legacy_exit_codes:
        props["unsolvable"] = int(outcome.msg == "unsolvable")
    else:
        props["unsolvable"] = int(
            outcome.msg in ["translate-unsolvable", "search-unsolvable"]
        )
    if not outcome.explained:
        props.add_unexplained_error(outcome.msg)

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


def get_parser():
    parser = CommonParser()
    parser.add_bottom_up_pattern(
        "search_start_time",
        r"\[t=(.+)s, \d+ KB\] g=0, 1 evaluated, 0 expanded",
        type=float,
    )
    parser.add_bottom_up_pattern(
        "search_start_memory",
        r"\[t=.+s, (\d+) KB\] g=0, 1 evaluated, 0 expanded",
        type=int,
    )
    parser.add_pattern(
        "planner_exit_code",
        r"(?:search exit code: (\d+)\n|Exit code:\s*(\d+)\n)",
        type=int,
    )
    parser.add_function(parse_exit_code)

    parser.add_repeated_pattern(
        "peak_memory",
        r"Peak memory: (\d+) KB",
        type=int,
    )
    parser.add_repeated_pattern(
        "actual_search_time",
        r"Actual search time: (\d+\.\d+)s",
        type=float,
    )
    return parser