
from downward.reports.absolute import AbsoluteReport
from lab.experiment import Experiment
import project


ATTRIBUTES = [
    "coverage",
    "error",
    "initial_h_value",
    "last_runlog_line",
    "memory",
    "plan_length",
    "planner_exit_code",
    "planner_time",
    "planner_wall_clock_time",
    "run_dir",
    "total_time",
    "translator_memory",
    "translator_time_done",
    "num_slots",
#    "num_atoms",
    "registered_states",
    "avg_edges_per_state",
    "state_set_occupied_tree",
    "state_set_allocated_tree",
    "state_set_size",
    "score_planner_memory",
    "num_atoms"
]


import pathlib
BASE_PATH = pathlib.Path(__file__).parent.resolve() / 'data'

exp = Experiment(BASE_PATH)
# 2025-10-22-A-tree-compression-dbdt_s_short-eval
exp.add_fetcher(f'{BASE_PATH}/2025-10-22-A-tree-compression-dbdt_s_short-eval/', merge=True)

# 2025-10-22-A-tree-compression-packed_short
exp.add_fetcher(f'{BASE_PATH}/2025-10-22-A-tree-compression-packed_short-eval/', merge=True)

exp.add_report(AbsoluteReport(attributes=ATTRIBUTES))

# Parse the commandline and run the given steps.
exp.run_steps()

