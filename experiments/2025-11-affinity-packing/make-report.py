
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
    "score_planner_memory",
    "entries_in_state_set",
    "bins_per_entry",
    "avg_bins_per_state",
    "state_set_size",
    "lookup_structure_size",
    "state_registry_size",
    "bins_per_state",
    "num_operators",
    "num_fluents",
    "num_derived",
    "operator_touches_sum",
    "operator_touches_avg",
    "num_atoms",
    "registered_states",
    "memory_error",
]

import pathlib
BASE_PATH = pathlib.Path(__file__).parent.resolve() / 'data'

exp = Experiment(BASE_PATH)
# 2025-11-01-A-affinity-int-packer
exp.add_fetcher(f'{BASE_PATH}/2025-11-01-A-affinity-int-packer-eval/', filter_algorithm=[
    "affinity-int-packer-01-blind-dtdb_s_dev_dd", 
    "affinity-int-packer-02-scp-dtdb_s_dev_dd", 
    "affinity-int-packer-03-blind-dtdb_h_dev_dd", 
    "affinity-int-packer-04-scp-dtdb_h_dev_dd"
])

# 2025-11-03-A-affinity-int-packer-dev-dd
exp.add_fetcher(f'{BASE_PATH}/2025-11-03-A-affinity-int-packer-dev-dd-eval/', merge=True)

# 2025-11-04-A-affinity-int-packer-full
exp.add_fetcher(f'{BASE_PATH}/2025-11-04-A-affinity-int-packer-full-eval/', merge=True)

# 2025-11-06-A-packed-unpacked-eval
exp.add_fetcher(f'{BASE_PATH}/2025-11-06-A-packed-unpacked-eval/', merge=True)

# 2025-11-06-A-packed-unpacked-eval
exp.add_fetcher(f'{BASE_PATH}/2025-11-06-A-packed-unpacked-eval/', merge=True, filter_algorithm=[
    "affinity-int-packer-03-blind-pck",
    "affinity-int-packer-04-scp-pck"])

# 2025-11-14-A-packed-unpacked-eval
exp.add_fetcher(f'{BASE_PATH}/2025-11-14-A-packed-unpacked-eval/', merge=True)

exp.add_report(AbsoluteReport(attributes=ATTRIBUTES))

# Parse the commandline and run the given steps.
exp.run_steps()

