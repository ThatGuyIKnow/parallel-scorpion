from downward.reports.absolute import AbsoluteReport
from lab.experiment import Experiment
import project
import pathlib

def fetch_experiement_data(exp_name: str, kwargs) -> Experiment:
    data_dir = project.DIR / "data" / exp_name
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Experiment data directory {data_dir} does not exist.")
    exp.add_fetcher(data_dir, **kwargs)

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
    "num_atoms",
    "registered_states",
    "avg_edges_per_state",
    "state_set_occupied_tree",
    "state_set_allocated_tree",
    "state_set_size",
    "score_planner_memory"
]

BASE_PATH = pathlib.Path(__file__).parent.resolve() / 'data'

exp = Experiment(BASE_PATH)

fetch_experiement_data("2025-10-14-A-tree-compression-new-packed", merge=True)
fetch_experiement_data("2025-10-14-B-tree-compression-old-packed", merge=True)
fetch_experiement_data("2025-10-14-C-packed_representation", merge=True)

exp.add_reporter(AbsoluteReport(attributes=ATTRIBUTES))

exp.run_steps()