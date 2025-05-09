import pandas as pd
import numpy as np
import pathlib
from pathlib import Path
from glob import glob
import json
import os
from dataclasses import dataclass
import itertools
from collections import defaultdict
from typing import List, Dict, Any, Union,  Hashable, Set, Sequence, Tuple
from scipy.stats import entropy
import random 
import math
import sys

@dataclass
class Run:
    domain: str
    problem: str
    solved: bool
    var_df: pd.DataFrame
    pck_df: pd.DataFrame
    sequence: pd.Series
    domain_sizes: pd.Series

LeafValue = Union[Tuple[str, Any], None]  # (var, value) or None
NodeValue = Union[int, LeafValue]

class TreeCompressor:
    """
    Compresses fixed-size states with a given variable ordering via recursive subdivision.
    Every node is a tuple of exactly 2 elements, each a pointer (int) or (var, value) tuple (leaf).
    """

    def __init__(self, variable_ordering: Sequence[Union[str, int]]):
        self.var_order = list(variable_ordering)
        self.subtree_table: Dict[Hashable, int] = {}
        self.id_to_subtree: Dict[int, Tuple[NodeValue, NodeValue]] = {}
        self.next_id: int = 0
        # Statistics
        self.node_share_count: Dict[int, int] = defaultdict(int)
        self.node_share_levels: Dict[int, Set[int]] = defaultdict(set)

    def _get_subtree_id(self, node: Tuple[NodeValue, NodeValue], level: int) -> int:
        node_key = self._make_hashable(node)
        if node_key in self.subtree_table:
            node_id = self.subtree_table[node_key]
            self.node_share_count[node_id] += 1
            self.node_share_levels[node_id].add(level)
            return node_id
        else:
            node_id = self.next_id
            self.next_id += 1
            self.subtree_table[node_key] = node_id
            self.id_to_subtree[node_id] = node
            self.node_share_levels[node_id].add(level)
            return node_id

    def _make_hashable(self, obj: Any) -> Hashable:
        if isinstance(obj, (str, int, float, bool, type(None), np.int64, np.int32)):
            return obj
        elif isinstance(obj, tuple):
            return tuple(self._make_hashable(x) for x in obj)
        elif isinstance(obj, list):
            return tuple(self._make_hashable(x) for x in obj)
        elif isinstance(obj, dict):
            return tuple((k, self._make_hashable(obj[k])) for k in self.var_order if k in obj)
        else:
            raise TypeError(f"Unsupported state type: {type(obj)}")

    def compress(self, state: Union[List[Any], Dict[Any, Any]], ordering: List[Any] = None, level: int = 0) -> int:
        """
        Recursively compress state. Each node stores exactly two fields, 
        each a pointer or a (var, value) pair.
        """
        if ordering is None:
            ordering = self.var_order
        n = len(ordering)
        # Leaf cases: 1 or 2 variables
        if n == 0:
            # Empty slot
            return self._get_subtree_id((None, None), level)
        elif n == 1:
            idx = ordering[0]
            val = state[idx] if isinstance(state, list) else state[idx]
            return self._get_subtree_id(((idx, val), None), level)
        elif n == 2:
            idx1, idx2 = ordering
            val1 = state[idx1] if isinstance(state, list) else state[idx1]
            val2 = state[idx2] if isinstance(state, list) else state[idx2]
            return self._get_subtree_id(((idx1, val1), (idx2, val2)), level)
        else:
            # Split ordering approximately in half
            mid = n // 2
            left_order, right_order = ordering[:mid], ordering[mid:]
            left_id = self.compress(state, left_order, level + 1)
            right_id = self.compress(state, right_order, level + 1)
            return self._get_subtree_id((left_id, right_id), level)

    @property
    def node_count(self) -> int:
        return len(self.id_to_subtree)

    def compute_tree_statistics(self) -> dict:
        """
        Computes overall statistics for the compressed tree.

        Returns
        -------
        stats : dict
            Dictionary containing fields:
                - 'total_unique_nodes': int
                - 'shared_nodes': int           # nodes used more than once
                - 'total_node_reuse': int       # sum of all node shared counts
                - 'avg_times_node_is_shared': float   # average for nodes with share > 0
                - 'num_variables': int
        """
        total_unique_nodes = len(self.id_to_subtree)
        shared_counts = list(self.node_share_count.values())
        shared_nodes = sum(1 for count in shared_counts if count > 0)
        total_node_reuse = sum(shared_counts)
        avg_times_node_is_shared = (
            total_node_reuse / shared_nodes if shared_nodes else 0.0
        )
        num_variables = len(self.var_order) if hasattr(self, 'var_order') else 0

        return {
            "total_unique_nodes": total_unique_nodes,
            "shared_nodes": shared_nodes,
            "total_node_reuse": total_node_reuse,
            "avg_times_node_is_shared": avg_times_node_is_shared,
            "num_variables": num_variables,
        }

def read_states_run(run_path: pathlib.PosixPath):
    properties_file = run_path / "static-properties"
    state_file = run_path / "state_data.csv"
    plan_file = run_path / "sas_plan"

    print(properties_file, state_file,plan_file)
    print(os.path.isfile(properties_file), os.path.isfile(state_file))
    if not os.path.isfile(properties_file) or \
        not os.path.isfile(state_file):
        return pd.DataFrame()

    properties = None
    with open(properties_file) as file:
        properties = json.load(file)
    
    df = pd.read_csv(state_file)
    df['domain'] = properties['domain']
    df['problem'] = properties['problem']
    df['solved'] = os.path.isfile(plan_file)
    return df

def read_states_domain(exp_data: pathlib.PosixPath):
    search_glob = map(Path, glob(str(exp_data / "runs-*" / "*")))
    
    dfs = []
    for run_path in search_glob:
        dfs.append(read_states_run(run_path))
    return dfs
    
def preprocess_df(df) -> Run:
    """
    Preprocesses a DataFrame for a run, extracting domain/problem info,
    states, variable and packet dataframe slices, and prints debug output.

    Returns:
        Run: Dataclass with all extracted and printed fields.
    """
    # Basic info
    domain = df.loc[0, "domain"]
    problem = df.loc[0, "problem"]
    solved = df.loc[0, "solved"]

    # Find all columns between 'var0' and 'domain' (exclusive of 'domain')
    cols = df.columns.tolist()
    var0_idx = cols.index('var0')
    domain_idx = cols.index('domain')

    var_cols = cols[var0_idx:domain_idx]
    buff0_idx = cols.index('buff0')
    pck_cols = cols[buff0_idx:var0_idx]

    # Attempt to extract domain_sizes; assuming domain_sizes in row 0, var_cols
    domain_sizes = df[var_cols].iloc[0]
    var_df = df.loc[1:, var_cols]

    pck_df = df.loc[1:, pck_cols] if pck_cols else pd.DataFrame()

    sequence = df.loc[1:, 'sequence']
    return Run(domain, problem, solved, var_df, pck_df, sequence, domain_sizes)

def _get_block_columns(states, block):
    """Returns np.ndarray where each row is the values of one state for variables in block."""
    if isinstance(states[0], dict):
        return np.array([[s[k] for k in block] for s in states])
    else:
        return np.array([[s[k] for k in block] for s in states])

def _entropy(cols: np.ndarray) -> float:
    """Entropy of a multivariate block."""
    df = pd.DataFrame(cols)
    _, counts = np.unique(df.to_records(index=False), return_counts=True, axis=0)
    probs = counts / counts.sum()
    return entropy(probs, base=2)

def _mutual_information(states, block1, block2) -> float:
    """Compute mutual information I(block1; block2) from the dataset of states."""
    X = _get_block_columns(states, block1)
    Y = _get_block_columns(states, block2)
    XY = np.hstack((X, Y))
    H_X = _entropy(X)
    H_Y = _entropy(Y)
    H_XY = _entropy(XY)
    # MI = H(X) + H(Y) - H(XY)
    return H_X + H_Y - H_XY

def greedy_mi_max_variable_order(states: List[Union[dict,list]], variables: List[Any]=None) -> List[Any]:
    """
    Greedily merges variable blocks that have the highest mutual information.
    
    Parameters
    ----------
    states : list of dict or list
        States as dicts or lists.
    variables : list, optional
        If not provided, inferred from data.
    
    Returns
    -------
    List[Any]
        Optimal variable reordering (high shared information earlier).
    """
    if not states:
        return []
    first = states[0]
    if variables is None:
        variables = list(first.keys()) if isinstance(first, dict) else list(range(len(first)))
    blocks = [[v] for v in variables]
    while len(blocks) > 1:
        best_mi = None
        best_pair = None
        for i, j in itertools.combinations(range(len(blocks)), 2):
            mi = _mutual_information(states, blocks[i], blocks[j])
            if best_mi is None or mi > best_mi:
                best_mi = mi
                best_pair = (i, j)
        # Merge the best pair
        i, j = best_pair
        merged_block = blocks[i] + blocks[j]
        # Remove both, append merged
        blocks = [blocks[k] for k in range(len(blocks)) if k not in (i,j)] + [merged_block]
    return blocks[0]

def _states_projection(states, block):
    first = states[0]
    if isinstance(first, dict):
        return [tuple(s[k] for k in block) for s in states]
    else:
        return [tuple(s[k] for k in block) for s in states]

def _unique_count(states, blocks):
    projections = []
    first = states[0]
    for s in states:
        row = []
        for block in blocks:
            if isinstance(first, dict):
                row.append(tuple(s[k] for k in block))
            else:
                row.append(tuple(s[k] for k in block))
        projections.append(tuple(row))
    return len(set(projections))

def globally_worst_variable_order(states: List[Union[dict, list]], variables: List[Any] = None) -> List[Any]:
    """
    Greedily merges the pair of variable blocks that results in the LARGEST number
    of unique projected subtrees at each step (best for fragmentation/worst for compression).
    
    Returns
    -------
    List[Any]
        Variable order minimizing subtree sharing for the supplied states.
    """
    if not states:
        return []
    first = states[0]
    if variables is None:
        variables = list(first.keys()) if isinstance(first, dict) else list(range(len(first)))
    blocks = [[v] for v in variables]
    while len(blocks) > 1:
        worst_score = None
        worst_i, worst_j = None, None
        for i, j in itertools.combinations(range(len(blocks)), 2):
            merged = [blocks[k] for k in range(len(blocks)) if k not in (i, j)] + [blocks[i] + blocks[j]]
            score = _unique_count(states, merged)
            if worst_score is None or score > worst_score:
                worst_score = score
                worst_i, worst_j = i, j
        # Merge worst pair
        new_block = blocks[worst_i] + blocks[worst_j]
        blocks = [blocks[k] for k in range(len(blocks)) if k not in (worst_i, worst_j)] + [new_block]
    return blocks[0]


def greedy_variable_ordering_tree(states: List[Union[dict, list]], variables: List[Any] = None) -> List[Any]:
    """
    Globally greedy pairwise merging of variable blocks:
    At each step, merges the pair of variable blocks resulting in the smallest
    number of unique projected subtrees. Not restricted to adjacent blocks.

    Returns
    -------
    List[Any]
        Variable order maximizing subtree sharing for the supplied states.
    """
    if not states:
        return []
    if variables is None:
        first = states[0]
        variables = list(first.keys()) if isinstance(first, dict) else list(range(len(first)))
    # Start with each variable as its own block
    blocks = [[v] for v in variables]
    while len(blocks) > 1:
        best_score = None
        best_i, best_j = None, None
        for i, j in itertools.combinations(range(len(blocks)), 2):
            merged = [blocks[k] for k in range(len(blocks)) if k not in (i, j)] + [blocks[i] + blocks[j]]
            score = _unique_count(states, merged)
            if best_score is None or score < best_score:
                best_score = score
                best_i, best_j = i, j
        # Merge best pair
        new_block = blocks[best_i] + blocks[best_j]
        blocks = [blocks[k] for k in range(len(blocks)) if k not in (best_i, best_j)] + [new_block]
    return blocks[0]
    from typing import List, Dict, Any, Sequence, Union

def greedy_variable_ordering_single(
    states: List[Union[dict, list]],
    variables: Sequence[Any] = None
) -> List[Any]:
    """
    Greedily order variables to maximize subtree sharing,
    using variable names/keys rather than relying on index.

    Converts all states to dicts with variable keys for robust access.

    Parameters
    ----------
    states : List[dict or list]
        The collection of states to be compressed.
    variables : Sequence[Any], optional
        Which variables to consider. If None, inferred from the first state.

    Returns
    -------
    ordering : List[Any]
        Greedy ordering of variable keys.
    """
    if not states:
        return []

    # Standardize input to list of dicts mapping key/index -> value
    if isinstance(states[0], dict):
        states_dicts = [s.copy() for s in states]  # Safe shallow copy
        if variables is None:
            variables = list(states_dicts[0].keys())
    else:
        if variables is None:
            variables = list(range(len(states[0])))
        states_dicts = [{k: v for k, v in zip(variables, s)} for s in states]

    remaining_vars = set(variables)
    ordering = []
    sub_states = states_dicts

    while remaining_vars:
        best_var = None
        best_score = None

        for var in remaining_vars:
            groups = {}
            for s in sub_states:
                v = s[var]
                groups.setdefault(v, []).append(s)
            # Unique sub-states remaining after removing this variable
            unique_substates = set()
            for group in groups.values():
                for g in group:
                    key_tuple = tuple(
                        sorted(
                            (k, g[k]) for k in g if k != var
                        )
                    )
                    unique_substates.add(key_tuple)
            score = len(unique_substates)
            if best_score is None or score < best_score:
                best_var = var
                best_score = score
        ordering.append(best_var)
        remaining_vars.remove(best_var)
        sub_states = [{k: v for k, v in s.items() if k != best_var}
                      for s in sub_states]

    return ordering

    
def pack_values_to_uint32(values: list, domain_sizes: list) -> list:
    """
    Pack a list of integer values into as few uint32s as possible, with each value using
    only as many bits as needed, and no value spanning across two uint32s.
    """
    assert len(values) == len(domain_sizes), "Lists must be the same length."
    bit_lengths = [math.ceil(math.log2(max(1, dom_size))) for dom_size in domain_sizes]
    packed = []
    current = 0  # current uint32 accumulator
    bits_used = 0  # bits currently used in this uint32
    for v, bits in zip(values, bit_lengths):
        if bits_used + bits > 32:
            packed.append(current & 0xFFFFFFFF)
            current = 0
            bits_used = 0
        current |= (v & ((1 << bits) - 1)) << bits_used
        bits_used += bits
    if bits_used > 0:
        packed.append(current & 0xFFFFFFFF)
    return packed

def aggregate_random_order_stats(run, states, representation, n_random=10, random_seed=42):
    """
    For one run and set of states, generate statistics over N random variable orderings.
    Aggregates with mean, std, min, max.

    Returns: list of dicts, one per aggregate (mean/std/min/max) for this run.
    """
    import numpy as np
    import random
    from collections import defaultdict

    rng = random.Random(random_seed)
    num_vars = len(states[0])
    field_collections = defaultdict(list)
    for _ in range(n_random):
        ordering = rng.sample(list(range(num_vars)), num_vars)
        compressor = TreeCompressor(ordering)
        for state in states:
            compressor.compress(state)
        stats = compressor.compute_tree_statistics()
        for key, value in stats.items():
            field_collections[key].append(value)
    results = []
    for agg_func, funcname in [
        (np.mean, "mean"),
        (np.std, "std"),
        (np.min, "min"),
        (np.max, "max"),
    ]:
        agg_stats = {field: float(agg_func(vals)) for field, vals in field_collections.items()}
        results.append({
            "domain": run.domain,
            "problem": run.problem,
            "solved": run.solved,
            "ordering_name": f"Random ({funcname}, N={n_random})",
            "representation": representation,
            "ordering": None,
            **agg_stats,
        })
    return results



def gather_statistics(run: list) -> list:
    """
    For each Run, iterates through different variable ordering strategies,
    compresses the states, collects tree and ordering statistics.
    Returns a list of result dicts (one per run/ordering).
    """
    results = []
    order_methods = [
        ('Identity',                          lambda run: list(range(run.shape[1]))),
        ('Greedy Variable Ordering',          lambda run: greedy_variable_ordering_single(run.values.tolist())),
        ('Greedy Variable Tree Ordering',     lambda run: greedy_variable_ordering_tree(run.values.tolist())),
        # ('Greedy Max Entropy Variable Order', lambda run: globally_worst_variable_order(run.var_df.values.tolist())),
        ('Greedy Mutual Information',         lambda run: greedy_mi_max_variable_order(run.values.tolist())),
    ]
    
    states_unpacked = run.var_df.values.tolist()
    # Packed states: each state packed per domain_sizes
    states_packed = [pack_values_to_uint32(state, run.domain_sizes) for state in states_unpacked]
    
    for representation, states in [("unpacked", states_unpacked), ("packed", states_packed)]:
        states = run.var_df.values.tolist()  # States as lists
        for order_name, order_fn in order_methods:
            ordering = order_fn(run.var_df.head(5000))
            compressor = TreeCompressor(ordering)
            for state in states[5000:15000]:
                compressor.compress(state)
            stats = compressor.compute_tree_statistics()
            results.append({
                "domain": run.domain,
                "problem": run.problem,
                "solved": run.solved,
                "ordering_name": order_name,
                "representation": representation,
                "ordering": ordering,
                **stats,  # flatten the stats into the dict
            })

        # Aggregate over random orderings
        agg_random_rows = aggregate_random_order_stats(run, states, representation, n_random=20, random_seed=42)
        results.extend(agg_random_rows)
    
    return results

def main(run_path=Path()):
    # You can replace this path with an experiment folder
    run_df = read_states_run(run_path)
    run_df = preprocess_df(run_df)
    # Gather statistics
    stats = gather_statistics(run_df)
    # Print or save as DataFrame/JSON
    stats_df = pd.DataFrame(stats)
    # Optionally save
    stats_df.to_csv(run_path / "statistics_results.csv", index=False)
    # or: with open("statistics_results.json", "w") as f: json.dump(stats, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        main(Path("."))
    else:
        main(Path(sys.argv[1]))