# Parallel-Scorpion HTG Experiment — Design

**Date:** 2026-06-15
**Author:** Oliver Harold Joergensen (with Claude Code)
**Status:** Approved direction; pending spec review

## 1. Goal

Get scorpion's MPI parallel best-first search working, add the paper's best
work-distribution method (GRAZHDA*/sparsity), and produce a Downward-Lab
experiment — mirroring `distributed-tyr-c/experiments/full-suite-experiment/` —
that runs **GBFS + FF** at **1 and 16 workers** on the **HTG** benchmark under a
**32 GB / 10 min** budget, emitting the *same* metrics distributed-tyr reports
(coverage, OOM, OOT, expansions, additional expansions, off-parent transfer
rate, peak memory, memory score, search time). Validate parsing on a small set
locally; run the full comparison on Tetralith (SLURM).

Reference paper: Jinnai & Fukunaga, *A Graph-Partitioning Based Approach for
Parallel Best-First Search* (HSDIP-17) — defines HDA*, the overhead metrics, and
GRAZHDA*/sparsity.

## 2. Background / findings

- **Parallel search lives on `origin/hda`**, not the current `scorpion` branch
  (sequential-only). Engine: `peager` (`parallel_eager_search::ParallelEagerSearch`),
  pure MPI. The `hda` driver wires `-np N` → `mpirun -np N`
  (`driver/arguments.py:456`, `driver/run_components.py:167-171`). So **workers =
  MPI ranks**: 1 worker = `-np 1`, 16 workers = `-np 16`. No threading
  (tyr's `WxT` reduces here to `1x1` and `16x1`).
- Work-distribution hashes on `hda`
  (`src/search/parallel_hash/distribution_hash.{h,cc}`):
  `zobrist` (ZHDA*), `abstraction` (AHDA*), `aabstraction` (DAHDA*),
  `fstructured` (feature abstract-Zobrist via greedy DTG bisection = GAZHDA*),
  `astructured` (action abstract-Zobrist). **GRAZHDA*/sparsity is absent**
  (confirmed: no `sparsit|grazhda|sparsest|graph.partition` in `hda` source).
- `peager` prints standard per-rank Fast Downward stats
  (`Expanded N state(s)`, `Peak memory: N KB`, `Actual search time`) via
  `statistics.print_detailed_statistics()`. **No communication / off-parent
  counter exists.**
- distributed-tyr's harness: Downward Lab generic `Experiment`, custom
  `SearchParser`, suites `SUITE_HTG` / `SUITE_HTG_TEST`
  (`experiments/suite.py`), limits via `--search-time-limit` + lab wall limit and
  `memory_limit=32_000` (MB); memory score = `compute_log_score` bounds
  `[2 MB, 32 GB]`. Metrics parsed from tyr's own `[Search] ...` / `[Total] ...`
  log lines.
- HTG benchmark present at
  `/home/workbox/Projects/downward-projects/benchmarks/htg-domains/` (full) and
  `.../htg-test/` (subset). ~23 lifted domains (blocksworld-large-simple,
  childsnack, genome-edit-distance, labyrinth, logistics-large-simple,
  organic-synthesis, pipesworld-tankage, rovers-large-simple,
  visitall-multidimensional).
- This machine: **12 cores / ~14 GB RAM** — cannot honestly run 16 ranks @ 32 GB.
  Full run → **Tetralith**; local → validation only.

## 3. Decisions (locked)

- **Branch:** new branch `hda-htg-experiment` off `origin/hda`. Leaves `scorpion`
  untouched.
- **Full-run environment:** Tetralith (SLURM), like tyr's REMOTE mode
  (16 cores/node, 32 GB, 10 min, email `olijo92@liu.se`). Use the same SLURM
  account as the current tyr full-suite scripts — `naiss2025-22-1329` (confirm
  before submitting). Local validation uses reduced limits.
- **Search (match tyr exactly):** tyr's today-dated full-suite scripts
  (`15-06-2026-A/B/C`) use **`--boost-preferred-queue 1000`** — preferred-operator
  queue **boosting enabled** (alternating dual-queue: a standard queue ordered by
  `h=ff()` plus a preferred-operator queue whose weight is bumped by 1000 on every
  new best h-value), `reopen_closed=false`. scorpion `peager` therefore uses
  **preferred operators from FF with an alternating dual-queue boosted by 1000**
  (FD `eager_greedy([ff()], preferred=[ff()], boost=1000)` semantics; exact
  peager open-list/boost syntax verified during validation), `reopen_closed=false`.
  scorpion's only parallel engine is **eager** (`peager`); tyr is **lazy** — the
  single unavoidable asymmetry, documented in the report. tyr's canonical full
  **10-min** script is `15-06-2026-C-gbfs-ff-lazy-10min.py` (mirror its limits).
  Note: `base-tyr` is compiled from C++ (`exe/gbfs_lazy`), whose
  `gbfs_lazy.hpp` defaults `use_preferred_actions=true` and
  `boost_preferred_queue=1000`, so it already runs the boosted dual-queue;
  `distributed-tyr-c`'s today-dated scripts were aligned to `1000` to match it.
  (The Python `gbfs_lazy.py` is only a demo with a no-op preferred branch and is
  not what runs.) The comparison target is the C++ `distributed-tyr-c`.
- **Hash / "best configuration":** implement **GRAZHDA*/sparsity** (paper's best)
  as the headline 16-worker config; keep `zobrist` as the 16-worker baseline.

## 4. Configurations

| nick | workers | hash | role |
|---|---|---|---|
| `gbfs-ff-1` | `-np 1` | n/a | sequential baseline (search-overhead denominator) |
| `gbfs-ff-16-grazhda` | `-np 16` | `grazhda()` | headline "best" (GRAZHDA*/sparsity) |
| `gbfs-ff-16-zobrist` | `-np 16` | `zobrist()` | baseline reference (ZHDA*) |

Peager search string (final syntax verified during validation), with preferred
operators + boost 1000 to match tyr, e.g.:
`--evaluator h=ff() --search "peager(alt([single(h), single(h, pref_only=true)], boost=1000), preferred=[h], reopen_closed=false, f_eval=h, hash=<H>())"`.

## 5. Component A — GRAZHDA*/sparsity hash (new C++)

New class `GraphPartitioningStructuredZobristHash : MapBasedHash` in
`src/search/parallel_hash/distribution_hash.{h,cc}`, plugin name **`grazhda`**,
registered in `distribution_hash.cc` alongside the others.

- Reuse `MapBasedHash` DTG build + the abstract-Zobrist map mechanism used by
  `FeatureBasedStructuredZobristHash`: assign **one shared random 32-bit value
  per DTG partition** (`map[var][value] = r`), so intra-partition transitions
  don't change the state hash (= no off-parent transfer); XOR across variables.
  The hashing path is therefore identical to the existing structured variants.
- Replace the greedy `divideIntoTwo` with `partitionBySparsestCut(var, structures)`:
  - Build the variable's DTG as an undirected weighted graph: nodes = domain
    values; edge weight `w(u,v)` = number of ground actions inducing transition
    `u→v` on this variable (the paper's `/ total ground actions` normalization is
    a positive constant that cancels in the argmax, so raw counts suffice).
  - DTGs have < 10 nodes ⇒ branch-and-bound / exhaustive over 2-way partitions,
    fixing node 0 in S₁ to remove symmetry. Maximize
    **Sparsity = (|S₁|·|S₂|) / Σ_{cut edges} w** (paper Eq. 5, k=2). A cut weight
    of 0 (disconnected component) is treated as maximal sparsity; tie-break toward
    balanced sizes.
  - Edge-weight source: ground-action counts per transition, obtained from the
    task operators / DTG transition labels. If labels do not retain per-transition
    action counts, fall back to unit weights (documented), which still optimizes
    the cut-ratio structure.
- Honor the same abstraction budget / `is_polynomial` options as the existing
  structured hashes for API consistency.

**Unit test:** the logistics package-location DTG from the paper (Fig. 3) must
yield the sparsest cut that cuts exactly 1 edge, whereas GreedyAFG (`fstructured`)
cuts 2.

## 6. Component B — `peager` communication instrumentation (new C++)

In `parallel_eager_search.{h,cc}`: add a counter incremented when a generated
node's hash-owner rank ≠ the current rank (i.e. the node will be transferred),
plus a generated-node counter. At termination, `MPI_Reduce` both to rank 0 and
print a single aggregate line, e.g.:

```
Off-parent transfers: <sent>/<generated> = <rate>
Total expansions: <sum over ranks>
```

This yields the paper's **Communication Overhead** and an aggregate expansion
count that does not depend on summing per-rank log lines. Per-rank
`Expanded`/`Peak memory` lines remain (parser still sums them as a cross-check).

## 7. Component C — experiment harness (Python, Downward Lab)

New directory `experiments/htg-parallel/` on the branch, mirroring tyr's
`full-suite-experiment/` layout and reusing the existing
`experiments/2024-12-parallel/project.py` patterns:

- `project.py` — REMOTE switch: `TetralithEnvironment` (16 cpus/node, 32 GB,
  10 min wall + headroom, account/email above) vs `LocalEnvironment` (reduced).
- `custom_parser.py` — extends the existing hda parser (Section 8).
- `2026-06-15-A-gbfs-ff-htg.py` — the full experiment: configs from Section 4,
  suite `SUITE_HTG`, `-np {1,16}`, 32 GB / 10 min, build/start/parse/fetch +
  report. A `*-mini` / validation variant uses `SUITE_HTG_TEST` + reduced limits
  locally.
- Limits expressed as tyr does: search-time limit passed to the driver
  (`--overall-time-limit 10m`), lab wall limit above it, `memory_limit=32_000` MB.

## 8. Component D — parser & metric mapping (the "measure the same" core)

Extend `CommonParser`/`get_parser()` to emit tyr-comparable attributes:

| tyr metric | scorpion derivation |
|---|---|
| `coverage` | plan found / search exit code (`downward.outcomes`) |
| `oom` | non-clean exit with no usable memory report / killed |
| `oot` | clean exit, no plan within the 10-min search limit |
| `expansions` (total) | aggregate `Total expansions` (Component B); cross-checked by summing per-rank `Expanded N` |
| `additional_expansions` (= Search Overhead) | report-level: `expansions@16 / expansions@1 − 1` per instance/hash |
| `off_parent_transfer_rate` (= Comm. Overhead) | parsed from Component B line, ∈ [0,1] |
| `peak_memory` / `total_peak_memory` | sum per-rank `Peak memory: N KB` |
| `score_peak_memory` (memory score) | `compute_log_score(success, total_peak_memory_bytes, lower=2 MB, upper=32 GB)` |
| `search_time` | per-rank `Actual search time` (max across ranks) / wall-clock |
| `total_time` | driver total time |

**Explicitly out of scope:** overapproximation ratio (excluded per user).
The "additional expansions" ratio is computed in the report by pairing each
16-worker run with its `-np 1` baseline on the same instance.

Cross-planner side-by-side (scorpion vs tyr) is feasible later by merging both
planners' lab `properties` into one report on the shared attribute names; not
required for this spec.

## 9. Validation plan

1. Build the `hda-htg-experiment` branch with MPI; smoke-test `peager` at
   `-np 1` and `-np 16` on a trivial task.
2. GRAZHDA* unit test (logistics DTG, Section 5).
3. Run the `*-mini` experiment on `SUITE_HTG_TEST` locally (`mpirun
   --oversubscribe` as needed) with reduced limits.
4. Confirm: every attribute parses (no `None` / unexplained errors); coverage,
   OOM, OOT classify correctly on at least one instance each; `off_parent_transfer_rate`
   ∈ [0,1] and is 0 at `-np 1`; `additional_expansions` computes across the 1↔16
   pair; memory score in [0,1].
5. Only then launch the full 32 GB / 10 min run on Tetralith.

## 10. Out of scope

- Overapproximation ratio.
- A lazy parallel engine (scorpion has only eager `peager`).
- tyr's threaded `4x4` layout (peager is MPI-only).
- Re-running tyr itself; we consume tyr's existing results for comparison.
