# QPU annealing run — CSP `s400_g50`, 100 reads (2026-11-06)

First full run of the CSP pipeline's terminal QUBOs on a D-Wave QPU. All 218 terminals of the
`s400_g50` decomposition were annealed and the schedule was reconstructed.

## Setup

| | |
|---|---|
| Instance | `data/test/slots_s400_g50.csv`: 400 satellites, 50 stations, 24,361 slots (1,192 after dominance) |
| Decomposition | run `csp_test1/s400_g50`: terminals ≤ 200 qubits / 40,000 couplers, separators ≤ 5 satellites, 2,000 separator states |
| Terminals | 218 domain-wall QUBOs (209 from the separator tree of the 913-qubit component), 13 incomplete leaves |
| QPU | Advantage_system6, 20 µs anneal, 100 reads per terminal, one QPU job per terminal |
| Embedding | minorminer, fixed per terminal; chain strength = uniform torque compensation, prefactor 1.414 |
| QPU access time | **6.23 s** in total (budget 7.5 s), about 28 ms per terminal, of which ~16 ms is programming |

## Results per terminal

The exact optimum of each terminal was computed classically (MILP, HiGHS) and compared with every QPU read.

| Terminal size | Terminals | With a feasible read | Exact optimum found | Feasible reads | Best read ÷ optimum (median / worst) | Chain breaks (mean) |
|---|---|---|---|---|---|---|
| ≤ 30 qubits | 23 | 23 | **20** | 99 % | 1.000 / 0.996 | 0.12 % |
| 31–100 | 66 | 66 | 0 | 88 % | 0.990 / 0.985 | 0.28 % |
| 101–150 | 89 | 89 | 0 | 10 % | 0.965 / 0.900 | 0.07 % |
| 151–250 | 40 | 40 | 0 | 14 % | 0.969 / 0.905 | 0.06 % |

- **Every terminal has at least one feasible read.** The earlier 5-read run left 71 of 218
  terminals without one; that matched the per-read feasibility rate, and 100 reads remove it.
- **Chains hold.** Chain breaks average about 0.1 %, with chains of up to 9 qubits and up to 347
  physical qubits per terminal. The `ChainStrengthWarning` reported by Ocean is a false alarm for
  QUBO input: the chains are built in spin space but the warning compares them with the binary
  QUBO couplings, which are 4 times larger.
- **The exact optimum is reached only on small terminals.** Above 30 qubits the QPU never reaches
  it in 100 reads. Over all terminals, the best reads add up to 97.4 % of the summed optima.

## Terminals above 100 qubits (129 terminals, 12,900 reads)

- **89 % of the reads are infeasible.** In 99.7 % of them a satellite holds two or more slots,
  i.e. the domain-wall encoding of "one slot per satellite" is broken; conflicts between different
  satellites cause only 0.3 %. Infeasible reads are close to valid: median 2 violated pairs,
  90th percentile 5.
- **The 11 % of feasible reads are well below the optimum:** median 92 % of it (range 70 %–99.2 %);
  only 0.1 % come within 1 % of the optimum.

## Broken domain walls

Each satellite's choice is encoded as a domain wall: a string of bits, one per slot, that must
contain a single wall. A domain wall is broken when its bit string is not a valid single wall; in
94 % of the broken ones this leaves the satellite with two or more slots. Counted over every
satellite of every read (380,800 satellite encodings), **24,771 domain walls broke (6.5 %)**. The
embedding chains, by contrast, broke in only 0.14 % of cases on average (7.1 % in the worst
terminal).

| Terminal size | Satellite encodings | Broken domain walls |
|---|---|---|
| ≤ 30 qubits | 7,400 | 15 (0.2 %) |
| 31–100 | 136,400 | 914 (0.7 %) |
| 101–150 | 157,200 | 16,810 (10.7 %) |
| 151–250 | 79,800 | 7,032 (8.8 %) |

| Slots per satellite (domain-wall length) | Satellite encodings | Broken domain walls |
|---|---|---|
| ≤ 5 | 270,100 | 0.2 % |
| 6–15 | 74,100 | 7.2 % |
| 16–30 | 11,400 | **42.9 %** |
| 31–60 | 25,200 | **55.2 %** |

**The failure is driven mainly by long domain walls, not by terminal size as such.** A satellite
with more than 15 slots breaks its encoding in roughly half of the reads. Large terminals contain
satellites with up to 54 slots, and with tens of such satellites in one terminal at least one of
them breaks in nearly every read, which is why those terminals are rarely feasible.

## Reconstructed schedule

| | Objective | Gap bound |
|---|---|---|
| Final hybrid schedule | **31,009.751**, certified globally optimal | 0 % |
| Schedule from the annealed terminals alone | 30,806.18 | 0.66 % |
| Classical pipeline (exact terminals + greedy fallback) | 31,007.16 | 0.16 % |

The certificate comes from the classical side: for the large component (`component_001`) the
bounded MILP (1.5 s) found a solution 155.5 (1.27 %) better than the annealed one and replaced
it. **Only 9 of the 218 annealed terminals are used in the final schedule.**

## Interpretation

The QPU handles terminals up to about 100 qubits well: they are almost always feasible and
within about 1 % of the optimum. Above 100 qubits it mostly fails to keep the one-slot-per-satellite
encoding, and its feasible answers are several percent below the optimum. The breakdown above
shows that what matters most is the length of each satellite's domain wall: short ones (≤ 5 slots)
almost never break, long ones (> 15 slots) break about half of the time.

A likely contributing cause is the energy scale: the reward for scheduling a
slot is at most 5 (`assignment_gain`), while the encoding and conflict penalties reach 26. After
the QPU rescales the couplings, the differences between good schedules are a small fraction of the
hardware range, where analog control noise matters most. Switching to one-hot is unlikely to help:
on these terminals it needs about 3 times the couplers (median 1,548 against 526), with cliques of
up to 54 variables per satellite.

## Next steps

1. **Shorter domain walls:** limit the number of slots per satellite that reaches the QPU, e.g. by
   branching on (or pre-fixing) satellites with long domains in the decomposition, or by thinning
   their slots before encoding. This targets the failure directly.
2. **Smaller terminals:** re-run the decomposition with a 100-qubit terminal limit, the range in
   which the QPU performed well. More terminals means more QPU jobs (~16 ms of programming each).
3. **Repair after sampling:** keep the best slot of each over-assigned satellite and drop any
   remaining conflicts. Infeasible reads are typically 2 violations from valid; this can be tested
   on the saved reads without new QPU time.
4. **Penalty tuning together with repair:** penalties closer to their minimum valid value leave
   more of the QPU's range to the objective.
5. **Longer anneal** (e.g. 100 µs): nearly free in QPU time, since programming dominates.
6. For the paper: a one-hot against domain-wall comparison on the same terminals.

## Files

All in `results/runs/csp/qubo_annealing/s400_g50_qpu/`:

- `annealing_results.csv`: one row per terminal (chosen read, feasibility shares, full QPU timing, chain breaks, warnings)
- `annealing_reads.csv`: every distinct read of every terminal
- `annealing_samplesets.json.gz`: the raw sample sets returned by the QPU
- `qpu_solver_properties.json`, `qpu_embeddings.json`: solver snapshot and the embeddings used
- `reconstructed_summary.csv`, `milp_component_results.csv`: reconstruction and MILP results
- `analysis/qpu_vs_exact.py`, `analysis/qpu_vs_exact_per_terminal.csv`: exact optimum per terminal and the QPU comparison
- `analysis/domain_wall_breaks.py`, `analysis/domain_wall_breaks_per_satellite.csv`: broken domain walls per satellite, by terminal size and domain length

The 5-read run of 2026-11-05 was written to the same folder and overwritten by this one; only its
summary figures (71 infeasible terminals, 3.62 s of QPU time) remain.
