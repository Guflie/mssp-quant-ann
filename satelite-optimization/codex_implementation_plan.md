# ESA Ground Station Scheduling — Phased Implementation Plan for Codex

## Goal
Build the project **incrementally** in Python using a **single Jupyter notebook (`.ipynb`) as the main development artifact** for the first stages, with clear markdown sections, commented code cells, and test checkpoints after each phase.

The implementation must follow the pipeline already defined in the document:

1. Problem scope
2. Discretization layer
3. Classical reduction
4. Conflict-graph construction
5. Validation of the MWIS abstraction
6. Classical MWIS solution
7. QUBO formulation
8. Selection of QPU-suitable subproblems
9. Pure quantum annealing execution

However, **do not implement everything at once**.

---

## Global Development Rules

These rules must be followed in every phase.

### 1. Development must be incremental
- Only implement the current approved phase.
- Do not implement future phases in advance unless explicitly asked.
- After each phase, stop and leave a clear testing checkpoint.

### 2. Use Python in notebook format
- The implementation should be done in **`.ipynb` notebook format**.
- Keep the notebook organized with alternating **Markdown** and **Code** cells.
- Each phase must have:
  - a short markdown explanation,
  - the code for that phase,
  - a small test/demo section,
  - and a final output or validation cell.

### 3. Keep the code clear and heavily commented
- All classes must include docstrings.
- All functions must include docstrings.
- Important blocks inside functions should also have inline comments.
- Prefer readability over cleverness.

### 4. Do not create unnecessary files
- Avoid generating a large project structure at the beginning.
- For the first implementation stages, prefer **one notebook**.
- Only add files if they are clearly necessary and explicitly justified.
- Do not create placeholder modules, helper files, or extra classes unless they are actually used.

### 5. Keep functions minimal and purposeful
- Do not create functions that are not used.
- Do not create abstraction layers too early.
- Every class and function should serve a direct role in the current phase.

### 6. Make every phase testable
- At the end of each phase, provide a small synthetic test case.
- The output should be readable directly in the notebook.
- If relevant, print intermediate objects clearly.

### 7. Preserve future extensibility without overengineering
- The code should be clean enough to extend later.
- But avoid building a framework too early.
- Keep the first versions simple and explicit.

---

## Recommended Initial File Strategy

### For now, create only:
- `ground_station_pipeline.ipynb`

### Optional later files only if needed:
- `data/` folder for sample input files
- `results/` folder for exported outputs

Do **not** create extra `.py` modules initially unless explicitly requested later.

---

# Phase Plan

## Phase 1 — Notebook skeleton and core data model

### Objective
Create the base notebook structure and define the core data representation for candidate contact slots.

### Why this phase comes first
Everything in the pipeline depends on having a clean, understandable data structure for candidate slots.

### Deliverables
Inside the notebook:

#### 1. Notebook title and introduction
Create markdown cells explaining:
- the purpose of the notebook,
- the current phase,
- and what will be implemented now.

#### 2. Imports cell
Keep imports minimal.
Likely imports:
- `dataclasses`
- `typing`
- `math` if needed
- `itertools` if needed
- `pandas` only if clearly useful

Do not import libraries that are not used yet.

#### 3. Core data class for one candidate slot
Create a `@dataclass` called something like `CandidateSlot`.

It should include at least these fields:
- `slot_id: str`
- `satellite_id: str`
- `antenna_id: str`
- `station_id: str`
- `start_time: float`
- `end_time: float`
- `priority: float`
- `urgency: float`

And also computed or stored fields for:
- `duration`
- `weight`

### Requirements for the class
- Add a class docstring.
- Add comments for each field group.
- Provide a method or property to compute duration.
- Provide a method to compute weight from coefficients.
- Keep this simple and explicit.

#### 4. Weight computation
Implement a function or method that computes:

`weight = alpha * priority + beta * duration + gamma * urgency`

Make sure:
- coefficients are arguments,
- the behavior is documented,
- and the result is easy to inspect.

#### 5. Basic pretty-print / inspection support
Make notebook output readable.
Either:
- rely on dataclass default repr if clean enough,
- or provide a small helper method for human-readable display.

Do not overengineer this.

#### 6. Small synthetic example
Create 3 to 5 synthetic candidate slots manually.
Use them to test:
- duration computation,
- weight computation,
- and readability of the class.

### Stop condition for Phase 1
Phase 1 is complete when:
- the notebook exists,
- the `CandidateSlot` class exists,
- example slots can be created,
- duration and weight are computed correctly,
- and the notebook is clean and understandable.

---

## Phase 2 — Step 2 implementation: discretization layer

### Objective
Implement the discretization layer that converts raw visibility intervals into candidate slots.

### Important simplification
At this stage, one candidate slot corresponds to **one full feasible visibility interval**.
Do not split windows into smaller chunks.

### Deliverables
Inside the notebook:

#### 1. Markdown explanation of Step 2
Explain:
- what the discretization layer does,
- what the input format is,
- and what the output format is.

#### 2. Raw input representation
Define a minimal raw visibility input format.
This can be either:
- a second dataclass like `RawVisibilityInterval`, or
- a simple dictionary-based structure.

Prefer a dataclass if it helps clarity.

Suggested fields:
- `satellite_id`
- `antenna_id`
- `station_id`
- `start_time`
- `end_time`
- `priority`
- `urgency`

Optional additional fields only if used:
- antenna availability interval
- blackout interval information
- duration limits

#### 3. Discretization function
Create a function like:
- `build_candidate_slots(...)`

This function should:
- receive raw visibility intervals,
- apply the Phase 2 assumptions,
- create one `CandidateSlot` per feasible interval,
- compute duration,
- compute weight,
- return a list of candidate slots.

#### 4. Feasibility filtering inside discretization
Implement the filters already defined in the document:
- inside 12-hour horizon,
- inside antenna availability,
- no blackout overlap,
- valid duration.

Keep this simple and explicit.

#### 5. Test case
Create a small synthetic set of raw intervals and show:
- which ones are accepted,
- which ones are rejected,
- and why.

### Stop condition for Phase 2
Phase 2 is complete when:
- raw visibility intervals can be converted to candidate slots,
- infeasible intervals are removed,
- weights are assigned,
- and the notebook clearly shows the output candidate set.

---

## Phase 3 — Step 3A and 3B: classical reduction core

### Objective
Implement the first real reduction pipeline:
- Step 3A: constraint propagation / deterministic feasibility filtering
- Step 3B: safe structural reductions

Do **not** implement heuristic reductions yet, except maybe leave a placeholder markdown section.

### Deliverables

#### 1. Markdown explanation of Step 3A and 3B
Explain that this phase implements:
- feasibility-driven reduction,
- duplicate removal,
- strong dominance removal.

#### 2. Step 3A functions
Implement explicit functions for:
- data consistency checks,
- availability consistency,
- blackout consistency,
- duration consistency,
- local transition consistency.

These can be separate functions or one orchestrating function that calls smaller helpers.

Recommended structure:
- small helper functions,
- one main function like `apply_constraint_propagation(...)`.

#### 3. Step 3B functions
Implement:
- duplicate removal,
- strong dominance removal.

For strong dominance, use the concrete local rule already defined in the document.
Start with the simplest safe version.

#### 4. Reduction pipeline function
Implement a function like:
- `run_safe_reduction_pipeline(...)`

This function should:
1. apply Step 3A,
2. apply duplicate removal,
3. apply dominance removal,
4. return the reduced candidate-slot list.

#### 5. Test case
Create a synthetic example where:
- some slots are invalid,
- some are duplicates,
- some are dominated.

Print:
- input size,
- size after Step 3A,
- size after Step 3B,
- final remaining slots.

### Stop condition for Phase 3
Phase 3 is complete when:
- the safe reduction pipeline works on a toy example,
- the reduction effects are visible in the notebook,
- and the code is clean and commented.

---

## Phase 4 — Step 3C: first heuristic reduction

### Objective
Implement only the first approved heuristic reduction:
- **Top-k per satellite**

Do not implement all heuristic options now.

### Deliverables

#### 1. Markdown explanation
Explain that this is the first optional heuristic reduction and that it is benchmark-dependent.

#### 2. Function implementation
Implement a function like:
- `keep_top_k_per_satellite(slots, k)`

Behavior:
- group slots by satellite,
- sort by descending weight,
- keep the top `k` for each satellite,
- return the reduced set.

#### 3. Pipeline integration
Integrate this as an optional stage after Phase 3 reductions.

#### 4. Test case
Build an example with several slots per satellite and show the effect of different `k` values.

### Stop condition for Phase 4
Phase 4 is complete when:
- top-k per satellite works,
- it can be turned on/off clearly,
- and the effect is easy to inspect.

---

## Phase 5 — Step 4: conflict graph construction

### Objective
Implement the weighted conflict graph from the reduced candidate-slot set.

### Deliverables

#### 1. Markdown explanation
Explain the graph meaning:
- node = surviving candidate slot
- edge = incompatibility
- node weight = slot value

#### 2. Graph representation
Use the simplest clean representation.

Preferred option:
- `networkx` if available and useful

If using `networkx`, keep it minimal and justified.
If not, implement a lightweight adjacency representation.

Do not overengineer.

#### 3. Node construction
Create one node per slot.
Store the slot object or a slot reference as node metadata.

#### 4. Edge construction
Implement exactly these conflict rules:
- same-antenna overlap conflict
- same-antenna reconfiguration-time conflict

#### 5. Graph diagnostics
Compute and display:
- number of nodes
- number of edges
- average degree
- density
- connected components

#### 6. Test case
Build a small reduced slot set, construct the graph, and display:
- node list
- edge list
- graph statistics

### Stop condition for Phase 5
Phase 5 is complete when:
- the conflict graph is built correctly from reduced slots,
- edges correspond to the intended rules,
- and diagnostics are printed clearly.

---

## Phase 6 — Step 5 and Step 6: validation + classical MWIS solving

### Objective
Implement:
- the basic validation logic of the MWIS abstraction,
- and the classical MWIS solver.

### Deliverables

#### 1. Classical MWIS formulation
Implement the MWIS problem from the graph.

#### 2. Solver choice
Use a practical classical exact method.
Preferred first choice:
- a binary ILP model using an available Python optimization package if installed

If not available, use a fallback approach that is clearly documented.

#### 3. Reconstruction of the selected schedule
Map selected graph nodes back to candidate slots.

#### 4. Feasibility check
Verify that the selected slot set is feasible under the simplified model.

#### 5. Small benchmark comparison
For a toy instance, compare:
- selected nodes,
- selected slots,
- objective value,
- feasibility.

### Stop condition for Phase 6
Phase 6 is complete when:
- the graph can be solved classically,
- the selected schedule can be reconstructed,
- and the notebook shows the result clearly.

---

## Phase 7 — Step 7: QUBO construction

### Objective
Translate the MWIS graph into QUBO form.

### Deliverables

#### 1. Markdown explanation
Explain the mapping:
- one binary variable per node,
- linear reward from node weight,
- quadratic penalty from conflict edge.

#### 2. QUBO builder function
Implement a function like:
- `build_mwis_qubo(graph, penalty)`

The function should return a clear QUBO representation, for example:
- dictionary of coefficients,
- or matrix form if appropriate.

#### 3. Diagnostics
Report:
- number of variables,
- number of quadratic couplings,
- penalty used,
- coefficient ranges.

#### 4. Test case
Build the QUBO for a small graph and print it clearly.

### Stop condition for Phase 7
Phase 7 is complete when:
- the graph can be translated to a QUBO representation,
- and the output is readable and testable.

---

## Phase 8 — Step 8: QPU-suitable subproblem selection

### Objective
Implement the first subproblem-selection strategy.

### Deliverables

#### 1. First strategy only
Implement only:
- connected-component decomposition

Optional later extension:
- dense local extraction

#### 2. Functions
Implement functions to:
- extract connected components,
- compute subproblem descriptors,
- rank or filter them for quantum suitability.

#### 3. Test case
Use a small graph with multiple components and show the extracted subproblems.

### Stop condition for Phase 8
Phase 8 is complete when:
- connected components are extracted,
- each component has descriptors,
- and candidate subproblems are clearly listed.

---

## Phase 9 — Step 9: pure quantum annealing execution wrapper

### Objective
Prepare the notebook to execute selected QUBO subproblems on annealing hardware.

### Important note
This phase should only start after the earlier classical phases are approved.

### Deliverables

#### 1. Markdown explanation
Explain that this stage handles:
- embedding,
- chain strength,
- repeated sampling,
- decoding,
- feasibility checks,
- classical comparison.

#### 2. Execution wrapper
Implement a minimal wrapper around the annealing API/library you plan to use.

#### 3. Decoding and evaluation
Implement:
- logical solution reconstruction,
- feasibility checks,
- objective evaluation,
- comparison against the classical baseline.

#### 4. Test on one selected subproblem
Only one at first.

### Stop condition for Phase 9
Phase 9 is complete when:
- one selected QUBO subproblem can be run end-to-end,
- and the result can be compared against the classical MWIS solution.

---

# Immediate Instruction for Codex

Start with **Phase 1 only**.

## What Codex should do now
Create a notebook named:
- `ground_station_pipeline.ipynb`

Inside it, implement only:
- the notebook structure,
- the core `CandidateSlot` dataclass,
- duration computation,
- weight computation,
- and a very small synthetic example with a few slots.

## What Codex should NOT do yet
Do not yet implement:
- discretization logic,
- reduction pipeline,
- graph construction,
- MWIS solver,
- QUBO,
- or annealing execution.

After Phase 1 is complete, stop and wait for approval.

---

# Coding Style Requirements for Codex

- Use clear names.
- Prefer explicit code over compact clever code.
- Add docstrings to every class and function.
- Add inline comments to nontrivial logic.
- Keep notebook outputs readable.
- Do not create unused helper functions.
- Do not create unnecessary files.

---

# Final Note for Codex

This project must be developed as a **phase-approved notebook pipeline**.
At the end of each phase:
1. make sure the notebook runs cleanly,
2. demonstrate the outputs clearly,
3. and stop before moving to the next phase.
