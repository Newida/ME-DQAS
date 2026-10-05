# ME-DQAS: Measurement-Efficient Differentiable Quantum Architecture Search

Code and data for the paper

> L. Theißinger, T. Gerlach, C. Bauckhage,
> *Measurement-Efficient Differentiable Quantum Architecture Search for Combinatorial Optimization*,
> IEEE International Conference on Quantum Artificial Intelligence (QAI), 2026.

ME-DQAS reduces the number of circuit evaluations needed for DQAS parameter
gradients when the cost Hamiltonian is diagonal (Ising/QUBO). Gradients are
either certified to be exactly zero by commutation, or estimated with a single
shifted circuit when the circuit suffix is diagonal or Clifford, instead of the
standard two-shift parameter-shift rule.

## Repository layout

```
src/        DQAS and ME-DQAS implementation, simulator, 3-SAT and MaxCut objectives
scripts/    experiment runners, dataset builders and plotting scripts
dataset/    the exact 100 3-SAT and 100 MaxCut instances and all run logs used in the paper
```

- `src/dqas.py`: baseline DQAS
- `src/me_dqas.py`: measurement-efficient gradient rules (zero-gradient certificate, diagonal and Clifford one-shift rules)
- `src/simulator.py`: shot-based simulation on Qiskit Aer
- `src/three_sat.py`, `src/maxcut.py`: problem encodings

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`qiskit` and `qiskit-aer` are only needed to run experiments. Regenerating the
figures from the saved dataset only needs `numpy` and `matplotlib`.

## Reproducing the paper figures from the saved data

Run these from the repository root:

```bash
# Fig. 1: final loss, completed updates and theta-gradient shot ratio
python scripts/plot_dqas_me_paper_bar_summary.py --output dataset/bar_summary_paper

# Fig. 2: breakdown of the theta-shot reduction
python scripts/plot_me_dqas_reduction_breakdown.py \
    --dataset dataset/100_problem_search:3-SAT \
    --dataset dataset/100_maxcut_search:MaxCut \
    --output-base dataset/me_dqas_reduction_breakdown
```

Each script also writes a `.summary.json` with the numbers reported in the paper
(for example a mean total theta-shot reduction of 41.4% on 3-SAT and 39.4% on MaxCut).

## Re-running the experiments

The datasets are built by searching over problem seeds, keeping instances that
are neither trivially solved nor flat, and running paired DQAS and ME-DQAS
comparisons under the same global shot budget (n = 8 qubits, circuit length
L = 8, gate set {Rx, Ry, Rz, CZ}, batch size 128, S_f = S_theta = 50, budget 750,000 shots).

```bash
# 3-SAT (100 instances)
python scripts/build_dqas_me_dataset.py --output-dir dataset/100_problem_search

# MaxCut (100 instances)
python scripts/build_dqas_me_maxcut_dataset.py --output-dir dataset/100_maxcut_search
```

All seeds and settings are stored in each dataset's `manifest.json`, and every
instance directory holds `problem.json` (instance, seeds, exact brute-force
optimum) and `comparison.json` (full paired run logs). Use `--help` on any
script for all options.

Single runs and comparisons can be launched with
`scripts/run_dqas_*_experiment.py`, `scripts/run_me_dqas_*_experiment.py` and
`scripts/compare_dqas_me_dqas_*.py`.

## Citation

```bibtex
@inproceedings{theissinger2026medqas,
  author    = {Thei{\ss}inger, Lukas and Gerlach, Thore and Bauckhage, Christian},
  title     = {Measurement-Efficient Differentiable Quantum Architecture Search for Combinatorial Optimization},
  booktitle = {IEEE International Conference on Quantum Artificial Intelligence (QAI)},
  year      = {2026}
}
```

## Acknowledgment

This work was funded by the Lamarr Institute in Germany.
