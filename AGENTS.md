# Scope & Editing Discipline
- Make minimal, localized diffs. Never refactor unrelated code or rename variables.
- Before editing >1 file or >30 lines, stop and present a plan first.
- Explain any change to a mathematical formula, loss, or update rule before writing code.

# Environment
- Always use the local virtual environment: `.venv/bin/python` and `.venv/bin/pytest`.
- Stack: PyTorch, JAX, NumPy. Do not mix frameworks in the same module unless asked.

# Verification
- Every new or modified loss, gradient, or equilibrium/optimization step must include a numerical check (e.g., finite-difference gradient check, shape assertions, or known toy-game analytical solution).
- Report exact command outputs; never claim a test passed without running it.

# Research Workflow
- Store experiment specs and derivations in `notes/` and runs/tables/figures in `outputs/`.
