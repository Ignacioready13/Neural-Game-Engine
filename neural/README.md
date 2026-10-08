# /neural — reserved for future work

This directory is an intentional placeholder. It is reserved for the **future,
separate neural-network codebase** (the "dynamics model" described in Phase 1's
long-term vision), which will:

- be written in Python (PyTorch), *not* TypeScript;
- consume the `(sequence of frames, sequence of control signals)` dataset that
  the classical `/engine` produces into `/data`;
- be fully independent of the classical engine — it will never call back into
  it, and the classical engine will be discarded once the model is trained.

Nothing else should be placed in this directory during Phase 1.
