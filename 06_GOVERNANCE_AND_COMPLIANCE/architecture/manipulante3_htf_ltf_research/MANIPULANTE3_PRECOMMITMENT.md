# MANIPULANTE 3.0 — PRE-COMMITMENT PROTOCOL

## Institutional Integrity Pre-Commitment
By initializing this research investigation, the quantitative engineering team pre-commits to the following non-negotiable operational tenets:

1. **Invariance of Selection Parameters:** Evaluation and shortlisting thresholds (e.g., `PF_val_net >= 1.10`, sample sizes `N_val >= 40`) are defined immutably before analyzing out-of-sample data distributions. Modifying filters post-hoc to retroactively force strategy survival is strictly prohibited.
2. **Absolute Data Sealing:** The TEST set (2023–latest available) will never be utilized to rank, tune, or optimize strategy parameters. It serves solely as a final pass/fail custody witness.
3. **Attribution Integrity:** Performance reporting will unconditionally reflect full deduction of $5.00 per lot commissions and simulated stress slippage. Gross yields will not be used to justify strategy viability.
4. **Reproducibility Guarantee:** All pseudo-random sampling steps (e.g., configuration subset reduction) are seeded deterministically with `RANDOM_SEED = 20260513` to maintain absolute forensic reproducibility.
5. **Transparency of Negative Results:** If the controlled pilot sweep fails to isolate any configuration satisfying the programmatic requirements, the task will terminate immediately with a definitive `PILOT_RED` verdict, avoiding infinite search space loops.
