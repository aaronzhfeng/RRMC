# RRMC

**RRMC (Robust Revision-MI Control)** is the working name for the method described in the Robust-MI Active Inquiry proposal: use **robust self-revision mutual information** as an uncertainty signal, then apply **risk-controlled thresholding** (Clopper–Pearson UCB) to decide **ask vs answer** in interactive tasks like AR-Bench.

## Docs

- **00_proposal (implementation-ready spec)**: `docs/00_proposal.md`
- **01_literature (curated references)**: `docs/01_literature.md`
- **Docs index / reading order**: `docs/README.md`

## Secrets / API keys

- Put secrets in a local `.env` file in the repo root (this file is git-ignored).
- Template: `configs/env.example`