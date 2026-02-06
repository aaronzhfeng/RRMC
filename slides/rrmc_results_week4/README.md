# RRMC Results Presentation

Slides presenting the RRMC stopping rule evaluation results on AR-Bench Detective Cases.

## Structure

The presentation covers:

1. **Background: Stopping Methods** - How each method works
   - Simple baselines: Fixed Turns, Verbalized Confidence, Self-Consistency, Semantic Entropy
   - MI-based methods: MI-Only, Robust MI
   - New baselines: KnowNo, CIP-Lite, UoT-Lite

2. **Initial Method Runs** - Baseline results with default thresholds

3. **Parameter Grid Search** - Systematic threshold tuning
   - Methodology
   - Results per method
   - Visualization of threshold vs. accuracy

4. **Optimized Method Runs** - Results with tuned thresholds

5. **Conclusions** - Key takeaways and future work

## Building

```bash
# Install theme and compile
make all

# Or just compile (if theme already installed)
make slides

# Quick single-pass compile
make quick

# Clean up
make clean
```

## Requirements

- LaTeX with Beamer
- pdflatex
- Metropolis theme (included via mtheme submodule)

## Updating Results

Edit `main.tex` to update:
- Tables with new accuracy/turns numbers
- TikZ plots with new data points
- "Validation Results" section with 20-puzzle run data
