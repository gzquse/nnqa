# Single-Column RevTeX Figures

This directory contains publication-quality figures optimized for single-column layout in RevTeX LaTeX format.

## Figure Specifications

- **Column Width**: ~3.3 inches (84mm) - standard RevTeX single column width
- **Resolution**: 300 DPI (publication quality)
- **Format**: Both PNG and PDF versions provided
- **Font Sizes**: Optimized for RevTeX (9-10pt base, readable and concise)
- **Layout**: Subplots stacked vertically for single-column compatibility

## Generated Figures

1. **fig1_degree_scaling.png/pdf** (963 x 1301 px)
   - RMSE vs polynomial degree (stacked vertically)
   - Correlation vs polynomial degree (stacked vertically)

2. **fig2_recovery_grid.png/pdf** (936 x 1899 px)
   - 3x2 grid showing polynomial recovery for each degree
   - Concise annotations, optimized for single column

3. **fig3_error_distribution.png/pdf** (939 x 1899 px)
   - Error distribution histograms for each degree
   - 3x2 grid layout

4. **fig4_correlation_summary.png/pdf** (997 x 1450 px)
   - Correlation scatter plot (stacked)
   - Performance metrics by degree (stacked)

5. **table_results.tex**
   - LaTeX table compatible with RevTeX format

## Usage in RevTeX

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\columnwidth]{fig1_degree_scaling.pdf}
  \caption{Recovery error and correlation vs polynomial degree.}
  \label{fig:degree_scaling}
\end{figure}
```

## Generation

To regenerate these figures:
```bash
python research/plot_single_column.py --all \
  --input research/results/cloud \
  --output research/figures/paper/single_column \
  --approach native \
  --no-display
```

Change `--approach native` to `--approach direct` to generate figures for the direct (1-qubit) approach.
