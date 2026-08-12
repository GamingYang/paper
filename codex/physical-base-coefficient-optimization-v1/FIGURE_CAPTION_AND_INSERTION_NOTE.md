# Low-order Prior Coefficient Optimization Figure Note

## Generated Files

- Main figure: `figures/fig_physical_base_abcd_optimization_grid.png`
- Main figure PDF: `figures/fig_physical_base_abcd_optimization_grid.pdf`
- Fit-check figure: `figures/fig_physical_base_source_fit.png`
- Fit-check figure PDF: `figures/fig_physical_base_source_fit.pdf`
- Coefficient table: `tables/physical_base_coefficients.csv`
- Source binned data: `tables/source_soh_binned_for_physical_base.csv`

## Data Protocol

The coefficient optimization uses only Batch 1-8 source-domain training samples:

- source samples: 115256
- source cells: 61
- source batches: 1-8
- no Batch 9 or Batch 10 samples are used for fitting the low-order nonlinear prior.

The fitted prior is:

\[
f_{base}(\tilde{p})=
0.6019\tilde{p}^3
-0.7450\tilde{p}^2
-0.0888\tilde{p}
+1.0758 .
\]

Source-domain prior-only RMSE is 0.03657.

## Suggested Caption

**Fig. X. Optimization landscape of the four coefficients in the low-order nonlinear degradation prior.** Each panel varies two coefficients while fixing the other two at their source-domain least-squares optimum. The color denotes the RMSE increase relative to the optimum solution, and the red point marks the fitted coefficient pair. The result shows that the cubic degradation prior is obtained through a source-only coefficient optimization process and serves as a low-complexity global degradation baseline before residual temporal correction.

## Suggested In-text Sentence

为进一步说明低阶非线性退化先验的构建过程，图 X 给出了四个系数 \(a\)、\(b\)、\(c\)、\(d\) 的二维优化切片。该图仅基于源域训练样本计算最小二乘目标函数，不包含目标适应集和最终盲测集信息。可以看出，四个系数共同决定全局退化基线的下降趋势与曲率形态，为后续残差时序建模提供了稳定的低复杂度参考。

## Visual QA

- Font: Times New Roman (`C:\Windows\Fonts\times.ttf`)
- Main PNG resolution: 6152 x 3171
- Fit PNG resolution: 4199 x 2484
- Manual visual check: the in-figure title/explanation has been removed, y-axis coefficient labels are horizontal and close to their own y-axis ticks, no obvious text overlap, no axis-label blocking, no hidden colorbar text, and no blank rendering.
