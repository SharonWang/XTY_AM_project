# Stage 2 Donor Plot Design

## Goal

Add manuscript-facing donor-level Stage 2 figures without changing the supplied
visual encodings. Both figures consume outputs from
`summarize_stage2_by_donor_and_tissue`; donors remain the biological replicates.

## Public functions

- `plot_stage2_primary`: donor points, median diamonds, donor IQR, and tissue-test
  annotations for prespecified primary methods/scales. The balanced-tail panel is
  explicitly labeled secondary.
- `plot_stage2_scale_sensitivity`: median donor effect and donor IQR across k or
  radius values, stratified by tissue.

## Safeguards

The public functions validate required columns and reject duplicate rows for the
same donor, method, scale, and tissue. Tissue-test rows used for annotation must
also be unique for method, scale, and tissue. This prevents accidental weighting
of a donor twice or arbitrary selection of one test row. Significance notation is
standardized to `*`, `**`, `***`, and `****`; the malformed seven-star label in
the pasted helper is not retained.

The scale-sensitivity figure is descriptive. Its line and IQR summarize donor
effects and do not imply that cells or cores are independent replicates.

## Documentation and validation

Keep one Python source file, export both functions in `__all__`, document inputs
and outputs in NumPy style, update README and proposal version history, and cover
return objects, filtering, duplicates, missing columns, method selection,
significance labels, and file saving with small synthetic tables.
