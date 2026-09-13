# Human Lung Xenium AM-AT2 Spatial Analysis Proposal

**Project:** Human lung alveolar macrophage heterogeneity
**Study:** Xu et al., *Cellular hallmarks and aging clock of the human lung parenchyma*, Nature Communications (2026), PMID 42457688
**Repository root:** `D:\Xiaonan\CODEX_projects\Xiaotong_AM\XTY_AM_project`
**Data root:** `D:\Xiaonan\CODEX_projects\Xiaotong_AM\Spatial\Spatial`
**Version:** 0.6
**Status:** Notebook 00 implemented and locally validated; awaiting scientific review of annotation evidence
**Created:** 2026-09-12
**Last updated:** 2026-09-13

## Document control

This is the living source of truth for the Xenium AM-AT2 analysis. Every material change to the question, cohort, inputs, state definition, statistics, spatial method, communication method, figure plan, or interpretation must update:

1. the version and last-updated date;
2. the affected section;
3. the change log, including rationale and expected effect; and
4. affected configuration or notebooks.

Outputs are not current until these items agree. If the proposal and analysis configuration conflict, record the resolution in the change log. Do not overwrite source data. Keep versioned analysis documents, code, configuration, logs, and normal-sized derived outputs under the D: repository root; keep the supplied Xenium source data at the declared D: data root. Create or modify no project files on C:.

Every material project change must be committed and pushed to the configured Git remote after verification. Commit messages must describe the scientific or technical change. Large source datasets and oversized derived artifacts must not be added to Git; track manifests, hashes, notebooks, summaries, and appropriately sized source-data tables instead.

## Computing locations and execution modes

| Setting | Analysis/repository path | Data path |
|---|---|---|
| Local development | `D:\Xiaonan\CODEX_projects\Xiaotong_AM\XTY_AM_project` | `D:\Xiaonan\CODEX_projects\Xiaotong_AM\Spatial\Spatial` |
| HPC full analysis | `/dssg/home/acct-svetoslav_chakarov/svetoslav_chakarov/Data/External_Data/Xu_NC2026_human/analysis_v1` | `/dssg/home/acct-svetoslav_chakarov/svetoslav_chakarov/Data/External_Data/Xu_NC2026_human/data/Spatial` |

Every notebook will have one visible parameter cell with `RUN_MODE = "local_test"` or `RUN_MODE = "hpc_full"`. Environment variables may override the default paths. The analysis logic must be identical between modes.

`local_test` will use complete cells from a deterministic small set of whole tissue cores. It will not subsample cells inside a selected core because doing so would corrupt nearest-neighbor distances and neighborhood composition. The local outputs validate code, data flow, plots, and statistical calculations only; they are not biological results. `hpc_full` will use all eligible cores and is the only mode used for manuscript inference.

## Executive summary

The published Xenium label `celltype_final == "Macrophages"` is a broad **macrophage candidate pool**, not an automatic AM definition. The analysis will first distinguish evidence for alveolar macrophages from interstitial macrophage or monocyte-like contamination, then test whether the validated AM population contains states corresponding to the mouse-defined **MHCII-low/ResAM1** and **MHCII-high/ResAM2** AM populations. Mouse `Cycling AM` is outside the research question and will be excluded.

Spatial inference will be within tissue cores, with donor as the biological replicate and core nested within donor. Alveolar cores are primary; all regions form a sensitivity analysis. Nearest-cell distances, multi-radius neighborhoods, within-core permutations, and donor-aware models will evaluate AM-AT2 association. Ligand-receptor results will require measured genes, expression, spatial support, and donor reproducibility and will be interpreted as putative communication, not proof of signaling.

This document is a proposal only. It makes no biological result claim yet.

## Questions and hypotheses

Primary questions:

1. Can macrophage-labelled cells be separated into two stable AM-like states across donors and cores?
2. Do states differ in distance to AT2 and in nearest/local cell-type composition?
3. Are state-specific AM-AT2 ligand-receptor programs supported by local co-occurrence?

Secondary questions address continuous age, sex, tissue region, cross-species agreement with the manuscript's mouse AM states, and robustness to classification and spatial choices.

The authoritative mouse signature source is `BD_DEgenes.csv`. For each of `ResAM1` and `ResAM2`, candidate genes must have adjusted P < 0.05, positive log fold change for that group, a documented mouse-human ortholog, and presence in the Xenium panel. The previous log-fold-change > 0.5 cutoff will not be imposed. `BD_AM_MHCII_SigDEgenes.csv` is retained to reproduce and audit the earlier high-confidence selection, not to limit the Xenium signature.

Provisional same-symbol matching identifies five MHCII-low candidates (`CTSL`, `MPEG1`, `CD2`, `CD68`, `CD44`) and nine MHCII-high candidates (`CD74`, `TNF`, `CXCL2`, `IL1B`, `CLEC4E`, `CD14`, `FN1`, `PIM1`, `MCEMP1`). These are not final until a versioned ortholog table resolves mouse `H2-*` and other non-identical symbols.

Candidate communication will be limited to panel-observable pairs, for example `SPP1-CD44`, `MIF-CD74/CD44/CXCR4`, `APP-CD74`, and directionally appropriate WNT-receptor pairs.

## Dataset interpretation

The master object has 332,063 cells, 389 genes, 22 donors, and 70 cores. It contains donor/core metadata, QC, final cell labels, centroids, embeddings, and spatial/connectivity graphs. There are 33,375 cells labelled `Macrophages` and 21,495 labelled `AT2`. The 33,375 macrophage-labelled cells are candidates requiring AM-versus-IM validation; they must not be reported as 33,375 AMs. Every core contains both published labels; 59 cores have at least 50 of each.

Regions comprise 40 alveolar, 22 vascular, 7 bronchial, and 1 core labelled literally `None`. Donors contribute one to four cores, so cells are not independent replicates. Centroids support cell-proximity analysis, but missing morphology images, polygons, and transcript coordinates preclude segmentation review, boundary-contact measurement, and subcellular localization.

## Required-file manifest

### Essential and operational inputs

| File | Requirement | Purpose |
|---|---|---|
| `Spatial\Spatial\xenium.h5ad` | **Required** | Authoritative expression, metadata, cell labels, donor/core structure, and spatial centroids. This is the only existing file strictly indispensable for the requested analysis. |
| `BD_DEgenes.csv` | **Required** | Complete mouse DE results for `ResAM1`, `ResAM2`, and `Cycling AM`. Only `ResAM1` and `ResAM2` rows enter the analysis. |
| `BD_AM_MHCII_SigDEgenes.csv` | Audit/reference | Reproduces the previous high-confidence cutoff list; used to verify selection logic but not to limit panel-overlapping genes. |
| `Spatial\Spatial\myeloid_xenium.h5ad` | Recommended | Efficient AM working subset. Validate cell IDs and values against the master; regenerate from the master if inconsistent. |
| `Spatial\Spatial\epithelial_xenium.h5ad` | Optional | Convenient AT2 validation subset. AT2 membership and coordinates remain authoritative in the master. |

### Conditional published-niche inputs

| File | Requirement | Purpose |
|---|---|---|
| `Spatial\Spatial\soapy_obs.csv` | Conditional | Published per-cell niche assignments for Figure 5h/i context. |
| `Spatial\Spatial\soapy_uns.pkl` | Conditional | Published SOAPy metadata. This roughly 647 MB executable pickle should not be deserialized casually; prefer recomputation or trusted isolated conversion. |
| `Spatial\Spatial\niche_cellprop_by_age.csv` | Conditional/reference | Published age-associated niche proportions, not input to the new AM-state model. |

These files are unnecessary if neighborhoods are recomputed transparently from the master coordinates and labels. That is the default; published niches are contextual validation.

### New references needed for full interpretation

| Resource | Requirement | Provenance rule |
|---|---|---|
| Versioned human ligand-receptor resource | Required for formal communication analysis | Store under `reference\` with source URL, version/date, license, download date, and filtering record. |
| Mouse MHCII-low/ResAM1 and MHCII-high/ResAM2 signatures | Required for cross-species comparison | Derive from `BD_DEgenes.csv`, store the processed signatures and provenance under `reference\`, map orthologs, and intersect with the 389-gene panel. |
| Xu et al. Figure 5 code | Already available | `reference\sc_Aging_clock\Figure 5`; use for conventions and panel mapping, not as an unreviewed dependency. |

### Manuscript for final integration

The authoritative manuscript is:

`D:\Xiaonan\work\项目\Svet\Xiaotong_Yu\XTY_manuscript\20260902_Xiaotong_figure for NI AM heterogeneity_V24\20260902_Xiaotong_figure for NI AM heterogeneity_V24\20260829_Xiaotong_YuX_et_al_V24.docx`

It is a read-only scientific target during analysis development. Edit it only after analysis/figure review and explicit authorization.

### Reproduction/reference files not required for the new analysis

- Figure 5a demographics: `age_map.json`, `sex_map.json`, `smoking_ethnicity_dist.csv`, `smoking_hist.csv`, `ethnicity_dist.csv`.
- Figure 5d compartment UMAPs: `stromal_xenium.h5ad`, `lymphoid_xenium.h5ad` plus the epithelial and myeloid subsets.
- Published fields: `celltype_final_D21.c3.h5ad`, `celltype_final_D9.c1.h5ad`, `celltype_final_D4.c2.h5ad`, `celltype_final_D1.c2.h5ad`, `celltype_final_D20.c3.h5ad`.
- Cell proportions: `cellprop.rds`, `cellprop_corrected.rds`.
- WNT panel: `wnt.pkl`.
- CXCL proximity: `cxcl9_myeloid_D21.c2.h5ad`, `cxcl9_myeloid_D9.c1.h5ad`, `cxcl9_neigh_prox.pkl`, `cxcl9_merged_t_neigh_prox.pkl`, `cxcl10_myeloid_D21.c2.h5ad`, `cxcl10_myeloid_D9.c1.h5ad`, `cxcl10_neigh_prox.pkl`, `cxcl10_merged_t_neigh_prox.pkl`.
- Published niches: `soapy_obs.csv`, `soapy_uns.pkl`, `niche_cellprop_by_age.csv`.
- Immune communication: `icam1_itgal.pkl`, `cd40lg_cd40_anxa1_fpr3.pkl`, `icam1_dc_itgal_cd8t_niche_D21.c2.h5ad`, `niche_boxplots.pkl`, `icam1_dc_itgal_cd4t_cd8t_D10.c2.h5ad`, `icam1_dc_itgal_cd4t_cd8t_D20.c1.h5ad`, `icam1_dc_itgal_cd4t_neigh_prox.pkl`, `icam1_dc_itgal_cd8t_neigh_prox.pkl`, `anxa1_cd4t_fpr3_dc_D10.c1.h5ad`, `anxa1_cd4t_fpr3_dc_D20.c3.h5ad`, `anxa1_cd4t_fpr3_dc_neigh_prox.pkl`, `cd40lg_cd4t_cd40_dc_D10.c2.h5ad`, `cd40lg_cd4t_cd40_dc_D20.c3.h5ad`, `cd40lg_cd4t_cd40_dc_neigh_prox.pkl`.
- QC: `qc_compare.csv`.
- Unrelated Figure 2f immunofluorescence: `mIF_KRT8_prop.csv`.

Files under `Spatial\__MACOSX` are 176-byte macOS resource-fork sidecars and must be excluded.

## Published file-to-figure map

| Paper panel | Files |
|---|---|
| Figure 5a | `age_map.json`, `sex_map.json`, `smoking_ethnicity_dist.csv` |
| Figure 5b | `celltype_final_D21.c3.h5ad`, `celltype_final_D9.c1.h5ad`, `celltype_final_D4.c2.h5ad` |
| Figure 5c | `xenium.h5ad` |
| Figure 5d | `epithelial_xenium.h5ad`, `stromal_xenium.h5ad`, `myeloid_xenium.h5ad`, `lymphoid_xenium.h5ad` |
| Figure 5e | `cellprop.rds` |
| Figure 5f | `wnt.pkl` |
| Figure 5g | `cxcl9_myeloid_D21.c2.h5ad`, `cxcl9_neigh_prox.pkl` |
| Figure 5h | `xenium.h5ad`, `soapy_obs.csv`, `soapy_uns.pkl` |
| Figure 5i | `niche_cellprop_by_age.csv` |
| Figure 5j | `icam1_itgal.pkl`, `cd40lg_cd40_anxa1_fpr3.pkl` |
| Figure 5k | `icam1_dc_itgal_cd8t_niche_D21.c2.h5ad`, `niche_boxplots.pkl` |
| Extended/Supplementary 6a | `xenium.h5ad` |
| Extended/Supplementary 6b | `qc_compare.csv` |
| Extended/Supplementary 6c | `celltype_final_D1.c2.h5ad`, `celltype_final_D20.c3.h5ad` |
| Extended/Supplementary 6d | `cellprop.rds`, `cellprop_corrected.rds` |
| Extended/Supplementary 6e | `cxcl9_myeloid_D9.c1.h5ad`, `cxcl9_merged_t_neigh_prox.pkl` |
| Extended/Supplementary 6f | both CXCL10 field H5ADs and both CXCL10 proximity PKLs |
| Extended/Supplementary 6g | both ICAM1 field H5ADs and the CD4T/CD8T ICAM1 proximity PKLs |
| Extended/Supplementary 6h | both ANXA1 field H5ADs and `anxa1_cd4t_fpr3_dc_neigh_prox.pkl` |
| Extended/Supplementary 6i | both CD40LG field H5ADs and `cd40lg_cd4t_cd40_dc_neigh_prox.pkl` |
| Figure 2f, unrelated | `mIF_KRT8_prop.csv` |

The released Figure 5 script does not use `smoking_hist.csv` or `ethnicity_dist.csv`; they appear to be legacy/intermediate files.

## Data quality and panel limitations

Confirmed strengths:

- unique cell IDs and no missing donor, core, age, sex, coordinate, QC, or final cell-type values;
- QC ranges match the paper's 30-1000 transcripts and 20-203 detected genes per cell;
- all cores contain macrophages and AT2;
- raw integer counts and normalized/log-transformed expression are available.

Important limitations:

- `Macrophages` is broad and must not be assumed to equal AM without marker and location checks.
- The panel lacks `FABP4`, `PPARG`, `INHBA`, `C1QA/B/C`, `VCAN`, `LPL`, `MRC1`, `GPNMB`, `NUPR1`, `CXCL11`, `SFTPC`, `SFTPA1/2`, and `ABCA3`.
- AT2 identity can use the provided label but cannot be reconstructed from usual surfactant markers.
- Full GM-CSF-PPARgamma and TGFbeta mechanisms cannot be directly tested.
- Age groups are `<40` and `>=50`, with no age 40-49 donors; continuous age is primary.
- Missing images, polygons, and transcript coordinates constrain spatial work to centroids.
- Pickles are executable serialization; prefer safe recomputation.

## Cohorts and analysis units

Primary cohort:

- validated alveolar macrophages selected from the published `Macrophages` candidate pool using the pre-MHCII identity-validation step;
- alveolar (`A`) cores;
- donors/cores passing existing QC and the stated minimum-cell rule.

Identity-validation evidence:

- alveolar macrophage evidence: `MARCO` and `APOE` (panel-measured), with canonical `FABP4`, `PPARG`, `INHBA`, `SIGLEC1`, `ITGAX`, `CD36`, and `ABCG1` unavailable;
- interstitial macrophage evidence: `LYVE1`, `CD163`, `FCGR3A`, and `MS4A4A`, with `FOLR2`, `MRC1`, and `C1QA/B/C` unavailable;
- monocyte/inflammatory evidence: `FCN1`, `S100A12`, `IL1B`, and `CLEC4E`;
- pan-macrophage confirmation: `CD68`, `AIF1`, `MPEG1`, and `TYROBP`.

Sensitivity cohorts:

1. all regions with region adjustment or stratification;
2. stricter macrophage and AT2 cell-count thresholds;
3. conservative AM-like cells with stronger `MARCO/APOE` evidence than IM/monocyte evidence;
4. exclusion of the core labelled literally `None`.

**AM identity decision gate:** notebook 00 will show UMAP, dotplot, feature-expression, and whole-core spatial evidence for AM, IM, and monocyte programs. A final AM gate will be approved only after review. It must avoid a threshold that selectively removes MHCII-high AMs, and downstream results will include sensitivity to conservative exclusion of IM/monocyte-like cells.

Analysis units:

- cell: expression scoring and geometry only;
- core: spatial permutation and summary unit;
- donor: biological replicate and inference unit.

Cell-level P values that ignore donor/core dependence will not be primary evidence.

## Analysis workflow

### Phase 1 - Input audit

1. Record paths, sizes, timestamps, and SHA-256 hashes.
2. Confirm dimensions, gene order, observation IDs, raw/normalized matrices, coordinate units, donor/core labels, and subset consistency.
3. Create a machine-readable required-file manifest and audit report.
4. Freeze software versions and random seeds.

### Phase 2 - Macrophage and AT2 cohorts

1. Subset published macrophage-labelled candidates, monocytes, AT1, and AT2 from the master without relabelling them.
2. Tabulate counts by donor, core, region, age, and sex.
3. Flag sparse cores. The default primary spatial-summary threshold is at least 50 macrophages and 50 AT2; retain smaller cores only where statistically valid and report threshold sensitivity.
4. Compare AM, IM, monocyte/inflammatory, and pan-macrophage marker evidence on UMAP, dotplots, feature plots, and whole-core spatial maps.
5. Define the AM analysis population only after visual and quantitative review. Preserve the original label and store the derived AM inclusion flag, score components, rule version, and confidence separately.
6. Check expression/QC for contamination, doublets, and region-specific artifacts.

### Phase 3 - Two-state AM definition

1. Use a documented normalization suitable for the targeted panel; retain raw counts for count-aware models.
2. Reproduce the supplied mouse tables before using them:
   - `ResAM1` and `ResAM2` high-confidence genes exactly equal adjusted P < 0.05 and log fold change > 0.5;
   - `Cycling AM` is excluded from the scientific analysis;
   - record that the supplied statistics appear to be one-versus-rest, not a direct `ResAM2` versus `ResAM1` contrast.
3. Map all mouse `ResAM1` and `ResAM2` genes to human orthologs with a saved, versioned mapping. Document one-to-many, many-to-one, and unmapped cases; do not rely on capitalization alone.
4. Intersect mapped genes with the Xenium panel. Define the expanded group-specific sets using adjusted P < 0.05 and positive log fold change, without the former 0.5 fold-change cutoff. Require the opposite group to have a lower effect estimate; report its direction and adjusted P for every retained gene.
5. Calculate separate unweighted mean scores for MHCII-high and MHCII-low using gene-wise standardized log-normalized expression, then calculate the contrast `MHCII-high score - MHCII-low score`. Use means so unequal signature sizes do not mechanically favor one state.
6. Preserve between-donor biological differences in the primary pooled score. A donor-centered score is allowed only as a sensitivity analysis for within-donor spatial comparisons because donor-centering would invalidate age-related score comparisons and could distort state proportions.
7. Use mouse log-fold-change weighting only as a sensitivity analysis; prevent a single large-effect gene from dominating by pre-specified weight clipping or rank scaling.
8. Compare a continuous-score model with pooled one- and two-component mixture models fitted with donor-balanced weights. Do not use a median split or fit independent donor-specific thresholds that force similar state proportions.
9. Assess component separation, posterior confidence, marker direction, donor/core representation, bootstrap stability, and leave-one-donor-out transfer. Retain an `uncertain` label for low posterior confidence.
10. Validate state differences with donor/core pseudobulk summaries and raw-count-aware models; cell-level P values are descriptive only.

**Decision gate:** hard MHCII-high and MHCII-low labels are primary only if a two-component model improves fit and transfers across donors without being driven by QC or region. Otherwise the continuous MHCII contrast is primary and hard labels are descriptive.

**Mouse-comparison limitation:** because the supplied DE is one-versus-rest and included Cycling AM in the reference pool, it is not identical to direct `ResAM2` versus `ResAM1` DE. Genes must therefore show coherent relative direction across both group rows. If the mouse expression object becomes available, direct pairwise donor-aware DE supersedes this approximation.

### Phase 4 - Spatial localization and nearest cells

Distances must be calculated within the same core; different cores can never be neighbors.

1. Plot representative and systematically selected cores with AM states, AT2, and major cell types.
2. For every AM calculate:
   - nearest-AT2 distance;
   - nearest non-AM cell identity and distance;
   - cell-type counts/proportions within 20, 30, 50, and 100 um;
   - AT2 enrichment relative to local cell density.
3. Assess tissue-boundary and density effects with geometric sensitivity checks and matched spatial nulls.
4. Permute AM-state labels within each core while preserving locations and state counts; use at least 1,000 permutations for final inference.
5. Compare states with donor-aware models and donor/core summaries.

Primary endpoints are log nearest-AT2 distance, probability the nearest non-AM cell is AT2, and AT2 enrichment within 30 and 50 um. Other radii and cell types are secondary, with false-discovery-rate control.

### Phase 5 - Neighborhood composition

1. Build radius-based neighborhoods and k-nearest-neighbor sensitivity analyses.
2. Compare cell-type composition around states with donor-aware tests.
3. Report counts, proportions, local expected proportions, and observed/expected enrichment.
4. Require headline neighbors to be directionally consistent across informative donors.
5. Compare against published SOAPy niches only after completing the transparent radius-based analysis.

### Phase 6 - AM-AT2 communication potential

1. Import a versioned human ligand-receptor resource and intersect both genes with the panel.
2. Analyze AM ligand to AT2 receptor and AT2 ligand to AM receptor separately.
3. Require each candidate to have:
   - measured ligand and receptor;
   - expression above a pre-specified prevalence threshold in relevant strata;
   - state-specific or directionally enriched expression;
   - AM-AT2 proximity/neighborhood support;
   - reproducibility across donors.
4. Rank candidates using expression prevalence, effect size, spatial enrichment, and donor consistency.
5. Evaluate `SPP1-CD44`, `MIF-CD74/CD44/CXCR4`, `APP-CD74`, and measurable WNT pairs while confirming direction in the selected resource.

Report these as **putative spatially supported ligand-receptor interactions**. Causal signaling needs independent validation.

### Phase 7 - Age, sex, and region

1. Use continuous age as primary.
2. Model core-level state proportion with binomial/beta-binomial or mixed-effects methods.
3. Where sample size supports it, model spatial endpoints with state, age, state-by-age, sex, region, and relevant QC/density covariates.
4. Treat `<40` versus `>=50` as descriptive/sensitivity comparisons.
5. Simplify models if 22 donors do not support all covariates.

### Phase 8 - Cross-species interpretation

1. Derive the mouse MHCII-low/ResAM1 and MHCII-high/ResAM2 signatures from the supplied DE table.
2. Map mouse-human orthologs and intersect with the Xenium panel.
3. Score human macrophages and report signature coverage.
4. Separate conserved evidence from untestable mechanisms; an unmeasured marker is not biologically absent.

## Statistical principles

- Donor is the replicate; core is nested within donor.
- Effect sizes and uncertainty intervals are primary; P values are supporting evidence.
- Spatial permutations occur within cores.
- Benjamini-Hochberg correction applies within defined test families.
- Donor bootstrap and leave-one-donor-out analyses assess robustness.
- Targeted-panel expression is summarized with detection prevalence and nonzero expression.
- Models, transformations, thresholds, and contrasts are fixed before final testing.
- Exploratory and confirmatory results are clearly separated.

## Technical acceptance criteria

1. Every analyzed cell maps uniquely to the master and one donor/core.
2. Coordinates never mix across cores.
3. State assignments are not dominated by one donor, core, QC metric, or region.
4. Headline markers have consistent direction in most informative donors.
5. Primary spatial effects remain directionally consistent under leave-one-donor-out analysis and at least two reasonable count thresholds.
6. Communication candidates have measured genes, documented provenance, expression support, spatial support, and donor replication.
7. Notebooks regenerate all outputs from declared inputs without manual editing.
8. Every final figure has source data and an audit trail.

## Notebook architecture

All analytical work will be performed in clearly commented Jupyter notebooks. Python is primary because the source is AnnData/H5AD and the spatial operations use SciPy/Scanpy-compatible structures. A dedicated R notebook may be used where `glmmTMB`, `lme4`, or another validated mixed-model implementation is preferable. Standalone analysis scripts will not be the scientific source of truth.

Each notebook must follow this visible sequence:

1. `tl;dr` updated only after successful execution;
2. context, question, cell definitions, and assumptions;
3. parameter cell with run mode, paths, seed, and thresholds;
4. required-input and schema checks that fail clearly;
5. methods in Markdown immediately before focused code cells;
6. bounded tables and labelled plots;
7. statistical checks, effect sizes, intervals, and multiplicity handling;
8. limitations and claims that are or are not supported;
9. output manifest and takeaways tied to executed values.

Planned notebooks:

| Notebook | Purpose | Main checks and outputs |
|---|---|---|
| `00_methodology_data_audit.ipynb` | Annotation and input validation | Paths/modes, hashes, H5AD schema, published labels, AM-versus-IM/monocyte marker evidence, AT1/AT2 evidence, UMAPs, marker dotplots/feature plots, whole-core spatial maps, QC, and explicit separation of source labels from derived identities. |
| `01_mouse_MHCII_signature_mapping.ipynb` | Reproduce mouse cutoffs and build human-panel signatures | Audit both DE CSVs, exclude Cycling AM, formal ortholog mapping, panel overlap, group-direction comparison, final MHCII-low/high gene tables, and coverage limitations. |
| `02_human_AM_MHCII_states.ipynb` | Score and validate human macrophage states | Macrophage/AT2 cohorts, score construction, continuous distributions, one- versus two-component comparison, posterior confidence, donor/core stability, pseudobulk validation, and state-assignment export. |
| `03_AM_spatial_neighborhoods.ipynb` | Locations, nearest cells, and neighborhoods | Whole-core geometry, representative maps, nearest-AT2 distances, nearest non-AM identity, 20/30/50/100 um neighborhoods, boundary/density checks, within-core permutations, and donor-aware effect estimates. |
| `04_AM_AT2_communication.ipynb` | Panel-constrained cell-cell communication | Versioned ligand-receptor import, gene intersection, AM-to-AT2 and AT2-to-AM directions, expression/prevalence filters, spatial support, donor consistency, FDR, and ranked interactions. |
| `05_integrated_statistics_figures.ipynb` | Final statistical QA and figures | Primary/secondary endpoint registry, mixed models, age/sex/region analyses, leave-one-donor-out checks, sensitivity analyses, final figures, source-data tables, and claim-to-evidence audit. |
| `06_manuscript_methods_results.ipynb` | Reproducible manuscript text support | Executed numbers for Methods, Results, legends, limitations, and the proposed Figure 7 section; no direct Word edit without authorization. |

Notebook dependencies are sequential. A downstream notebook must read saved, versioned tables from the prior notebook rather than depend on hidden in-memory state. Each output records the producing notebook, proposal version, run mode, timestamp, input hashes, software versions, and parameter hash.

## Git directory structure

```text
XTY_AM_project/
|-- XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md
|-- README.md
|-- .gitignore
|-- config/
|   `-- analysis_config.yaml
|-- notebooks/
|   |-- 00_methodology_data_audit.ipynb
|   |-- 01_mouse_MHCII_signature_mapping.ipynb
|   |-- 02_human_AM_MHCII_states.ipynb
|   |-- 03_AM_spatial_neighborhoods.ipynb
|   |-- 04_AM_AT2_communication.ipynb
|   |-- 05_integrated_statistics_figures.ipynb
|   `-- 06_manuscript_methods_results.ipynb
|-- reference/
|   |-- orthologs/
|   `-- ligand_receptor/
|-- environment/
|   |-- environment.yml
|   `-- renv.lock
|-- outputs/
|   |-- local_test/
|   |-- hpc_full/
|   `-- manuscript_ready/
|-- logs/
`-- docs/
```

The repository will contain notebooks, configuration, environment locks, small reference tables with provenance, bounded local-test outputs, final summary/source-data tables, and manuscript-ready plots of reasonable size. It will not contain raw H5AD/PKL/RDS data, large intermediate matrices, full spatial neighbor graphs, caches, checkpoints, or unrestricted HPC outputs. Those exclusions will be enforced in `.gitignore`.

Local-test notebooks will be executed top-to-bottom and retain bounded outputs so plots, checks, and calculations can be inspected immediately. Full HPC notebooks will write to the HPC `analysis_v1/outputs/hpc_full` tree. Large full-run artifacts remain outside Git; hashes, manifests, summary tables, model outputs, and final figures are committed when appropriately sized.

## Validation, commit, and HPC handoff

For every notebook change:

1. run a structural notebook check with `nbformat`;
2. execute top-to-bottom in `local_test` mode with the same functions used on HPC;
3. inspect executed tables/plots and reconcile key counts independently;
4. run code-quality and statistical assertions;
5. confirm outputs are bounded and no source data enter Git;
6. run Git whitespace/diff checks;
7. update this proposal and change log for material method changes;
8. commit with a descriptive message and push immediately;
9. report a failed push explicitly and retain the local commit for retry.

The exact HPC execution command and environment activation will be recorded in `README.md` and in each notebook. The HPC run is acceptable only after all local-test checks pass and the software environment is reproduced from the committed lock file.

Deliverables include an input/hash manifest, audit/cohort report, ortholog and final signature tables, per-cell state assignments with confidence, donor/core state summaries, spatial source tables, model/permutation results, provenance-rich AM-AT2 interactions, manuscript-ready figures, source data, and executable Methods/Results/legend support.

## Manuscript placement and figure

Insert the human Xenium Results section after **"Adapted AMs support epithelial differentiation and lung homeostasis"** and before the Discussion. Working title:

> Human spatial transcriptomics identifies mouse-derived MHCII AM states and their spatial relationships with AT2 cells

The expected main figure is Figure 7, subject to final numbering:

1. dataset/cohort overview;
2. MHCII-high/low AM scores, state evidence, and markers;
3. state proportions by donor and age;
4. representative spatial maps;
5. nearest-AT2 distance and AT2 enrichment;
6. nearest/enriched cell-type composition;
7. spatially supported AM-AT2 interactions;
8. cross-species comparison.

Supplementary panels will show donor-level results, alternative definitions, radii, region comparisons, and complete interaction results.

## Risks and mitigation

| Risk | Mitigation |
|---|---|
| Canonical AM/AT2 genes are absent | Use observable signatures, report coverage, rely cautiously on provided AT2 labels, and avoid unmeasured-mechanism claims. |
| Mouse DE is one-versus-rest and included Cycling AM | Exclude Cycling AM rows, require coherent ResAM1/ResAM2 direction, document the limitation, and replace with direct pairwise DE if the mouse object becomes available. |
| Expanded panel signatures contain few significant genes | Report coverage and single-gene influence; use unweighted means as primary, bounded weights as sensitivity, and do not claim full mouse-state conservation. |
| Broad macrophage label includes interstitial macrophages or monocyte-like cells | Treat it only as a candidate pool; compare AM/IM/monocyte programs on UMAP and spatial maps; approve an AM flag before MHCII analysis; preserve ambiguous cells and sensitivity analyses. |
| Only two canonical AM markers are measured | Combine `MARCO/APOE` with negative evidence from measured IM/monocyte programs and spatial context; report the incomplete marker coverage and avoid overconfident AM claims. |
| States form a continuum | Compare mixture, clustering, and scores; use a continuous exposure if hard labels are unstable. |
| Cell-level pseudoreplication | Use donor-aware models, core summaries, within-core permutations, and leave-one-donor-out tests. |
| Density/boundary bias | Use local expectations, density covariates, spatial nulls, and boundary sensitivity. |
| Database matches are overinterpreted | Require measured expression, proximity, donor replication, and cautious language. |
| Pickle intermediates are unsafe/opaque | Prefer recomputation or trusted isolated conversion. |
| Unequal cores per donor | Use donor-weighted summaries and hierarchical models. |

## Decision gates before implementation

1. Approve this proposal and the mouse-derived MHCII-high/low framing.
2. Review notebook 00 and approve the AM-versus-IM/monocyte inclusion rule before MHCII scoring.
3. Approve the expanded-signature rule: full `ResAM1`/`ResAM2` DE table, adjusted P < 0.05, positive fold change, coherent relative direction, no 0.5 fold-change cutoff, and formal ortholog mapping.
4. Select and version the ligand-receptor resource.
5. Retain alveolar cores as primary unless a recorded scientific reason changes this.
6. Accept hard two-state labels only if stability criteria pass; otherwise use continuous scores.
7. Edit the Word manuscript only after results/figure review and explicit authorization.

## Change log

| Version | Date | Sections changed | Change and rationale | Expected effect |
|---|---|---|---|---|
| 0.1 | 2026-09-12 | All | Initial proposal created after inspection of the Xenium folder, official Figure 5 code, and target manuscript. It establishes required files, paper-panel mapping, a donor-aware two-state AM design, spatial/AT2 analyses, communication requirements, and living-document controls. | Reviewable source of truth before implementation. |
| 0.2 | 2026-09-12 | Document control; planned structure | Made `XTY_AM_project` the authoritative version-controlled working root and added the requirement to verify, commit, and push every material change. The parent proposal remains an immutable version 0.1 snapshot. | Future proposal, code, configuration, and eligible outputs are auditable through Git and the configured remote. |
| 0.3 | 2026-09-12 | Computing locations; inputs; AM definition; notebook architecture; Git structure; validation; risks; manuscript plan | Added local/HPC paths and run modes; made the full mouse `ResAM1`/`ResAM2` DE table authoritative; excluded Cycling AM; removed the 0.5 fold-change restriction; corrected scoring, mixture, and donor-bias safeguards; replaced standalone scripts with seven inspectable notebooks; and defined Git/HPC validation and artifact policies. | Makes the methodology consistent with the mouse MHCII-high/low origin and creates a locally testable, HPC-runnable, auditable notebook workflow. |
| 0.4 | 2026-09-12 | Executive summary; dataset interpretation; cohorts; Phase 2; notebook 00; risks; decision gates | Clarified that the published `Macrophages` label is only a candidate pool. Added explicit AM-versus-interstitial-macrophage/monocyte validation using measured positive and negative marker evidence plus spatial context, with a review gate before MHCII subtyping. | Prevents interstitial macrophages from being silently analyzed or reported as alveolar macrophages while retaining activated MHCII-high AMs for review. |
| 0.5 | 2026-09-12 | Status; notebook 00 implementation; validation | Implemented the first executable notebook with the deposited UMAP, focused macrophage/AT1/AT2 UMAP, marker dotplot, competing AM/IM/monocyte evidence scores, and complete-core spatial maps. Local testing uses four deterministic intact alveolar cores for expression-heavy steps. Corrected backed raw-count marker extraction to use explicit raw-variable indices and added an exact zero-pattern concordance assertion between transformed and raw matrices. | Makes annotation review immediately reproducible while preventing gene-column misalignment from corrupting detection fractions. No AM inclusion rule or MHCII state is finalized. |
| 0.6 | 2026-09-13 | Notebook 00 UMAP, HPC display, descriptive summaries, validation tests | Independent review identified and corrected three issues: the overview now plots all 332,063 cells rather than a stratified subset; both local and HPC modes use four deterministic complete alveolar cores for bounded spatial display while HPC marker calculations use all cells; and marker dotplots now average donor-level summaries equally. Added donor/core marker tables, donor/core macrophage-program summaries, stronger marker/HPC/display contracts, and clearer plot-title spacing. | Meets the full-UMAP requirement, prevents unbounded 70-core HPC figures, and removes unequal-cell-count donor dominance from descriptive marker summaries. |
