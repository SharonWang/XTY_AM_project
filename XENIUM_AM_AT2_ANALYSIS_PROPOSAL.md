# Human Lung Xenium AM-AT2 Spatial Analysis Proposal

**Project:** Human lung alveolar macrophage heterogeneity
**Study:** Xu et al., *Cellular hallmarks and aging clock of the human lung parenchyma*, Nature Communications (2026), PMID 42457688
**Repository root:** `D:\Xiaonan\CODEX_projects\Xiaotong_AM\XTY_AM_project`
**Data root:** `D:\Xiaonan\CODEX_projects\Xiaotong_AM\Spatial\Spatial`
**Version:** 0.2
**Status:** Proposed design; awaiting scientific review before implementation
**Created / last updated:** 2026-09-12

## Document control

This is the living source of truth for the Xenium AM-AT2 analysis. Every material change to the question, cohort, inputs, state definition, statistics, spatial method, communication method, figure plan, or interpretation must update:

1. the version and last-updated date;
2. the affected section;
3. the change log, including rationale and expected effect; and
4. affected configuration or scripts.

Outputs are not current until these items agree. If the proposal and analysis configuration conflict, record the resolution in the change log. Do not overwrite source data. Keep versioned analysis documents, code, configuration, logs, and normal-sized derived outputs under the D: repository root; keep the supplied Xenium source data at the declared D: data root. Create or modify no project files on C:.

Every material project change must be committed and pushed to the configured Git remote after verification. Commit messages must describe the scientific or technical change. Large source datasets and oversized derived artifacts must not be added to Git; track manifests, hashes, scripts, summaries, and appropriately sized source-data tables instead.

## Executive summary

The analysis will test whether the broad Xenium `Macrophages` population contains two reproducible AM states and whether those states differ in spatial relationships with AT2 and other lung cells. The working comparison is an **early/resident-like state** versus an **adapted-like/MHC-II-inflammatory state**, defined only with genes measured by the 389-gene panel. A two-state result will not be forced if the evidence instead supports a continuum or more states.

Spatial inference will be within tissue cores, with donor as the biological replicate and core nested within donor. Alveolar cores are primary; all regions form a sensitivity analysis. Nearest-cell distances, multi-radius neighborhoods, within-core permutations, and donor-aware models will evaluate AM-AT2 association. Ligand-receptor results will require measured genes, expression, spatial support, and donor reproducibility and will be interpreted as putative communication, not proof of signaling.

This document is a proposal only. It makes no biological result claim yet.

## Questions and hypotheses

Primary questions:

1. Can macrophage-labelled cells be separated into two stable AM-like states across donors and cores?
2. Do states differ in distance to AT2 and in nearest/local cell-type composition?
3. Are state-specific AM-AT2 ligand-receptor programs supported by local co-occurrence?

Secondary questions address continuous age, sex, tissue region, cross-species agreement with the manuscript's mouse AM states, and robustness to classification and spatial choices.

Working markers are:

- early/resident-like: `MARCO`, `VSIG4`, `APOE`, `CD163`, with `CHIT1` supportive;
- adapted-like/MHC-II-inflammatory: `CD74`, `HLA-DQB1`, `CD14`, `CD44`, `ICAM1`, `CXCL9`, and `CXCL10`.

Candidate communication will be limited to panel-observable pairs, for example `SPP1-CD44`, `MIF-CD74/CD44/CXCR4`, `APP-CD74`, and directionally appropriate WNT-receptor pairs.

## Dataset interpretation

The master object has 332,063 cells, 389 genes, 22 donors, and 70 cores. It contains donor/core metadata, QC, final cell labels, centroids, embeddings, and spatial/connectivity graphs. There are 33,375 `Macrophages` and 21,495 `AT2` cells. Every core contains both; 59 cores have at least 50 of each.

Regions comprise 40 alveolar, 22 vascular, 7 bronchial, and 1 core labelled literally `None`. Donors contribute one to four cores, so cells are not independent replicates. Centroids support cell-proximity analysis, but missing morphology images, polygons, and transcript coordinates preclude segmentation review, boundary-contact measurement, and subcellular localization.

## Required-file manifest

### Essential and operational inputs

| File | Requirement | Purpose |
|---|---|---|
| `Spatial\Spatial\xenium.h5ad` | **Required** | Authoritative expression, metadata, cell labels, donor/core structure, and spatial centroids. This is the only existing file strictly indispensable for the requested analysis. |
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
| Mouse early/adapted AM signatures from the manuscript | Required for cross-species comparison | Store exact signatures and origins under `reference\`; map orthologs and intersect with the 389-gene panel. |
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

- `Macrophages` from the master H5AD;
- alveolar (`A`) cores;
- donors/cores passing existing QC and the stated minimum-cell rule.

Sensitivity cohorts:

1. all regions with region adjustment or stratification;
2. stricter macrophage and AT2 cell-count thresholds;
3. cells restricted by a panel-derived AM-likeness score;
4. exclusion of the core labelled literally `None`.

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

1. Subset macrophage-labelled and AT2-labelled cells from the master.
2. Tabulate counts by donor, core, region, age, and sex.
3. Flag sparse cores. The default primary spatial-summary threshold is at least 50 macrophages and 50 AT2; retain smaller cores only where statistically valid and report threshold sensitivity.
4. Check expression/QC for contamination, doublets, and region-specific artifacts.

### Phase 3 - Two-state AM definition

1. Use a documented normalization suitable for the targeted panel; retain raw counts for count-aware models.
2. Construct early/resident-like and adapted-like/MHC-II-inflammatory scores from the measured markers listed above.
3. Compare:
   - a two-component model on the score contrast;
   - donor-aware unsupervised clustering plus biological annotation;
   - nearest-centroid/signature assignment from ortholog-mapped, panel-intersected mouse signatures.
4. Assess marker consistency, separation, donor mixing, bootstrap stability, and leave-one-donor-out stability.
5. Test solutions with two to four states. Do not force two groups if unstable.
6. Retain classification confidence and an `uncertain` sensitivity label.

**Decision gate:** use the method with best donor-level stability and panel-supported interpretation. If no stable two-state result emerges, use continuous scores as primary exposures and hard labels only descriptively.

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

1. Extract the manuscript's mouse early/adapted AM signatures.
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
7. Scripts regenerate all outputs from declared inputs without manual editing.
8. Every final figure has source data and an audit trail.

## Planned structure and deliverables

All new files remain under the project root:

```text
D:\Xiaonan\CODEX_projects\Xiaotong_AM\XTY_AM_project\
|-- XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md
|-- reference\
|-- config\
|-- scripts\
|-- outputs\
|   |-- audit\
|   |-- qc\
|   |-- am_states\
|   |-- spatial\
|   |-- communication\
|   |-- tables\
|   `-- figures\
`-- logs\
```

Deliverables will include an input/hash manifest, frozen configuration, audit/cohort report, per-cell state assignments and confidence, donor/core state summaries, spatial source tables, model/permutation results, a provenance-rich AM-AT2 interaction table, reproducible scripts, manuscript-ready figures with source data, and Methods/Results/legends/limitations text.

## Manuscript placement and figure

Insert the human Xenium Results section after **"Adapted AMs support epithelial differentiation and lung homeostasis"** and before the Discussion. Working title:

> Human spatial transcriptomics identifies conserved AM states and preferential association of adapted-like AMs with AT2 cells

The expected main figure is Figure 7, subject to final numbering:

1. dataset/cohort overview;
2. AM embedding, state scores, and markers;
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
| Broad macrophage label includes non-AMs | Use alveolar cores, AM-likeness sensitivity, donor/core marker review, and uncertain labels. |
| States form a continuum | Compare mixture, clustering, and scores; use a continuous exposure if hard labels are unstable. |
| Cell-level pseudoreplication | Use donor-aware models, core summaries, within-core permutations, and leave-one-donor-out tests. |
| Density/boundary bias | Use local expectations, density covariates, spatial nulls, and boundary sensitivity. |
| Database matches are overinterpreted | Require measured expression, proximity, donor replication, and cautious language. |
| Pickle intermediates are unsafe/opaque | Prefer recomputation or trusted isolated conversion. |
| Unequal cores per donor | Use donor-weighted summaries and hierarchical models. |

## Decision gates before implementation

1. Approve this proposal and the two-state framing.
2. Approve the exact mouse early/adapted signatures or authorize extraction.
3. Select and version the ligand-receptor resource.
4. Retain alveolar cores as primary unless a recorded scientific reason changes this.
5. Accept hard two-state labels only if stability criteria pass; otherwise use continuous scores.
6. Edit the Word manuscript only after results/figure review and explicit authorization.

## Change log

| Version | Date | Sections changed | Change and rationale | Expected effect |
|---|---|---|---|---|
| 0.1 | 2026-09-12 | All | Initial proposal created after inspection of the Xenium folder, official Figure 5 code, and target manuscript. It establishes required files, paper-panel mapping, a donor-aware two-state AM design, spatial/AT2 analyses, communication requirements, and living-document controls. | Reviewable source of truth before implementation. |
| 0.2 | 2026-09-12 | Document control; planned structure | Made `XTY_AM_project` the authoritative version-controlled working root and added the requirement to verify, commit, and push every material change. The parent proposal remains an immutable version 0.1 snapshot. | Future proposal, code, configuration, and eligible outputs are auditable through Git and the configured remote. |
