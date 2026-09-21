# Project Charter

## Goal

Build a reproducible OAI-based multimodal pipeline for future knee OA progression.

## Core linked domains

- knee X-ray imaging
- WOMAC/KOOS or other OAI patient-reported outcomes
- structured clinical/demographic risk factors
- linked physical-performance measures if sufficiently available

Only genuinely participant-linked OAI data belong in the core study. Unrelated external gait,
sensor, imaging, or text datasets must not be presented as linked to OAI participants.

## Primary research question

Can baseline radiographic, patient-reported, clinical, and physical-performance information
predict future OA progression better than any individual measurement domain alone?

## Modeling concept

Separate domain-specific models/encoders followed by late fusion.

This is a future modeling concept, not an authorization to choose architectures or train models
during Milestone 0/1.

## Key experiment

Per-domain ablation and comparison of single-domain versus multimodal performance.

## Research-first priorities

- cohort construction
- participant/knee/visit linkage
- missingness
- imaging QC
- longitudinal outcome construction
- leakage prevention
- calibration
- reproducibility
- interpretability

## Milestone 0/1 scope

This milestone covers repository initialization, documentation, safe local data handling, and a
read-only inventory utility. It excludes OAI-specific mappings, cohort implementation, outcome
implementation, data splitting, feature construction, and all model building or training.

## Decision governance

Every scientific decision must be documented with its rationale, evidence, date, and reviewer.
The following require explicit human review after inspection of the authorized OAI release:

- source files and identifier fields used for participant/knee/visit linkage;
- what qualifies as baseline and the prediction horizon;
- radiographic and/or symptomatic progression criteria;
- inclusion/exclusion criteria and handling of bilateral knees;
- physical-performance measure availability and inclusion thresholds;
- missing-data strategy, evaluation metrics, and calibration assessment;
- participant-grouped development partitions and any temporal constraints.

No silent defaults should be encoded for these items.

## Data governance

Raw and access-controlled OAI data remain local and outside Git. File inventories and derived
outputs must be reviewed for sensitive content before they are shared or committed. The project
will preserve provenance from each authorized source file to every derived analytic artifact.
