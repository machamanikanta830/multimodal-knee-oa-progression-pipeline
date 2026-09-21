# Local Data Layout

OAI files are access-controlled research data. Keep them local, comply with the applicable OAI
access and usage terms, and never commit them to Git.

## Where to place files

Place the untouched downloaded release under:

```text
data/raw/<release-or-download-directory>/
```

You may instead keep the release outside the repository and pass its absolute directory to the
inventory command. This can be preferable when institutional storage controls require it.

The pipeline expects official OAI data packages; variable codings and dictionary details are
documented in `docs/data-dictionary.md`. Package structure and access governance are detailed in
`docs/data_access_and_privacy.md`. Exact package names and structure must always be checked against
official OAI documentation. Do not rename files or infer variable/visit meanings merely to match an
expected layout.

## Intended separation

- `raw/`: immutable copies of downloaded source files; never edited in place.
- `interim/`: reproducible intermediate products such as validated linkage tables or QC outputs.
- `processed/`: analysis-ready datasets produced by versioned code after scientific review.

The contents of all three directories are ignored by Git because derived files may still contain
access-controlled information. Only the directory markers and this README are intended to be
tracked.

## Safe first step

From the repository root:

```bash
python -m data.inventory data/raw
python -m data.inventory data/raw --json
```

The inventory utility does not modify files or print raw row values. It does expose filenames,
column names, and aggregate dimensions, so review its output before sharing or committing it.

## Prohibited practices

- Do not commit raw OAI data, derived row-level data, or access credentials.
- Do not combine unrelated external gait, sensor, or text data and describe it as participant-
  linked OAI data.
- Do not create OAI variable mappings, visit mappings, or progression labels without evidence
  from the actual release and documented scientific review.
