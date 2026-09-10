# Release checklist

- [ ] Run `python scripts/validate_repository.py`.
- [ ] Confirm the aggregate CSV values match the final manuscript.
- [ ] Keep `configs/reference.json` as the paper configuration record.
- [ ] Keep `data/archived_run/` clearly labelled as a different archived experiment.
- [ ] Do not add raw Davidson tweet text or usernames.
- [ ] Do not restore old human-evaluation spreadsheets.
- [ ] Keep external baseline generation separate from the common evaluation scripts.
- [ ] Confirm `CITATION.cff` and `LICENSE` with the co-authors.
- [ ] Optionally record the final software environment with `python scripts/capture_environment.py`.
- [ ] Create a tagged release after the public repository has been checked.
