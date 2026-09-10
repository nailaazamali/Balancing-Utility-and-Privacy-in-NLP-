# Uploading to GitHub

The folder is prepared for upload as the code and aggregate-results release for the paper.

## Upload steps

1. Extract `XAILeakageNLP_GitHub_Ready.zip`.
2. Open a terminal in the `XAILeakageNLP_GitHub` folder.
3. Run:

```bash
python scripts/validate_repository.py
```

4. If the checks pass, create and push the repository:

```bash
git init
git add .
git commit -m "Initial paper implementation release"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/XAILeakageNLP.git
git push -u origin main
```

The files can also be uploaded through the GitHub web interface.

## Check before making it public

- Confirm the author names in `CITATION.cff`.
- Confirm that the current `LICENSE` is agreed by all copyright holders.
- Keep `data/archived_run/` labelled as a separate archived experiment.
- Do not describe the repository as an exact row-level reproduction of the paper.
- If the original paper split or row-level human annotations are released later, add them as a separate version rather than rebuilding them from aggregate counts.

Once the repository is final, a tagged release such as `v1.0.0` can be created for citation.
