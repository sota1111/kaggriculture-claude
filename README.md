# kaggriculture-claude

Claude lineage for the Kaggle `kaggriculture` competition.

Build and validate:

```bash
bash scripts/build_submission.sh
```

Submit:

```bash
kaggle competitions submit -c kaggriculture \
  -f submission.tar.gz -m "kaggriculture-claude champion"
```

The archive contains `main.py` at its root and exports `agent(obs)`.
