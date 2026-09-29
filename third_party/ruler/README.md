# Official RULER runtime

This environment is deliberately separate from the controller and AIPerf
environments. It contains only the dependencies imported by the pinned
RULER synthetic-data generators.

Create it from the repository root:

```sh
uv sync --project third_party/ruler
uv run --project third_party/ruler python -m nltk.downloader punkt punkt_tab
```

The official generators also require their source data. The checkout tracks
`english_words.json` through git-lfs, so fetch it before anything reads it:

```sh
git -C "$checkout" lfs pull
```

From the pinned RULER checkout, run the repository-provided download helpers once:

```sh
repo_root="$PWD"
cd "$repo_root/data/ruler-source/scripts/data/synthetic/json"
bash download_qa_dataset.sh
uv run --project "$repo_root/third_party/ruler" python download_paulgraham_essay.py
cd "$repo_root"
```

If the helper stalls on its first HTTP HotpotQA mirror, interrupt it and run
its HTTPS fallback from the same `json` directory:

```sh
wget --tries=3 --timeout=20 \
  'https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json' \
  -O hotpotqa.json
```

The resulting JSON files are local inputs to workload preparation and are not
checked into this project. Their SHA-256 hashes are recorded in the generated
RULER manifest.
