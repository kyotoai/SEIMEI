Topic: {topic}
n_samples_per_topic: {n_samples_per_topic}
n_hyper_params: {n_hyper_params}
file_stub: {file_stub}

# Python module requirements
- Produce a single Python module that defines `generate(csv_output_path: str, n_samples_per_topic: int, n_hyper_params: int) -> None`.
- Inside `generate`, sample at least `n_hyper_params` distinct hyper-parameter configurations for the topic and emit roughly `n_samples_per_topic` rows per configuration.
- Use deterministic randomness (e.g., `rng = numpy.random.default_rng(1729 + idx)`) so reruns are reproducible.
- The module may rely on Python’s standard library plus `numpy`, `pandas`, `json`, `uuid`, `datetime`, and `random`. No external data fetches or file writes beyond the provided CSV path.
- Include a CLI entry point with `argparse` that forwards `--csv-output-path`, `--n-samples-per-topic`, and `--n-hyper-params` into `generate`.

# Data expectations
- Write a UTF-8 CSV with a header row. Recommended columns include:
  - `topic` — constant string matching the topic name.
  - `sample_id` — unique identifier per row (e.g., UUID4).
  - `params_json` — JSON dump of the hyper-parameters used for that row.
  - `payload_json` — JSON describing the row-level features relevant to the topic.
  - `timestamp` — ISO8601 datetime if the topic has a natural timeline, else empty string.
- Capture semantics representative of the topic (seasonality, anomalies, churn patterns, uplift, etc.) and show how the chosen hyper-parameters influence the generated data.
- Ensure the output directory already exists; do not attempt to create sibling files or folders.

# Narrative deliverables
- Provide a user-facing question that probes understanding of the hyper-parameters or generation logic.
- Provide a concise, unambiguous reference answer that explains the correct conclusion and references the underlying generator behaviour.
- Vary the phrasing of questions across topics so the dataset is not repetitive.

# Quality checklist
- Validate that the CSV contains more than just a header and that column names are populated.
- Avoid excessive reliance on pandas-specific features that obscure the hyper-parameters—clarity is preferred over cleverness.
- Keep docstrings or inline comments minimal but informative enough for a human reviewer to tweak hyper-parameters quickly.
