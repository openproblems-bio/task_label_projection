---
description: "Use when writing, fixing, or reviewing config.vsh.yaml files in src/metrics/. Covers required metadata, the info.metrics list structure, docker engine setup, nextflow runner labels, and how to verify components."
applyTo: "src/metrics/**/config.vsh.yaml"
---
# Metric Config Guidelines

## Structure Overview

Metrics differ from methods: metadata (`label`, `summary`, `description`, `references`) lives inside the `info.metrics` list, not at the top level. A single component can expose multiple metric values.

```yaml
__merge__: /src/api/comp_metric.yaml
name: "my_metric"                              # snake_case, unique component name
info:
  metrics:
    - name: my_metric_value1                   # snake_case, unique metric name
      label: My Metric Value 1                 # human-readable, used in tables
      summary: "One sentence summary."
      description: "Longer description."
      references:
        doi: 10.xxxx/xxxxx
      min: 0
      max: 1
      maximize: true                           # true if higher = better
    - name: my_metric_value2
      label: My Metric Value 2
      summary: "..."
      description: "..."
      references:
        doi: 10.xxxx/xxxxx
      min: 0
      max: 1
      maximize: false
resources:
  - type: python_script                        # or r_script
    path: script.py                            # or script.R
engines:
  - type: docker
    image: openproblems/base_python:1          # see base images below
    setup:
      - type: python
        packages: [scikit-learn]
runners:
  - type: executable
  - type: nextflow
    directives:
      label: [midtime, midmem, midcpu]
```

## Required Fields per Metric Entry

Each entry in `info.metrics` must have:
- `name`: unique metric identifier, snake_case
- `label`: short human-readable name
- `summary`: one sentence
- `description`: full description
- `references.doi`: DOI(s) for the metric
- `min` / `max`: numeric range of possible values
- `maximize`: `true` if higher score = better performance

## Base Docker Images

| Image | Use for |
|---|---|
| `openproblems/base_python:1` | Python, CPU |
| `openproblems/base_r:1` | R, CPU |

Metrics rarely need GPU images.

## Nextflow Runner Labels

Metrics are typically lightweight. Use conservative defaults:

| Category | Options |
|---|---|
| Time | `lowtime`, `midtime`, `hightime` |
| Memory | `lowmem`, `midmem`, `highmem` |
| CPU | `lowcpu`, `midcpu`, `highcpu` |

## Rebuilding the Docker Image

After changing the `setup` section:
```bash
viash run src/metrics/<name>/config.vsh.yaml -- ---setup cachedbuild
```

## Verification

```bash
viash test src/metrics/<name>/config.vsh.yaml
```

Both test scripts must succeed (`2 out of 2`).
