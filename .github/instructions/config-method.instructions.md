---
description: "Use when writing, fixing, or reviewing config.vsh.yaml files in src/methods/ or src/control_methods/. Covers required metadata, info fields, docker engine setup, nextflow runner labels, and how to verify components."
applyTo: "src/methods/**/config.vsh.yaml,src/control_methods/**/config.vsh.yaml"
---
# Method & Control Method Config Guidelines

## Structure Overview

```yaml
__merge__: /src/api/comp_method.yaml          # or comp_control_method.yaml
name: "my_method"                              # snake_case, unique
label: My Method                               # human-readable, used in tables
summary: "One sentence summary."               # used in overview tables
description: |                                 # multi-paragraph, used in docs
  Longer description...
references:                                    # omit for control methods
  doi:
    - 10.xxxx/xxxxx
links:                                         # omit for control methods
  repository: https://github.com/...
  documentation: https://...
info:
  preferred_normalization: log_cp10k           # or counts, log_scran_pooling
  variants:
    my_method_default:
    my_method_variant:
      some_param: value
arguments:                                     # only if method has extra params
  - name: "--some_param"
    type: integer
    description: "..."
    example: 100                                 # use example, NOT default
    info:
      test_default: 1                            # override value used during viash test only
resources:
  - type: python_script                        # or r_script
    path: script.py                            # or script.R
engines:
  - type: docker
    image: openproblems/base_python:1          # see base images below
    setup:
      - type: python
        packages: [package1, package2]
runners:
  - type: executable
  - type: nextflow
    directives:
      label: [midtime, highmem, midcpu]        # adjust to actual needs
```

## Methods vs Control Methods

| Field | Method | Control Method |
|---|---|---|
| `__merge__` | `/src/api/comp_method.yaml` | `/src/api/comp_control_method.yaml` |
| `references` | required | omit |
| `links` | recommended | omit |
| inputs | `input_train`, `input_test` | `input_train`, `input_test`, `input_solution` |

## Required Metadata Fields

- `name`: unique, matches `[a-z][a-z0-9_]*`
- `label`: short human-readable name
- `summary`: one sentence
- `description`: one or more paragraphs
- `references.doi` (methods only): list of DOIs

## info Section

- `preferred_normalization`: one of `counts`, `log_cp10k`, `log_scran_pooling`
- `variants`: each key becomes a separate benchmark entry. Override any argument value by nesting it under the variant key. Every method needs at least one variant with the same name as the method.

## Arguments

- Do **not** set `default` on any argument — defaults belong to the library, not the config. Use `example` to document a typical value.
- Use `info.test_default` to override a parameter value **only during `viash test`** (not in benchmarks). This is useful to reduce epoch counts, disable slow quality checks, etc., so tests run quickly without affecting real benchmark results.
- Argument names use `--snake_case`. Viash exposes them in the script as `par['snake_case']` (Python) or `par$snake_case` (R).
- After adding, removing, or renaming any argument, regenerate the `## VIASH START` block in the script so the `par` dict stays in sync:
  ```bash
  viash config inject src/methods/<name>/config.vsh.yaml
  ```

```yaml
arguments:
  - name: --n_epochs
    type: integer
    description: "Number of training epochs."
    example: 100
    info:
      test_default: 1          # 1 epoch during testing for speed
  - name: --flow_threshold
    type: double
    description: "Flow error threshold. Set to 0 to skip flow quality check."
    example: 0.4
    info:
      test_default: 0          # skip check during testing
```

## Base Docker Images

| Image | Use for |
|---|---|
| `openproblems/base_python:1` | Python, CPU |
| `openproblems/base_r:1` | R, CPU |
| `openproblems/base_pytorch_nvidia:1` | PyTorch + NVIDIA GPU |
| `openproblems/base_tensorflow_nvidia:1` | TensorFlow + NVIDIA GPU |

## Nextflow Runner Labels

Set in `runners[type=nextflow].directives.label`. Pick one from each category:

| Category | Options |
|---|---|
| Time | `lowtime`, `midtime`, `hightime` |
| Memory | `lowmem`, `midmem`, `highmem`, `veryhighmem` |
| CPU | `lowcpu`, `midcpu`, `highcpu` |
| GPU (optional) | `gpu`, `biggpu` |

## Rebuilding the Docker Image

After changing the `setup` section:
```bash
viash run src/methods/<name>/config.vsh.yaml -- ---setup cachedbuild
```

## Verification

```bash
viash test src/methods/<name>/config.vsh.yaml
```

Both test scripts must succeed (`2 out of 2`).
