---
description: "Use when writing, fixing, or reviewing method/metric script.R files in src/methods/, src/metrics/, or src/control_methods/. Covers script style, API compliance, and how to verify components."
applyTo: "src/methods/**/script.R,src/metrics/**/script.R,src/control_methods/**/script.R"
---
# Method & Metric Script Guidelines (R)

## Core Principle

`script.R` should represent **typical bioinformatician usage** of the tool with minimal modifications. Only adapt what is strictly necessary to:
1. Read inputs from the paths provided by `par`
2. Pass the right data structures to the method
3. Convert the method's output back into the expected output structures
4. Write outputs to `par$output`

Do **not** restructure the method's native API, add abstraction layers, or rewrite the algorithm logic.

## Finding API Specs

Input/output file formats are defined in `src/api/`. Key files:
- `file_train.yaml` / `file_test.yaml` — training and test AnnData fields
- `file_prediction.yaml` — expected output format for methods
- `file_score.yaml` — expected output format for metrics
- `comp_method.yaml`, `comp_metric.yaml` — component argument specs

Always check these before deciding what fields to read or write.

## The `## VIASH START` / `## VIASH END` Block

This block is **auto-generated** by viash from the component's `config.vsh.yaml` arguments. It is replaced at build/test time with a real CLI parser. Keep it in the script only as a local development convenience.

- **Do not edit it manually** to add or remove parameters — edit `config.vsh.yaml` instead.
- After adding, removing, or renaming an argument in the config, regenerate the block:
  ```bash
  viash config inject src/methods/<name>/config.vsh.yaml
  ```
- Argument names in the config (`--my_param`) map directly to `par$my_param` keys.

## Common Patterns

**Reading inputs:**
```r
library(anndata, warn.conflicts = FALSE)
input_train <- read_h5ad(par$input_train)
input_test <- read_h5ad(par$input_test)
```

**Writing prediction output:**
```r
output <- anndata::AnnData(
  obs = data.frame(
    row.names = input_test$obs_names,
    label_pred = pred_labels
  ),
  uns = list(
    method_id = meta$name,
    dataset_id = input_test$uns[["dataset_id"]],
    normalization_id = input_test$uns[["normalization_id"]]
  ),
  shape = c(input_test$n_obs, 0L)
)
output$write_h5ad(par$output, compression = "gzip")
```

## Dependency Fixes

If a library has a dependency conflict (e.g., incompatible with a newer Bioconductor version, `anndata` R package, etc.), prefer replacing it with an alternative that provides the same model/algorithm natively rather than pinning transitive dependencies.

Update `config.vsh.yaml` to remove the broken package from the `setup` block when replacing it.

## Verification

After any change to a method script or config, verify with:
```bash
viash test src/methods/<name>/config.vsh.yaml
# or
viash test src/metrics/<name>/config.vsh.yaml
```

Both test scripts must succeed (`2 out of 2`).
