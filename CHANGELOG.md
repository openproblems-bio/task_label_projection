# task_label_projection devel

* Add new method: CellMapper, which is a k-NN based approach to map cells across representations and can be used for label projection. Two versions are included here, one based on PCA or CCA embeddings (`linear`) and one based on an scvi embedding (`scvi`)  (PR #22)
* Update scPRINT to better handle large datasets, including a new default model (PR #20)

# task_label_projection 2.0.0

A major update to the OpenProblems framework, switching from a Python-based framework to a Viash + Nextflow-based framework. This update features the same concepts as the previous version, but with a new implementation that is more flexible, scalable, and maintainable.

## Migration

* Added expected input/output interfaces in `src/api` and document them in `README.md`.

* Store common resources used across tasks in a git submodule `common`.

* Methods, metrics, workflows and other components are implemented as Viash components with a per-component Docker image.

## New functionality

* Switched to larger datasets derived from CELLxGENE.

* Added scGPT zero shot (PR #2).

* Added scGPT fine-tuned (PR #3).

* Added SCimilarity (PR #4).

* Added UCE method (PR #6).

* Added geneformer (PR #7, #16).

* Added scPRINT (PR #8).

## Major changes

* Updated the task API (PR #9).

## Bug fixes

* Convert to dgCMatrix in SingleR (PR #5).

* Multiple fixes prior to release (PR #11, #13, #14, #15, #17).

## Documentation

* Update README (PR #10).


# task_label_projection 1.0.0

This version can be found [here](https://github.com/openproblems-bio/openproblems/tree/v1.0.0/openproblems/tasks/label_projection).
