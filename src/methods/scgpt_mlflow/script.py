import os
import sys

import anndata as ad
import mlflow
import pandas as pd

## VIASH START
par = {
    "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
    "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
    "output": "output.h5ad",
    "model": "resources_test/.../model",
}
meta = {"name": "scgpt_mlflow"}
## VIASH END

sys.path.append(meta["resources_dir"])
from exit_codes import exit_non_applicable  # noqa: E402
from mlflow import embed_and_classify  # noqa: E402
from unpack import unpack_directory  # noqa: E402

print("====== scGPT (MLflow model) ======", flush=True)

print("\n>>> Reading training data...", flush=True)
print(f"Training H5AD file: '{par['input_train']}'", flush=True)
input_train = ad.read_h5ad(par["input_train"])
print(input_train, flush=True)

if input_train.uns["dataset_organism"] != "homo_sapiens":
    exit_non_applicable(
        f"scGPT (MLflow) can only be used with human data "
        f'(dataset_organism == "{input_train.uns["dataset_organism"]}")'
    )

print("\n>>> Reading test data...", flush=True)
print(f"Test H5AD file: '{par['input_test']}'", flush=True)
input_test = ad.read_h5ad(par["input_test"])
print(input_test, flush=True)

print("\n>>> Unpacking model...", flush=True)
model_dir, model_temp = unpack_directory(par["model"])

print("\n>>> Loading model...", flush=True)
model = mlflow.pyfunc.load_model(model_dir)
print(model, flush=True)

# Use embed_and_classify helper
predictions = embed_and_classify(
    input_train,
    input_test,
    model,
    layers=["counts"],
    var={"feature_name": "feature_name"},
    model_params={"gene_col": "feature_name"},
)

input_test.obs["label_pred"] = predictions
print(input_test.obs["label_pred"].value_counts(), flush=True)

print("\n>>> Storing output...", flush=True)
output = ad.AnnData(
    obs=input_test.obs[["label_pred"]],
    uns={
        "method_id": meta["name"],
        "dataset_id": input_test.uns["dataset_id"],
        "normalization_id": input_test.uns["normalization_id"],
    },
)
print(output, flush=True)

print("\n>>> Writing output to file...", flush=True)
print(f"Output H5AD file: '{par['output']}'", flush=True)
output.write_h5ad(par["output"], compression="gzip")

print("\n>>> Cleaning up temporary files...", flush=True)
if model_temp is not None:
    model_temp.cleanup()

print("\n>>> Done!", flush=True)
