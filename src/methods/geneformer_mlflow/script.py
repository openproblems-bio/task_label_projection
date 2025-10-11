import os
import sys
import tempfile

import anndata as ad
import mlflow
import pandas as pd
import numpy as np
import sklearn.neighbors

## VIASH START
par = {
    "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
    "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
    "output": "output.h5ad",
    "model": "resources_test/.../model",
}
meta = {"name": "geneformer_mlflow"}
## VIASH END

sys.path.append(meta["resources_dir"])
from exit_codes import exit_non_applicable  # noqa: E402
from unpack import unpack_directory  # noqa: E402

print("====== Geneformer (MLflow model) ======", flush=True)

n_processors = os.cpu_count()
print(f"Available processors: {n_processors}", flush=True)

print("\n>>> Reading training data...", flush=True)
print(f"Training H5AD file: '{par['input_train']}'", flush=True)
input_train = ad.read_h5ad(par["input_train"])
print(input_train, flush=True)

if input_train.uns["dataset_organism"] != "homo_sapiens":
    exit_non_applicable(
        f"Geneformer (MLflow) can only be used with human data "
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

# Temporary file for model input
h5ad_file = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)

print("\n>>> Embedding training data...", flush=True)
print("Writing temporary input H5AD file...", flush=True)
input_adata = ad.AnnData(X=input_train.layers["counts"].copy())
input_adata.var_names = input_train.var["feature_id"].values
input_adata.obs["cell_idx"] = np.arange(input_adata.n_obs)
input_adata.obs["n_counts"] = input_adata.X.sum(axis=1)
input_adata.var["ensembl_id"] = input_train.var["feature_id"]
print(input_adata, flush=True)

print(f"Temporary H5AD file: '{h5ad_file.name}'", flush=True)
input_adata.write(h5ad_file.name)
del input_adata

print("Running model...", flush=True)
input_df = pd.DataFrame({"input_uri": [h5ad_file.name]})
embedding_train = model.predict(input_df, params={"nproc": n_processors})

print("\n>>> Training kNN classifier...", flush=True)
classifier = sklearn.neighbors.KNeighborsClassifier()
classifier.fit(embedding_train, input_train.obs["label"].astype(str))

print("\n>>> Embedding test data...", flush=True)
print("Writing temporary input H5AD file...", flush=True)
input_adata = ad.AnnData(X=input_test.layers["counts"].copy())
input_adata.var_names = input_test.var["feature_id"].values
input_adata.obs["cell_idx"] = np.arange(input_adata.n_obs)
input_adata.obs["n_counts"] = input_adata.X.sum(axis=1)
input_adata.var["ensembl_id"] = input_test.var["feature_id"]
print(input_test, flush=True)

print(f"Temporary H5AD file: '{h5ad_file.name}'", flush=True)
input_adata.write(h5ad_file.name)
del input_adata

print("Running model...", flush=True)
input_df = pd.DataFrame({"input_uri": [h5ad_file.name]})
embedding_test = model.predict(input_df, params={"nproc": n_processors})

print("\n>>> Classifying test data...", flush=True)
input_test.obs["label_pred"] = classifier.predict(embedding_test)
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
h5ad_file.close()
os.unlink(h5ad_file.name)

print("\n>>> Done!", flush=True)
