import os
import sys
import tempfile
import zipfile
import tarfile

import anndata as ad
import scimilarity

## VIASH START
par = {
    'input_train': 'resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad',
    'input_test': 'resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad',
    "output": "output.h5ad",
    "model": "model_v1.1",
}
meta = {
    "name": "scimilarity",
}
## VIASH END

print(f"====== SCimilarity version {scimilarity.__version__} ======", flush=True)

print("\n>>> Reading training data...", flush=True)
print(f"Training H5AD file: '{par['input_train']}'", flush=True)
input_train = ad.read_h5ad(par['input_train'])
print(input_train, flush=True)

if input_train.uns["dataset_organism"] != "homo_sapiens":
    raise ValueError(
        f"SCimilarity can only be used with human data "
        f"(dataset_organism == \"{input_train.uns['dataset_organism']}\")"
    )

print("\n>>> Reading test data...", flush=True)
print(f"Test H5AD file: '{par['input_test']}'", flush=True)
input_test = ad.read_h5ad(par['input_test'])
print(input_test, flush=True)

if os.path.isdir(par["model"]):
    print(f"\n>>> Using existing model directory...", flush=True)
    model_temp = None
    model_dir = par["model"]
else:
    model_temp = tempfile.TemporaryDirectory()
    model_dir = model_temp.name

    if zipfile.is_zipfile(par["model"]):
        print(f"\n>>> Extracting model directory from .zip...", flush=True)
        print(f".zip path: '{par['model']}'", flush=True)
        with zipfile.ZipFile(par["model"], "r") as zip_file:
            zip_file.extractall(model_dir)
    elif tarfile.is_tarfile(par["model"]) and par["model"].endswith(
        ".tar.gz"
    ):
        print(f"\n>>> Extracting model directory from .tar.gz...", flush=True)
        print(f".tar.gz path: '{par['model']}'", flush=True)
        with tarfile.open(par["model"], "r:gz") as tar_file:
            tar_file.extractall(model_dir)
            model_dir = os.path.join(model_dir, os.listdir(model_dir)[0])
    else:
        raise ValueError(
            f"The 'model' argument should be a directory a .zip file or a .tar.gz file"
        )

print(f"Model directory: '{model_dir}'", flush=True)

print("\n>>> Loading SCimilarity model...", flush=True)
cell_annotator = scimilarity.CellAnnotation(model_path=model_dir)

print("\n>>> Preprocessing training data...", flush=True)
print("Creating input object...", flush=True)
# Some of the functions modify the adata so make sure we have a copy
train = ad.AnnData(X=input_train.layers["counts"], layers={"counts":input_train.layers["counts"]})
# Set var_names to gene symbols
train.var_names = input_train.var["feature_name"]
print("Aligning genes...", flush=True)
# Check the number of genes in the dataset and reduce the overlap threshold if
# necessary (mostly for subsampled test datasets)
gene_overlap_threshold = 5000
if 0.8 * train.n_vars < gene_overlap_threshold:
    from warnings import warn

    warn(
        f"The number of genes in the dataset ({train.n_vars}) "
        f"is less than or close to {gene_overlap_threshold}. "
        f"Setting gene_overlap_threshold to 0.8 * n_var ({int(0.8 * train.n_vars)})."
    )
    gene_overlap_threshold = int(0.8 * train.n_vars)

train = scimilarity.utils.align_dataset(
    train,
    cell_annotator.gene_order,
    gene_overlap_threshold=gene_overlap_threshold,
)
train = scimilarity.utils.consolidate_duplicate_symbols(train)
print("Normalizing...", flush=True)
train = scimilarity.utils.lognorm_counts(train)

print("\n>>> Embedding training data...", flush=True)
train.obsm["X_scimilarity"] = cell_annotator.get_embeddings(train.X)

print("\n>>> Annotating training data...", flush=True)
predictions, nn_idxs, nn_dists, nn_stats = cell_annotator.get_predictions_knn(
    train.obsm["X_scimilarity"]
)
train.obs["prediction"] = predictions.values
print(train.obs["prediction"].value_counts(), flush=True)

# print("Store outputs", flush=True)
# output = ad.AnnData(
#     obs=adata.obs[[]],
#     var=adata.var[[]],
#     obsm={
#         "X_emb": cell_embeddings,
#     },
#     uns={
#         "dataset_id": adata.uns["dataset_id"],
#         "normalization_id": adata.uns["normalization_id"],
#         "method_id": meta["name"],
#     },
# )
# print(output)

# print("Write output to file", flush=True)
# output.write_h5ad(par["output"], compression="gzip")

# if model_temp is not None:
#     print("Cleanup model directory", flush=True)
#     model_temp.cleanup()

print("\n>>> Done!", flush=True)
