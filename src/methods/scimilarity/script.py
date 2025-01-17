import os
import sys
import tempfile
import zipfile
import tarfile

import anndata as ad
import scimilarity
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

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
    print(
        f"SCimilarity can only be used with human data "
        f"(dataset_organism == \"{input_train.uns['dataset_organism']}\")",
        flush=True
    )
    sys.exit(99)

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

print("\n>>> Matching predictions to labels...", flush=True)
# The labels in the SCimilarity model will be different to those in the dataset
# This step creates a mapping from the SCimilarity labels to the dataset labels

# Get levels and values for real labels
labels = input_train.obs["label"].astype("category")
label_values = list(labels)
label_levels = sorted(list(labels.cat.categories))

# Get levels and values for predicted labels
predicted = train.obs["prediction"].astype("category")
predicted_values = list(predicted)
predicted_levels = sorted(list(predicted.cat.categories))

# If there are any predicted labels that exactly match a dataset label we use
# them directly
matches = {}
lower_label_levels = [l.lower() for l in label_levels]
print("---- EXACT MATCHES ----", flush=True)
for pred in predicted_levels:
    if pred.lower() in lower_label_levels:
        matches[pred] = label_levels[lower_label_levels.index(pred.lower())]
        print(pred, flush=True)

# Remove any predicted labels that have exact matches
predicted_levels = [pred for pred in predicted_levels if pred not in matches.keys()]

# Calculate Jaccard distance between each pair of predicted and dataset labels
jaccard = np.zeros((len(label_levels), len(predicted_levels)))
combos = [(label, pred) for label in label_levels for pred in predicted_levels]

for label, pred in combos:
    labels_bin = [1 if l == label else 0 for l in label_values]
    predicted_bin = [1 if p == pred else 0 for p in predicted_values]

    label_idx = label_levels.index(label)
    predicted_idx = predicted_levels.index(pred)
    jaccard[label_idx, predicted_idx] = distance.jaccard(labels_bin, predicted_bin)

# Use linear sum assignment to match predicted labels to dataset labels based on
# the Jaccard distances. This algorithm may not match all predicted labels so
# we remove any that have been matched and repeat until all predicted labels are
# matched to a dataset label.
print("\n---- INFERRED MATCHES ----", flush=True)
print(f"{'PREDICTED' : <40}{'LABEL' : <40}", flush=True)
while not all(pred in matches for pred in predicted_levels):
    # Get predicted labels that have not yet been matched
    not_matched = [pred for pred in predicted_levels if pred not in matches.keys()]
    not_matched_idx = [predicted_levels.index(pred) for pred in not_matched]

    # Get assignments for currently unmatched predicted labels
    assignments = linear_sum_assignment(jaccard[:, not_matched_idx])

    # Store any new matches
    for label, pred in zip(assignments[0], assignments[1]):
        predicted_level = not_matched[pred]
        label_level = label_levels[label]
        matches[predicted_level] = label_level

        if (len(predicted_level) > 39):
            predicted_level = predicted_level[:36] + '...'

        if (len(label_level) > 39):
            label_level = label_level[:36] + '...'

        print(f"{predicted_level: <40}{label_level: <40}", flush=True)

print("\n>>> Preprocessing test data...", flush=True)
print("Creating input object...", flush=True)
test = ad.AnnData(X=input_test.layers["counts"], layers={"counts":input_test.layers["counts"]})
test.var_names = input_test.var["feature_name"]
print("Aligning genes...", flush=True)
test = scimilarity.utils.align_dataset(
    test,
    cell_annotator.gene_order,
    gene_overlap_threshold=gene_overlap_threshold,
)
test = scimilarity.utils.consolidate_duplicate_symbols(test)
print("Normalizing...", flush=True)
test = scimilarity.utils.lognorm_counts(test)

print("\n>>> Embedding test data...", flush=True)
test.obsm["X_scimilarity"] = cell_annotator.get_embeddings(test.X)

print("\n>>> Annotating test data...", flush=True)
predictions, nn_idxs, nn_dists, nn_stats = cell_annotator.get_predictions_knn(
    test.obsm["X_scimilarity"]
)
test.obs["prediction"] = predictions.values
print(test.obs["prediction"].value_counts(), flush=True)

print("\n>>> Converting predictions to labels...", flush=True)
input_test.obs["label_pred"] = test.obs["prediction"].values
input_test.obs = input_test.obs.replace(dict(label_pred=matches))
print(input_test.obs["label_pred"].value_counts(), flush=True)

print("\n>>> Storing output...", flush=True)
output = ad.AnnData(
    obs=input_test.obs[["label_pred"]],
    uns={
        'method_id': meta['name'],
        'dataset_id': input_test.uns['dataset_id'],
        'normalization_id': input_test.uns['normalization_id']
    }
)
print(output, flush=True)

print("\n>>> Writing output to file...", flush=True)
print(f"Output H5AD file: '{par['output']}'", flush=True)
output.write_h5ad(par["output"], compression="gzip")

if model_temp is not None:
    print("\n>>> Cleaning up temporary directories...", flush=True)
    model_temp.cleanup()

print("\n>>> Done!", flush=True)
