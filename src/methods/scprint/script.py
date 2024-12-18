import anndata as ad
from scdataloader import Preprocessor
import sys
from huggingface_hub import hf_hub_download
from scprint.tasks import Embedder
from scprint import scPrint
import torch
import os
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

## VIASH START
par = {
    "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
    "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
    "output": "output.h5ad",
    "model_name": "large",
    "model": None,
}
meta = {"name": "scprint"}
## VIASH END

print(f"====== scPRINT version {scprint.__version__} ======", flush=True)

print("\n>>> Reading input data...", flush=True)
input_train = ad.read_h5ad(par['input_train'])
input_test = ad.read_h5ad(par['input_test'])

if input_train.uns["dataset_organism"] == "homo_sapiens":
    input_train.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    input_test.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
elif input_train.uns["dataset_organism"] == "mus_musculus":
    input_train.obs["organism_ontology_term_id"] = "NCBITaxon:10090"
    input_test.obs["organism_ontology_term_id"] = "NCBITaxon:10090"
else:
    raise ValueError(
        f"scPRINT requires human or mouse data, not '{input.uns['dataset_organism']}'"
    )

adata = input_train.copy()
adata.X = adata.layers["counts"]

adata_test = input_test.copy()
adata_test.X = adata_test.layers["counts"]

print("\n>>> Preprocessing train data...", flush=True)
preprocessor = Preprocessor(
    # Lower this threshold for test datasets
    min_valid_genes_id=1000 if input_train.n_vars < 2000 else 10000,
    # Turn off cell filtering to return results for all cells
    filter_cell_by_counts=False,
    min_nnz_genes=False,
    do_postp=False,
    # Skip ontology checks
    skip_validate=True,
)
adata = preprocessor(adata)

model_checkpoint_file = par["model"]
if model_checkpoint_file is None:
    print(f"\n>>> Downloading '{par['model_name']}' model...", flush=True)
    model_checkpoint_file = hf_hub_download(
        repo_id="jkobject/scPRINT", filename=f"{par['model_name']}.ckpt"
    )

print(f"Model checkpoint file: '{model_checkpoint_file}'", flush=True)
model = scPrint.load_from_checkpoint(
    model_checkpoint_file,
    transformer="normal",  # Don't use this for GPUs with flashattention
    precpt_gene_emb=None,
)

print("\n>>> Embedding train data...", flush=True)
if torch.cuda.is_available():
    print("CUDA is available, using GPU", flush=True)
    precision = "16"
    dtype = torch.float16
else:
    print("CUDA is not available, using CPU", flush=True)
    precision = "32"
    dtype = torch.float32

n_cores_available = len(os.sched_getaffinity(0))

print(f"Using {n_cores_available} worker cores")
embedder = Embedder(
    batch_size=32,
    how="random expr",
    max_len=4000,
    add_zero_genes=0,
    num_workers=n_cores_available,
    doclass=True,
    doplot=False,
    precision=precision,
    dtype=dtype,
    pred_embedding=["cell_type_ontology_term_id"],
    keep_all_cls_pred=False,
    output_expression="none"
)
embedded, _ = embedder(model, adata, cache=False)

print("\n>>> Matching predictions to labels...", flush=True)
# The labels predicted by scPRINT might be different to those in the dataset
# This step creates a mapping from the predicted labels to the dataset labels

# Get levels and values for real labels
labels = input_train.obs["label"].astype("category")
label_values = list(labels)
label_levels = sorted(list(labels.cat.categories))

# Get levels and values for predicted labels
predicted = embedded.obs["conv_pred_cell_type_ontology_term_id"].astype("category")
predicted_values = list(predicted)
predicted_levels = sorted(list(predicted.cat.categories))

# If there are any predicted labels that exactly match a dataset label we use them directly
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
preprocessor = Preprocessor(
    # Lower this threshold for test datasets
    min_valid_genes_id=1000 if input_test.n_vars < 2000 else 10000,
    # Turn off cell filtering to return results for all cells
    filter_cell_by_counts=False,
    min_nnz_genes=False,
    do_postp=False,
    # Skip ontology checks
    skip_validate=True,
)
adata_test = preprocessor(adata_test)

print("\n>>> Embedding test data...", flush=True)
embedder = Embedder(
    batch_size=32,
    how="random expr",
    max_len=4000,
    add_zero_genes=0,
    num_workers=n_cores_available,
    doclass=True,
    doplot=False,
    precision=precision,
    dtype=dtype,
    pred_embedding=["cell_type_ontology_term_id"],
    keep_all_cls_pred=False,
    output_expression="none"
)
embedded_test, _ = embedder(model, adata_test, cache=False)

print("\n>>> Converting predictions to labels...", flush=True)
input_test.obs["label_pred"] = embedded_test.obs["conv_pred_cell_type_ontology_term_id"].values
input_test.obs = input_test.obs.replace(dict(label_pred=matches))

print("\n>>> Storing output...", flush=True)
output = ad.AnnData(
  obs=input_test.obs[["label_pred"]],
  uns={
    'method_id': meta['name'],
    'dataset_id': input_test.uns['dataset_id'],
    'normalization_id': input_test.uns['normalization_id']
  }
)

print("\n>>> Writing output AnnData to file...", flush=True)
output.write_h5ad(par["output"], compression="gzip")

print("\n>>> Done!", flush=True)
