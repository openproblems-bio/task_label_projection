import sys
import pathlib
import time

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from adata_hf_datasets import InitialEmbedder
from adata_hf_datasets.dataset import AnnDataSetConstructor
from mmcontext.embed.model_utils import prepare_model_and_embed
from mmcontext.eval import OmicsQueryAnnotator
from sentence_transformers import SentenceTransformer

## VIASH START
par = {
    "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
    "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
    "output": "output.h5ad",
    # mmcontext-specific parameters
    "model_id": None,
    "batch_size": 64,
    "cache_dir": "mmcontext_cache",
    "initial_embedding": "gs10k",
    "n_top": 5,
    "force_model": False,
}
meta = {
    "name": "mmcontext_zeroshot",
    # "resources_dir" will be injected by Viash when built; default here for local testing
    "resources_dir": "target/executable/methods/mmcontext_zeroshot",
}
## VIASH END

sys.path.append(meta["resources_dir"])



def compute_mmcontext_embeddings(
    adata_path: str,
    adata: ad.AnnData,
    cache_dir: str,
    model_id: str,
    initial_embedding: str,
) -> np.ndarray:
    """
    Compute mmcontext embeddings for all cells in `adata` using the same
    pipeline as your external script, adapted for normalized-then-log data.
    """
    cache_dir_path = pathlib.Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    # Initial embeddings (e.g. gs10k)
    emb_key = f"X_{initial_embedding}"
    print("=== Computing initial embeddings ===", flush=True)
    embedder = InitialEmbedder(method=initial_embedding)
    embedder.prepare(adata)
    embeddings = embedder.embed(adata)
    adata.obsm[emb_key] = embeddings

    # Persist processed AnnData so AnnDataSetConstructor can link to it
    processed_path = adata_path.replace(".h5ad", "_pp.h5ad")
    print(f"=== Writing processed AnnData to {processed_path} ===", flush=True)
    adata.write_h5ad(processed_path)

    print("=== Building AnnData dataset for mmcontext ===", flush=True)
    adata.obs["sample_idx"] = adata.obs.index
    constructor = AnnDataSetConstructor(dataset_format="single")
    constructor.add_anndata(
        adata,
        sentence_keys=["sample_idx"],
        adata_link=processed_path,
    )
    ds = constructor.get_dataset()

    print("=== Loading MMContext model ===", flush=True)
    model = SentenceTransformer(model_id, trust_remote_code=True)

    print("=== Computing MMContext embeddings ===", flush=True)
    emb_df, _ = prepare_model_and_embed(
        model,
        ds,
        index_col="sample_idx",
        adata_download_dir=str(cache_dir_path),
        main_col="cell_sentence_1",
        layer_key=emb_key,
    )

    # Ensure embeddings align with adata.obs index order
    emb_df_sorted = emb_df.set_index("sample_idx").reindex(
        list(adata.obs.index)
    ).reset_index()
    embedding_matrix = np.vstack(emb_df_sorted["embedding"].to_numpy())
    adata.obsm["mmcontext_emb"] = embedding_matrix
    return embedding_matrix


print("Reading input files", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])

# Example organism check; adjust or remove depending on mmcontext’s capabilities.
#if input_train.uns.get("dataset_organism") not in (None, "homo_sapiens"):
#    exit_non_applicable(
#        "mmcontext_zeroshot currently only supports human data "
#        f'(dataset_organism == "{input_train.uns["dataset_organism"]}")'
#    )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: '{device}'", flush=True)

print("Preparing test data for mmcontext", flush=True)
if "counts" in input_test.layers:
    input_test.X = input_test.layers["counts"]

# Updated preprocessing: normalize first, then log-transform
print("=== Preprocessing test data (normalize -> log1p) ===", flush=True)
sc.pp.normalize_total(input_test)
sc.pp.log1p(input_test)

# Determine model ID
default_model_id = "jo-mengr/mmcontext-pubmedbert-gs10k"
model_id = par.get("model_id") or default_model_id
print(f"Using mmcontext model ID: {model_id}", flush=True)

# Quick safety check: initial embedding should match the checkpoint naming.
if (not par.get("force_model", False)) and (par["initial_embedding"] not in model_id):
    raise ValueError(
        "The configured initial embedding method does not appear to match the mmcontext model.\n"
        f"- initial_embedding: '{par['initial_embedding']}'\n"
        f"- model_id: '{model_id}'\n\n"
        "The initial embedding method needs to match the one that was used to train the model.\n"
        "If you are absolutely sure they match, re-run with `--force_model true` to skip this check."
    )

start_time = time.time()

embeddings = compute_mmcontext_embeddings(
    adata_path=par["input_test"],
    adata=input_test,
    cache_dir=par["cache_dir"],
    model_id=model_id,
    initial_embedding=par["initial_embedding"],
)

print("=== Creating annotator and predicting labels ===", flush=True)
annotator_model = SentenceTransformer(model_id, trust_remote_code=True)
annotator = OmicsQueryAnnotator(annotator_model)

# Use unique labels from the test data as the zero-shot label set
label_space = input_test.obs["label"].unique()
annotator.annotate_omics_data(
    input_test,
    labels=label_space,
    emb_key="mmcontext_emb",
    n_top=par["n_top"],
)

# OmicsQueryAnnotator writes the best label to obs["best_label"]
input_test.obs["label_pred"] = input_test.obs["best_label"]
input_test.obs["label_pred"] = input_test.obs["label_pred"].astype("category")

elapsed_time = time.time() - start_time
print(f"mmcontext zero-shot prediction completed in {elapsed_time:.2f} seconds", flush=True)

print("Write output AnnData to file", flush=True)
output = ad.AnnData(
    obs=input_test.obs[["label_pred"]],
    uns={
        "method_id": meta["name"],
        "dataset_id": input_test.uns["dataset_id"],
        "normalization_id": input_test.uns["normalization_id"],
    },
)
output.write_h5ad(par["output"], compression="gzip")
