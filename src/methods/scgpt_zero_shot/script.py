import os
import tarfile
import zipfile
import anndata as ad
import gdown
import scgpt
import torch
import tempfile
import numpy as np

## VIASH START
par = {
  'input_train': 'resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad',
  'input_test': 'resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad',
  'output': 'output.h5ad',
  'model_name': 'scGPT_human',
  'model': 'scGPT_human',
  "n_hvg": 3000,
}
meta = {
  'name': 'scgpt'
}
## VIASH END

# Functions to perform similarity search when faiss is not used
def l2_sim(a, b):
  sims = -np.linalg.norm(a - b, axis=1)
  return sims

def get_similar_vectors(vector, ref, top_k=10):
  sims = l2_sim(vector, ref)
  top_k_idx = np.argsort(sims)[::-1][:top_k]
  return top_k_idx, sims[top_k_idx]

print('Reading input files', flush=True)
input_train = ad.read_h5ad(par['input_train'])
input_test = ad.read_h5ad(par['input_test'])

if input_train.uns["dataset_organism"] != "homo_sapiens":
  raise ValueError(
    f"scGPT can only be used with human data "
    f"(dataset_organism is \"{input_train.uns['dataset_organism']}\")"
  )

if par["model"] is None:
  print(f"\n>>> Downloading '{par['model_name']}' model...", flush=True)
  model_drive_ids = {
      "scGPT_human": "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
      "scGPT_CP": "1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB",
  }
  drive_path = (
      f"https://drive.google.com/drive/folders/{model_drive_ids[par['model_name']]}"
  )
  model_temp = tempfile.TemporaryDirectory()
  model_dir = model_temp.name
  print(f"Downloading from '{drive_path}'", flush=True)
  gdown.download_folder(drive_path, output=model_dir, quiet=True)
else:
  if os.path.isdir(par["model"]):
    print(f">>> Using model directory...", flush=True)
    model_temp = None
    model_dir = par["model"]
  else:
    model_temp = tempfile.TemporaryDirectory()
    model_dir = model_temp.name
    if zipfile.is_zipfile(par["model"]):
      print(f">>> Extracting model from .zip...", flush=True)
      print(f".zip path: '{par['model']}'", flush=True)
      with zipfile.ZipFile(par["model"], "r") as zip_file:
          zip_file.extractall(model_dir)
    elif tarfile.is_tarfile(par["model"]) and par["model"].endswith(".tar.gz"):
      print(f">>> Extracting model from .tar.gz...", flush=True)
      print(f".tar.gz path: '{par['model']}'", flush=True)
      with tarfile.open(par["model"], "r:gz") as tar_file:
          tar_file.extractall(model_dir)
          model_dir = os.path.join(model_dir, os.listdir(model_dir)[0])
    else:
      raise ValueError(
        f"The 'model' argument should be a directory a .zip file or a .tar.gz file"
      )

print(f"Model directory: '{model_dir}'", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: '{device}'", flush=True)
print("GPU is recommended") if device == "cpu" else None

print('Preprocessing and embedding train data', flush=True)
input_train.X = input_train.layers["counts"]
if par["n_hvg"]:
  print(f"Selecting top {par['n_hvg']} highly variable genes", flush=True)
  idx = input_train.var["hvg_score"].to_numpy().argsort()[::-1][: par["n_hvg"]]
  input_train = input_train[:, idx].copy()

ref_embed = scgpt.tasks.embed_data(
  input_train,
  model_dir,
  gene_col="feature_name",
  obs_to_save="label",
  batch_size=64,
  device=device,
  use_fast_transformer=False,
  return_new_adata=True,
)

print('Preprocessing and embedding test data', flush=True)
input_test.X = input_test.layers["counts"]
if par["n_hvg"]:
  print(f"Selecting top {par['n_hvg']} highly variable genes", flush=True)
  idx = input_test.var["hvg_score"].to_numpy().argsort()[::-1][: par["n_hvg"]]
  input_test = input_test[:, idx].copy()

test_embed = scgpt.tasks.embed_data(
  input_test,
  model_dir,
  gene_col="feature_name",
  batch_size=64,
  device=device,
  use_fast_transformer=False,
  return_new_adata=True,
)

print('Generate predictions', flush=True)
k = 10  # number of neighbors
if par["use_faiss"]:
  import faiss
  # Declaring index
  index = faiss.IndexFlatL2(ref_embed.shape[1])
  index.add(ref_embed)
  # Query dataset, k - number of closest elements (returns 2 numpy arrays)
  distances, labels = index.search(test_embed, k)
  
idx_list=[i for i in range(test_embed.X.shape[0])]
preds = []
for k in idx_list:
  if par["use_faiss"]:
    idx = labels[k]
  else:
    idx, sim = get_similar_vectors(test_embed.X[k][np.newaxis, ...], ref_embed.X, k)
  pred = ref_embed.obs["label"][idx].value_counts()
  preds.append(pred.index[0])

input_test.obs["label_pred"] = preds
input_test.obs["label_pred"] = input_test.obs["label_pred"].astype("category")

print("Write output AnnData to file", flush=True)
output = ad.AnnData(
  obs=input_test.obs[["label_pred"]],
  uns={
    'method_id': meta['name'],
    'dataset_id': input_test.uns['dataset_id'],
    'normalization_id': input_test.uns['normalization_id']
  }
)
output.write_h5ad(par['output'], compression='gzip')
