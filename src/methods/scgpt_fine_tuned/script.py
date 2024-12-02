import tempfile
import time
import anndata as ad
import json
import os
from pathlib import Path
from typing import Tuple, Dict
import warnings
import gdown
import torch
import numpy as np
from scipy.sparse import issparse
import torch
import sklearn
from torchtext.vocab import Vocab
import scgpt

## VIASH START
par = {
  'input_train': 'resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad',
  'input_test': 'resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad',
  'output': 'output.h5ad'
}
meta = {
  'name': 'scgpt',
  'temp_dir': 'tmp'
}
## VIASH END

def prepare_data() -> Tuple[Dict[str, torch.Tensor]]:
  masked_values_train = scgpt.tokenizer.random_mask_value(
    tokenized_train["values"],
    mask_ratio=mask_ratio,
    mask_value=mask_value,
    pad_value=pad_value,
  )

  masked_values_valid = scgpt.tokenizer.random_mask_value(
    tokenized_valid["values"],
    mask_ratio=mask_ratio,
    mask_value=mask_value,
    pad_value=pad_value,
  )

  print(
    f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
    f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
  )

  input_gene_ids_train, input_gene_ids_valid = (
    tokenized_train["genes"],
    tokenized_valid["genes"],
  )

  input_values_train, input_values_valid = masked_values_train, masked_values_valid
  target_values_train, target_values_valid = (
    tokenized_train["values"],
    tokenized_valid["values"],
  )

  tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
  tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

  tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
  tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

  train_data_pt = {
    "gene_ids": input_gene_ids_train,
    "values": input_values_train,
    "target_values": target_values_train,
    "batch_labels": tensor_batch_labels_train,
    "celltype_labels": tensor_celltype_labels_train,
  }

  valid_data_pt = {
    "gene_ids": input_gene_ids_valid,
    "values": input_values_valid,
    "target_values": target_values_valid,
    "batch_labels": tensor_batch_labels_valid,
    "celltype_labels": tensor_celltype_labels_valid,
  }

  return train_data_pt, valid_data_pt

from torch.utils.data import Dataset, DataLoader

# dataset
class SeqDataset(Dataset):
  def __init__(self, data: Dict[str, torch.Tensor]):
    self.data = data

  def __len__(self):
    return self.data["gene_ids"].shape[0]

  def __getitem__(self, idx):
    return {k: v[idx] for k, v in self.data.items()}

# data_loader
def prepare_dataloader(
  data_pt: Dict[str, torch.Tensor],
  batch_size: int,
  shuffle: bool = False,
  intra_domain_shuffle: bool = False,
  drop_last: bool = False,
  num_workers: int = 0,
) -> DataLoader:
  if num_workers == 0:
      num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

  dataset = SeqDataset(data_pt)

  data_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
    pin_memory=True,
  )
  return data_loader


def train(model: torch.nn.Module, loader: DataLoader) -> None:
  """
  Train the model for one epoch.
  """
  model.train()
  (
    total_loss,
    total_mse,
    total_cls,
    total_cce,
    total_mvc,
    total_ecs
  ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  total_error = 0.0
  start_time = time.time()
  print("done training!")
  num_batches = len(loader)
  for batch, batch_data in enumerate(loader):
    input_gene_ids = batch_data["gene_ids"].to(device)
    input_values = batch_data["values"].to(device)
    target_values = batch_data["target_values"].to(device)
    batch_labels = batch_data["batch_labels"].to(device)
    celltype_labels = batch_data["celltype_labels"].to(device)

    src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
    with torch.cuda.amp.autocast(enabled=amp):
      output_dict = model(
        input_gene_ids,
        input_values,
        src_key_padding_mask=src_key_padding_mask,
        batch_labels=None,
        CLS=CLS,
        CCE=CCE,
        MVC=MVC,
        ECS=ECS,
        do_sample=False,
      )

      masked_positions = input_values.eq(mask_value)  # the postions to predict
      loss = 0.0
      metrics_to_log = {}
      if MLM:
        loss_mse = criterion(output_dict["mlm_output"], target_values, masked_positions)
        loss = loss + loss_mse
        metrics_to_log = {"train/mse": loss_mse.item()}
      if CLS:
        loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
        loss = loss + loss_cls
        metrics_to_log.update({"train/cls": loss_cls.item()})

        error_rate = 1 - (
          (output_dict["cls_output"].argmax(1) == celltype_labels)
          .sum()
          .item()
        ) / celltype_labels.size(0)
      if CCE:
        loss_cce = 10 * output_dict["loss_cce"]
        loss = loss + loss_cce
        metrics_to_log.update({"train/cce": loss_cce.item()})
      if MVC:
        loss_mvc = criterion(
          output_dict["mvc_output"], target_values, masked_positions
        )
        loss = loss + loss_mvc
        metrics_to_log.update({"train/mvc": loss_mvc.item()})
      if ECS:
        loss_ecs = 10 * output_dict["loss_ecs"]
        loss = loss + loss_ecs
        metrics_to_log.update({"train/ecs": loss_ecs.item()})

      model.zero_grad()
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings("always")
        torch.nn.utils.clip_grad_norm_(
          model.parameters(),
          1.0,
          error_if_nonfinite=False if scaler.is_enabled() else True,
        )
        if len(w) > 0:
          logger.warning(
            f"Found infinite gradient. This may be caused by the gradient scaler. The current scale is {scaler.get_scale()}. This warning can be ignored if no longer occurs after autoscaling of the scaler."
          )
      scaler.step(optimizer)
      scaler.update()

      total_loss += loss.item()
      total_mse += loss_mse.item() if MLM else 0.0
      total_cls += loss_cls.item() if CLS else 0.0
      total_cce += loss_cce.item() if CCE else 0.0
      total_mvc += loss_mvc.item() if MVC else 0.0
      total_ecs += loss_ecs.item() if ECS else 0.0
      
      if batch % log_interval == 0 and batch > 0:
        lr = scheduler.get_last_lr()[0]
        ms_per_batch = (time.time() - start_time) * 1000 / log_interval
        cur_loss = total_loss / log_interval
        cur_mse = total_mse / log_interval
        cur_cls = total_cls / log_interval if CLS else 0.0
        cur_cce = total_cce / log_interval if CCE else 0.0
        cur_mvc = total_mvc / log_interval if MVC else 0.0
        cur_ecs = total_ecs / log_interval if ECS else 0.0

        cur_error = total_error / log_interval
        logger.info(
          f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
          f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
          f"loss {cur_loss:5.2f} | "
          + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if MLM else "")
          + (f"cls {cur_cls:5.2f} | " if CLS else "")
          + (f"err {cur_error:5.2f} | " if CLS else "")
          + (f"cce {cur_cce:5.2f} |" if CCE else "")
          + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
          + (f"ecs {cur_ecs:5.2f} |" if ECS else "")
        )
        total_loss = 0
        total_mse = 0
        total_cls = 0
        total_cce = 0
        total_mvc = 0
        total_ecs = 0
        total_error = 0
        start_time = time.time()

def evaluate(model: torch.nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
  """
  Evaluate the model on the evaluation data.
  """
  model.eval()
  total_loss = 0.0
  total_error = 0.0
  total_num = 0
  predictions = []
  with torch.no_grad():
    for batch_data in loader:
      input_gene_ids = batch_data["gene_ids"].to(device)
      input_values = batch_data["values"].to(device)
      batch_labels = batch_data["batch_labels"].to(device)
      celltype_labels = batch_data["celltype_labels"].to(device)

      src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
      with torch.cuda.amp.autocast(enabled=hyperparameter_defaults["amp"]):
        output_dict = model(
          input_gene_ids,
          input_values,
          src_key_padding_mask=src_key_padding_mask,
          batch_labels=None,
          CLS=CLS,  # evaluation does not need CLS or CCE
          CCE=False,
          MVC=False,
          ECS=False,
          do_sample=False,
        )
        output_values = output_dict["cls_output"]
        loss = criterion_cls(output_values, celltype_labels)

      total_loss += loss.item() * len(input_gene_ids)
      accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
      total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
      total_num += len(input_gene_ids)
      preds = output_values.argmax(1).cpu().numpy()
      predictions.append(preds)

  if return_raw:
    return np.concatenate(predictions, axis=0)

  return total_loss / total_num, total_error / total_num


def test(model: torch.nn.Module, adata: DataLoader) -> float:
  all_counts = (
    adata.layers["X_binned"].A
    if issparse(adata.layers["X_binned"])
    else adata.layers["X_binned"]
  )

  celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
  celltypes_labels = np.array(celltypes_labels)

  batch_ids = adata.obs["batch_id"].tolist()
  batch_ids = np.array(batch_ids)

  tokenized_test = scgpt.tokenizer.tokenize_and_pad_batch(
    all_counts,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
  )

  input_values_test = scgpt.tokenizer.random_mask_value(
    tokenized_test["values"],
    mask_ratio=mask_ratio,
    mask_value=mask_value,
    pad_value=pad_value,
  )

  test_data_pt = {
    "gene_ids": tokenized_test["genes"],
    "values": input_values_test,
    "target_values": tokenized_test["values"],
    "batch_labels": torch.from_numpy(batch_ids).long(),
    "celltype_labels": torch.from_numpy(celltypes_labels).long(),
  }

  test_loader = DataLoader(
    dataset=SeqDataset(test_data_pt),
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), batch_size // 2),
    pin_memory=True,
  )

  model.eval()
  predictions = evaluate(
    model,
    loader=test_loader,
    return_raw=True,
  )

  return predictions, celltypes_labels

print('Reading input files', flush=True)
input_train = ad.read_h5ad(par['input_train'])
input_test = ad.read_h5ad(par['input_test'])
input_solution = ad.read_h5ad("resources_test/task_label_projection/cxg_immune_cell_atlas/solution.h5ad")

if input_train.uns["dataset_organism"] != "homo_sapiens":
  raise ValueError(
    f"scGPT can only be used with human data "
    f"(dataset_organism is \"{input_train.uns['dataset_organism']}\")"
  )

drive_path = f"https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
print(f'Downloading scGPT_human model from {drive_path}', flush=True)
model = meta['temp_dir']
gdown.download_folder(drive_path, output=model, quiet=True)
print(f"Model directory: '{model}'", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: '{device}'", flush=True)

scgpt.utils.set_seed(0)

# logging
log_interval = 100  # iterations
save_eval_interval = 5  # epochs
logger = scgpt.logger
scgpt.utils.add_file_handler(logger, "run.log")

# Settings
n_bins = 51
max_seq_len = 3001
ecs_threshold = 0.0 
amp = True  # Automatic Mixed Precision
lr = 1e-4
schedule_ratio = 0.9  # ratio of epochs for learning rate schedule
schedule_interval = 1
epochs = 10
batch_size = 32
mask_ratio = 0.0
mask_value = -1
pad_value = -2
# settings for training
MLM = False  # whether to use masked language modeling
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = ecs_threshold > 0  # Elastic cell similarity objective

### Pre-process data

# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]

# Merge datasets to appy the same pre-processing steps
input_train.obs["celltype"] = input_train.obs["label"].astype("category")
input_train.obs["is_test"]  = "0"
input_test.obs["is_test"]  = "1" 
input_train.var["gene_name"] = input_train.var["feature_name"]
input_test.var["gene_name"] = input_test.var["feature_name"]
input_train = ad.concat([input_train, input_test], join="outer", merge="same", uns_merge="same")

# make the batch category column
batch_id_labels = input_train.obs["is_test"].astype("category").cat.codes.values
input_train.obs["batch_id"] = batch_id_labels
celltype_id_labels = input_train.obs["celltype"].astype("category").cat.codes.values
celltypes = input_train.obs["celltype"].unique()
num_types = len(celltypes)
id2type = dict(enumerate(input_train.obs["celltype"].astype("category").cat.categories))
input_train.obs["celltype_id"] = celltype_id_labels
input_train.var["gene_name"] = input_train.var["feature_name"].tolist()

model_dir = Path(model)
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"
vocab = scgpt.tokenizer.gene_tokenizer.GeneVocab.from_file(vocab_file)
for s in special_tokens:
  if s not in vocab:
    vocab.append_token(s)

input_train.var["id_in_vocab"] = [
  1 if gene in vocab else -1 for gene in input_train.var["gene_name"]
]    
gene_ids_in_vocab = np.array(input_train.var["id_in_vocab"])
input_train = input_train[:, input_train.var["id_in_vocab"] >= 0]

logger.info(
  f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
  f"in vocabulary of size {len(vocab)}."
)

# model
with open(model_config_file, "r") as f:
  model_configs = json.load(f)
logger.info(
  f"Resume model from {model_file}, the model args will override the "
  f"config {model_config_file}."
)
embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]

preprocessor = scgpt.preprocess.Preprocessor(
  use_key="counts",  # the key in input_train.layers to use as raw data
  binning=n_bins,  # whether to bin the raw data and to what number of bins
  result_binned_key="X_binned",  # the key in input_train.layers to store the binned data

)

input_test = input_train[input_train.obs["is_test"] == "1"]
input_train = input_train[input_train.obs["is_test"] == "0"]

preprocessor(input_train, batch_key=None)
preprocessor(input_test, batch_key=None)

all_counts = (
  input_train.layers["X_binned"].A
  if issparse(input_train.layers["X_binned"])
  else input_train.layers["X_binned"]
)
genes = input_train.var["gene_name"].tolist()

celltypes_labels = input_train.obs["celltype_id"].tolist()  # make sure count from 0
celltypes_labels = np.array(celltypes_labels)

batch_ids = input_train.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

(
  train_data,
  valid_data,
  train_celltype_labels,
  valid_celltype_labels,
  train_batch_labels,
  valid_batch_labels,
) = sklearn.model_selection.train_test_split(
  all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
if model is None:
  vocab = Vocab(VocabPybind(genes + special_tokens, None))  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

tokenized_train = scgpt.tokenizer.tokenize_and_pad_batch(
  train_data,
  gene_ids,
  max_len=max_seq_len,
  vocab=vocab,
  pad_token=pad_token,
  pad_value=pad_value,
)
tokenized_valid = scgpt.tokenizer.tokenize_and_pad_batch(
  valid_data,
  gene_ids,
  max_len=max_seq_len,
  vocab=vocab,
  pad_token=pad_token,
  pad_value=pad_value,
)
logger.info(
  f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
  f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
  f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
  f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)

### Load the pre-trained scGPT model

ntokens = len(vocab)  # size of vocabulary
model = scgpt.model.TransformerModel(
  ntokens,
  embsize,
  nhead,
  d_hid,
  nlayers,
  n_cls=num_types,
  vocab=vocab,
  dropout=0.2,
  pad_token=pad_token,
  pad_value=pad_value,
  num_batch_labels=num_batch_types,
  n_input_bins=n_bins,
  ecs_threshold=ecs_threshold,
  use_fast_transformer=True,
)

# only load params that are in the model and match the size
map_location = torch.device('cpu') if device == 'cpu' else None
model_dict = model.state_dict()
pretrained_dict = torch.load((model_file), map_location=map_location)
pretrained_dict = {
  k: v
  for k, v in pretrained_dict.items()
  if k in model_dict and v.shape == model_dict[k].shape
}
for k, v in pretrained_dict.items():
  logger.info(f"Loading params {k} with shape {v.shape}")
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")

model.to(device)

criterion = scgpt.loss.masked_mse_loss
criterion_cls = torch.nn.CrossEntropyLoss()
criterion_dab = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4 if amp else 1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=amp)

### Finetune scGPT with task-specific objectives

best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None

import copy
for epoch in range(1, epochs + 1):
  epoch_start_time = time.time()
  train_data_pt, valid_data_pt = prepare_data()
  train_loader = prepare_dataloader(
    train_data_pt,
    batch_size=batch_size,
    shuffle=False,
    intra_domain_shuffle=True,
    drop_last=False,
  )
  valid_loader = prepare_dataloader(
    valid_data_pt,
    batch_size=batch_size,
    shuffle=False,
    intra_domain_shuffle=False,
    drop_last=False,
  )

  train(model, loader=train_loader)
  val_loss, val_err = evaluate(model, loader=valid_loader)
  elapsed = time.time() - epoch_start_time
  logger.info("-" * 89)
  logger.info(
    f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
    f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
  )
  logger.info("-" * 89)

  if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model = copy.deepcopy(model)
    best_model_epoch = epoch
    logger.info(f"Best model with score {best_val_loss:5.4f}")

  scheduler.step()

### Inference with fine-tuned scGPT model

predictions, labels = test(best_model, input_test)
input_test.obs['label_pred'] = [id2type[p] for p in predictions]

save_dict = {
  "predictions": predictions,
  "labels": labels,
  "id_maps": id2type
}

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
