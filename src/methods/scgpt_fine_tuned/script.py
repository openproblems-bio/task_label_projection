import sys
import tarfile
import tempfile
import time
import zipfile
import anndata as ad
import json
import os
import gdown
import torch
import numpy as np
from scipy.sparse import issparse
import torch
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
import scgpt
from sklearn.model_selection import train_test_split
import copy


## VIASH START
par = {
  'input_train': 'resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad',
  'input_test': 'resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad',
  'output': 'output.h5ad',
  'model': 'scGPT_human'
}
meta = {
  'name': 'scgpt_fine_tuned',
  'temp_dir': 'tmp'
}
## VIASH END


sys.path.append(meta["resources_dir"])
from functions import prepare_data, prepare_dataloader, train, test, evaluate


### Load input data and model
print("Reading input files", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])

if input_train.uns["dataset_organism"] != "homo_sapiens":
  raise ValueError(
    f"scGPT can only be used with human data "
    f"(dataset_organism is \"{input_train.uns['dataset_organism']}\")"
  )

if par["model"] is None:
  print(f"\n>>> Downloading scGPT model...", flush=True)
  model_drive_id = {"scGPT_human": "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"}
  drive_path = (
    f"https://drive.google.com/drive/folders/{model_drive_id['scGPT_human']}"
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
      raise ValueError(f"The 'model' argument should be a directory a .zip file or a .tar.gz file")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: '{device}'", flush=True)


### Settings

scgpt.utils.set_seed(0)

# Recommended hyperparameter setup for cell-type annotation tasks
hyperparameters = dict(
  mask_ratio=0.0,
  epochs=10,
  n_bins=51,
  MVC=False, # Masked value prediction for cell embedding
  ecs_threshold=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
  dab_weight=0.0,
  lr=1e-4,
  batch_size=32,
  dropout=0.2,  # dropout probability
  schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
  fast_transformer=True,
  pre_norm=False,
  amp=True,  # Automatic Mixed Precision
  include_zero_gene = False,
  freeze = False, #freeze
  DSBN = False,  # Domain-spec batchnorm
)

# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = ["<pad>", "<cls>", "<eoc>"]
mask_value = -1
pad_value = -2
max_seq_len = 3001

# settings for training
training_settings = dict(
  MLM = False,  # whether to use masked language modeling
  CLS = True,  # celltype classification objective
  ADV = False,  # Adversarial training for batch correction
  CCE = False,  # Contrastive cell embedding objective
  ECS = hyperparameters["ecs_threshold"] > 0,  # Elastic cell similarity objective
  DAB = False,  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
  adv_E_delay_epochs = 0,  # delay adversarial training on encoder for a few epochs
  adv_D_delay_epochs = 0,
  log_interval = 100  # logging interval
)

explicit_zero_prob = training_settings["MLM"] and hyperparameters["include_zero_gene"]  # whether explicit bernoulli for zeros
per_seq_batch_sample = False

# settings for optimizer
schedule_interval = 1

DAB_separate_optim = True if training_settings["DAB"] > 1 else False

logger = scgpt.logger
scgpt.utils.add_file_handler(logger, "run.log")


### Load and pre-process data

input_train.obs["celltype"] = input_train.obs["label"].astype("category")
data_is_raw = False
filter_gene_by_counts = False

# Make the batch category column
num_types = len(input_train.obs["celltype"].unique())
id2type = dict(enumerate(input_train.obs["celltype"].astype("category").cat.categories))
input_train.obs["celltype_id"] = input_train.obs["celltype"].astype("category").cat.codes.values

model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"
vocab = scgpt.tokenizer.gene_tokenizer.GeneVocab.from_file(vocab_file)
for token in special_tokens:
  if token not in vocab:
    vocab.append_token(token)

input_train.var["id_in_vocab"] = [
  1 if gene in vocab else -1 for gene in input_train.var["feature_name"]
]    
input_train = input_train[:, input_train.var["id_in_vocab"] >= 0]

input_test.var["id_in_vocab"] = [
  1 if gene in vocab else -1 for gene in input_test.var["feature_name"]
]    
input_test = input_test[:, input_train.var["id_in_vocab"] >= 0]

gene_ids_in_vocab = np.array(input_train.var["id_in_vocab"]) + np.array(input_test.var["id_in_vocab"])

logger.info(
  f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
  f"in vocabulary of size {len(vocab)}."
)

# Pre-trained model
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
  filter_gene_by_counts=filter_gene_by_counts, 
  filter_cell_by_counts=False, 
  normalize_total=1e4,  # whether to normalize the raw data and to what sum
  result_normed_key="X_normed",  # the key in input_train.layers to store the normalized data
  log1p=data_is_raw,  # whether to log1p the normalized data
  result_log1p_key="X_log1p",
  subset_hvg=False,  # whether to subset the raw data to highly variable genes
  hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
  binning=hyperparameters["n_bins"],  # whether to bin the raw data and to what number of bins
  result_binned_key="X_binned",  # the key in input_train.layers to store the binned data
)

preprocessor(input_train, batch_key=None)
preprocessor(input_test, batch_key=None)

all_counts = (
  input_train.layers["X_binned"].A
  if issparse(input_train.layers["X_binned"])
  else input_train.layers["X_binned"]
)
genes = input_train.var["feature_name"].tolist()

celltypes_labels = input_train.obs["celltype_id"].tolist()  # make sure count from 0
celltypes_labels = np.array(celltypes_labels)

batch_ids = np.zeros(len(input_train.obs), dtype=int)
num_batch_types = len(set(batch_ids))

(
  train_data,
  valid_data,
  train_celltype_labels,
  valid_celltype_labels,
  train_batch_labels,
  valid_batch_labels,
) = train_test_split(
  all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

if model_dir is None:
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
  append_cls=True,  # append <cls> token at the beginning
  include_zero_gene=hyperparameters["include_zero_gene"],
)
tokenized_valid = scgpt.tokenizer.tokenize_and_pad_batch(
  valid_data,
  gene_ids,
  max_len=max_seq_len,
  vocab=vocab,
  pad_token=pad_token,
  pad_value=pad_value,
  append_cls=True,
  include_zero_gene=hyperparameters["include_zero_gene"],
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
  nlayers_cls=n_layers_cls,
  n_cls=num_types if training_settings["CLS"] else 1,
  vocab=vocab,
  dropout=hyperparameters["dropout"],
  pad_token=pad_token,
  pad_value=pad_value,
  do_mvc=hyperparameters["MVC"],
  do_dab=training_settings["DAB"],
  use_batch_labels=False,
  num_batch_labels=num_batch_types,
  domain_spec_batchnorm=hyperparameters["DSBN"],
  input_emb_style="continuous",
  n_input_bins=hyperparameters["n_bins"],
  cell_emb_style="cls",
  mvc_decoder_style="inner product",
  ecs_threshold=hyperparameters["ecs_threshold"],
  explicit_zero_prob=explicit_zero_prob,
  use_fast_transformer=hyperparameters["fast_transformer"],
  fast_transformer_backend="flash", # "linear" or "flash"
  pre_norm=hyperparameters["pre_norm"],
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

# Freeze all pre-decoder weights
for name, para in model.named_parameters():  
  if hyperparameters["freeze"] and "encoder" in name and "transformer_encoder" not in name:
    para.requires_grad = False

post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")

model.to(device)

criterion = scgpt.loss.masked_mse_loss
criterion_cls = torch.nn.CrossEntropyLoss()
criterion_dab = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"], eps=1e-4 if hyperparameters["amp"] else 1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=hyperparameters["schedule_ratio"])

scaler = torch.cuda.amp.GradScaler(enabled=hyperparameters["amp"])


### Finetune scGPT with task-specific objectives

best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None

for epoch in range(1, hyperparameters["epochs"] + 1):
  epoch_start_time = time.time()
  train_data_pt, valid_data_pt = prepare_data(
    tokenized_train, 
    tokenized_valid, 
    train_batch_labels,
    valid_batch_labels,
    train_celltype_labels,
    valid_celltype_labels,
    hyperparameters["mask_ratio"], 
    mask_value, 
    pad_value, 
    epoch,
    per_seq_batch_sample
  )
  train_loader = prepare_dataloader(
    train_data_pt,
    batch_size=hyperparameters["batch_size"],
    shuffle=False,
    intra_domain_shuffle=True,
    drop_last=False,
    per_seq_batch_sample=per_seq_batch_sample
  )
  valid_loader = prepare_dataloader(
    valid_data_pt,
    batch_size=hyperparameters["batch_size"],
    shuffle=False,
    intra_domain_shuffle=False,
    drop_last=False,
    per_seq_batch_sample=per_seq_batch_sample
  )

  train(
    model=model, 
    train_loader=train_loader,
    device=device,
    vocab=vocab,
    pad_token=pad_token,
    hyperparameters=hyperparameters,
    training_settings=training_settings,
    mask_value=mask_value,
    explicit_zero_prob=explicit_zero_prob,
    criterion=criterion,
    criterion_cls=criterion_cls,
    criterion_dab=criterion_dab, 
    criterion_adv=None,    
    scaler=scaler,
    optimizer=optimizer,
    discriminator=None,
    epoch=epoch, 
    optimizer_D=None, 
    optimizer_E=None,
    scheduler=scheduler,
  )

  val_loss, val_err = evaluate(
    model, 
    valid_loader,
    device,
    vocab,
    pad_token,
    hyperparameters,
    training_settings,
    criterion_cls,
    criterion_dab,
    return_raw=False
  )

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

predictions = test(
  best_model, 
  input_test,
  hyperparameters,
  gene_ids,
  max_seq_len,
  vocab,
  pad_token,
  pad_value,
  mask_value, 
  device,
  training_settings,
  criterion_cls,
  criterion_dab,
)

input_test.obs['label_pred'] = [id2type[p] for p in predictions]


### Write output
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
