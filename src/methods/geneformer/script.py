import os
import sys
from tempfile import TemporaryDirectory
import anndata as ad
from geneformer import (
    Classifier,
    TranscriptomeTokenizer,
    DataCollatorForCellClassification,
)
from huggingface_hub import hf_hub_download
import numpy as np
import datasets
import pickle
from transformers import BertForSequenceClassification, Trainer

## VIASH START
par = {
    "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
    "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
    "output": "output.h5ad",
    "model": "Geneformer-V2-316M",
}
meta = {"name": "geneformer"}
## VIASH END

n_processors = os.cpu_count()

sys.path.append(meta["resources_dir"])
from exit_codes import exit_non_applicable

print(">>> Reading input files", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])

if input_train.uns["dataset_organism"] != "homo_sapiens":
    exit_non_applicable(
        f"Geneformer can only be used with human data "
        f"(dataset_organism == '{input_train.uns['dataset_organism']}')"
    )

# check whether genes are ensembl ids
input_train.var_names = input_train.var["feature_id"]
input_test.var_names = input_test.var["feature_id"]
is_ensembl = all(var_name.startswith("ENSG") for var_name in input_train.var_names)
if not is_ensembl:
    exit_non_applicable(
        f"Geneformer requires input_train.var_names to contain ENSEMBL gene ids"
    )

print(f">>> Getting settings for model '{par['model']}'...", flush=True)

# Parse model details based on new V2 naming scheme
if par["model"] == "Geneformer-V1-10M":
    model_details = {
        "dataset": "30M",
        "input_size": 2048,
        "version": "V1"
    }
    dictionaries_subfolder = "geneformer/gene_dictionaries_30m"
    model_dataset_suffix = "30M"
elif par["model"] == "Geneformer-V2-104M":
    model_details = {
        "dataset": "104M", 
        "input_size": 4096,
        "version": "V2"
    }
    dictionaries_subfolder = "geneformer"
    model_dataset_suffix = "104M"
elif par["model"] == "Geneformer-V2-316M":
    # Note: V2 models use 104M dictionaries even for 316M model
    model_details = {
        "dataset": "104M",
        "input_size": 4096,
        "version": "V2"
    }
    dictionaries_subfolder = "geneformer" 
    model_dataset_suffix = "104M"
else:
    raise ValueError(f"Invalid model: {par['model']}")

print(model_details, flush=True)

print(">>> Getting model dictionary files...", flush=True)
print(f"Dictionaries subfolder: '{dictionaries_subfolder}'")

dictionary_files = {
    "ensembl_mapping": hf_hub_download(
        repo_id="ctheodoris/Geneformer",
        subfolder=dictionaries_subfolder,
        filename=f"ensembl_mapping_dict_gc{model_dataset_suffix}.pkl",
    ),
    "gene_median": hf_hub_download(
        repo_id="ctheodoris/Geneformer",
        subfolder=dictionaries_subfolder,
        filename=f"gene_median_dictionary_gc{model_dataset_suffix}.pkl",
    ),
    "gene_name_id": hf_hub_download(
        repo_id="ctheodoris/Geneformer",
        subfolder=dictionaries_subfolder,
        filename=f"gene_name_id_dict_gc{model_dataset_suffix}.pkl",
    ),
    "token": hf_hub_download(
        repo_id="ctheodoris/Geneformer",
        subfolder=dictionaries_subfolder,
        filename=f"token_dictionary_gc{model_dataset_suffix}.pkl",
    ),
}

print(">>> Creating working directory...", flush=True)
work_dir = TemporaryDirectory()

input_train_dir = os.path.join(work_dir.name, "input_train")
os.makedirs(input_train_dir)
tokenized_train_dir = os.path.join(work_dir.name, "tokenized_train")
os.makedirs(tokenized_train_dir)
classifier_train_dir = os.path.join(work_dir.name, "classifier_train")
os.makedirs(classifier_train_dir)
classifier_fine_tuned_dir = os.path.join(work_dir.name, "classifier_fine_tuned")
os.makedirs(classifier_fine_tuned_dir)
input_test_dir = os.path.join(work_dir.name, "input_test")
os.makedirs(input_test_dir)
tokenized_test_dir = os.path.join(work_dir.name, "tokenized_test")
os.makedirs(tokenized_test_dir)

print(f"Working directory: '{work_dir.name}'", flush=True)

print(f">>> Getting model files for model '{par['model']}'...", flush=True)
model_files = {
    "model": hf_hub_download(
        repo_id="ctheodoris/Geneformer",
        subfolder=par["model"],
        filename="model.safetensors",
    ),
    "config": hf_hub_download(
        repo_id="ctheodoris/Geneformer",
        subfolder=par["model"],
        filename="config.json",
    ),
}
model_dir = os.path.dirname(model_files["model"])

print(">>> Preparing input data...", flush=True)
input_train.X = input_train.layers["counts"]
input_train.var["ensembl_id"] = input_train.var["feature_id"]
input_train.obs["n_counts"] = input_train.layers["counts"].sum(axis=1)
input_train.obs["celltype"] = input_train.obs["label"]
num_types = len(input_train.obs["celltype"].unique())
input_train.write_h5ad(os.path.join(input_train_dir, "input_train.h5ad"))

input_test.X = input_test.layers["counts"]
input_test.var["ensembl_id"] = input_test.var["feature_id"]
input_test.obs["n_counts"] = input_test.layers["counts"].sum(axis=1)
input_test.write_h5ad(os.path.join(input_test_dir, "input_test.h5ad"))

def tryParallelFunction(fun, label):
    try:
        fun(nproc=n_processors)
    except RuntimeError as e:
        # retry with nproc=1 if error message contains "One of the subprocesses has abruptly died"
        if "subprocess" in str(e) and "died" in str(e):
            print(f"{label} failed. Error message: {e}", flush=True)
            print(f"Retrying with nproc=1", flush=True)
            fun(nproc=1)
        else:
            raise e

print(">>> Tokenizing train data...", flush=True)
special_token = model_details["version"] == "V2"
print(f"Input size: {model_details['input_size']}, Special token: {special_token}")

def tokenize_train(nproc):
    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={"celltype": "celltype"},
        nproc=nproc,
        model_input_size=model_details["input_size"],
        special_token=special_token,
        gene_median_file=dictionary_files["gene_median"],
        token_dictionary_file=dictionary_files["token"],
        gene_mapping_file=dictionary_files["ensembl_mapping"],
    )
    tokenizer.tokenize_data(
        input_train_dir, tokenized_train_dir, "tokenized", file_format="h5ad"
    )
    return tokenizer


tokenizer = tryParallelFunction(tokenize_train, "Tokenizing train data")

print(">>> Tokenizing test data...", flush=True)
special_token = model_details["version"] == "V2"
print(f"Input size: {model_details['input_size']}, Special token: {special_token}")

def tokenize_test(nproc):
    tokenizer = TranscriptomeTokenizer(
        model_input_size=model_details["input_size"],
        special_token=special_token,
        gene_median_file=dictionary_files["gene_median"],
        token_dictionary_file=dictionary_files["token"],
        gene_mapping_file=dictionary_files["ensembl_mapping"],
        nproc=nproc,
    )
    tokenizer.tokenize_data(
        input_test_dir, tokenized_test_dir, "tokenized", file_format="h5ad"
    )
    return tokenizer


tokenizer = tryParallelFunction(tokenize_test, "Tokenizing test data")
print(">>> Fine-tuning pre-trained geneformer model...", flush=True)

def train_classifier(nproc):
    training_args={}
    if par["num_train_epochs"]:
        training_args["num_train_epochs"] = par["num_train_epochs"]
    if par["warmup_steps"]:
        training_args["warmup_steps"] = par["warmup_steps"]
    
    cc = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "celltype", "states": "all"},
        nproc=nproc,
        token_dictionary_file=dictionary_files["token"],
        num_crossval_splits=1,
        split_sizes={"train": 0.9, "valid": 0.1, "test": 0.0},
        training_args=training_args
    )

    cc.prepare_data(
        input_data_file=os.path.join(tokenized_train_dir, "tokenized.dataset"),
        output_directory=classifier_train_dir,
        output_prefix="classifier",
        max_trials=par["max_trials"],
    )

    train_data = datasets.load_from_disk(
        classifier_train_dir + "/classifier_labeled.dataset"
    )

    cc.train_classifier(
        model_directory=model_dir,
        num_classes=num_types,
        train_data=train_data,
        eval_data=None,
        output_directory=classifier_fine_tuned_dir,
        predict=False,
    )

    return cc


cc = tryParallelFunction(
    train_classifier,
    "Fine-tuning pre-trained geneformer model",
)

print(">>> Generating predictions...", flush=True)

# dictionary mapping labels from classifier to cell types
with open(f"{classifier_train_dir}/classifier_id_class_dict.pkl", "rb") as f:
    id_class_dict = pickle.load(f)

with open(dictionary_files["token"], "rb") as f:
    token_dict = pickle.load(f)

# Load fine-tuned model
model = BertForSequenceClassification.from_pretrained(classifier_fine_tuned_dir)

test_data = datasets.load_from_disk(tokenized_test_dir + "/tokenized.dataset")
test_data = test_data.add_column("label", [0] * len(test_data))

# Get predictions
trainer = Trainer(
    model=model,
    data_collator=DataCollatorForCellClassification(token_dictionary=token_dict),
)
predictions = trainer.predict(test_data)

# Select the most likely cell type based on the probability vector from the predictions of each cell
predicted_label_ids = np.argmax(predictions.predictions, axis=1)
predicted_logits = [
    predictions.predictions[i][predicted_label_ids[i]]
    for i in range(len(predicted_label_ids))
]
input_test.obs["label_pred"] = [id_class_dict[p] for p in predicted_label_ids]

print(">>> Write output AnnData to file", flush=True)
output = ad.AnnData(
    obs=input_test.obs[["label_pred"]],
    uns={
        "method_id": meta["name"],
        "dataset_id": input_test.uns["dataset_id"],
        "normalization_id": input_test.uns["normalization_id"],
    },
)
output.write_h5ad(par["output"], compression="gzip")
