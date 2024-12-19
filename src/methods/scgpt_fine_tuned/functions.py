import os
import time
import warnings
import numpy as np
import scgpt
from scipy.sparse import issparse
import torch
from torch.utils.data import Dataset, DataLoader


def prepare_data(
    tokenized_train, 
    tokenized_valid, 
    train_batch_labels,
    valid_batch_labels,
    train_celltype_labels,
    valid_celltype_labels,
    mask_ratio, 
    mask_value, 
    pad_value, 
    epoch, 
    sort_seq_batch=False
):
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

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]
        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

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

# dataset
class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt,
    batch_size,
    shuffle=False,
    intra_domain_shuffle=False,
    drop_last=False,
    num_workers=0,
    per_seq_batch_sample=False
):
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
                dataset=dataset,
                batch_sampler=scgpt.SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


def train(
    model, 
    loader,
    device,
    vocab,
    pad_token,
    hyperparameters,
    training_settings,
    mask_value,
    explicit_zero_prob,
    criterion,
    criterion_cls,
    criterion_dab,
    criterion_adv,
    scaler,
    optimizer,
    discriminator,
    epoch,
    optimizer_D, 
    optimizer_E,
    scheduler,
):
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
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
        with torch.cuda.amp.autocast(enabled=hyperparameters["amp"]):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if hyperparameters["DSBN"] else None,
                CLS=training_settings["CLS"],
                CCE=training_settings["CCE"],
                MVC=hyperparameters["MVC"],
                ECS=training_settings["ECS"],
                do_sample=False,
            )

            masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss = 0.0
            metrics_to_log = {}
            if training_settings["MLM"]:
                loss_mse = criterion(output_dict["mlm_output"], target_values, masked_positions)
                loss = loss + loss_mse
                metrics_to_log = {"train/mse": loss_mse.item()}
            if explicit_zero_prob:
                loss_zero_log_prob = scgpt.loss.criterion_neg_log_bernoulli(
                   output_dict["mlm_zero_probs"], 
                   target_values, 
                   masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if training_settings["CLS"]:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})
                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
            if training_settings["CCE"]:
                loss_cce = 10 * output_dict["loss_cce"]
                loss = loss + loss_cce
                metrics_to_log.update({"train/cce": loss_cce.item()})
            if hyperparameters["MVC"]:
                loss_mvc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_mvc
                metrics_to_log.update({"train/mvc": loss_mvc.item()})
            if hyperparameters["MVC"] and explicit_zero_prob:
                loss_mvc_zero_log_prob = scgpt.loss.criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_mvc_zero_log_prob
                metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
            if training_settings["ECS"]:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            if training_settings["DAB"]:
                # try weighting and separate optimizer
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + hyperparameters["dab_weight"] * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

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
                    scgpt.logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
            scaler.step(optimizer)
            scaler.update()

            if training_settings["ADV"]:
                # rerun the model for adversarial training
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if hyperparameters["DSBN"] else None,
                    CLS=training_settings["CLS"],
                    CCE=training_settings["CCE"],
                    MVC=hyperparameters["MVC"],
                    ECS=training_settings["ECS"],
                    do_sample=False,
                )

                # TRAINING DISCRIMINATOR
                loss_adv_D = criterion_adv(
                    discriminator(output_dict["cell_emb"].detach()), batch_labels
                )
                if epoch > training_settings["adv_D_delay_epochs"]:
                    discriminator.zero_grad()
                    loss_adv_D.backward()
                    optimizer_D.step()

                # TRAINING ENCODER
                loss_adv_E = -criterion_adv(
                    discriminator(output_dict["cell_emb"]), batch_labels
                )
                # NOTE: the loss is negative here because we want to maximize
                # the cross_entropy_loss, in other words, disguise against the discriminator
                if epoch > training_settings["adv_E_delay_epochs"]:
                    model.zero_grad()
                    discriminator.zero_grad()
                    loss_adv_E.backward()
                    optimizer_E.step()

        total_loss += loss.item()
        total_mse += loss_mse.item() if training_settings["MLM"] else 0.0
        total_cls += loss_cls.item() if training_settings["CLS"] else 0.0
        total_cce += loss_cce.item() if training_settings["CCE"] else 0.0
        total_mvc += loss_mvc.item() if hyperparameters["MVC"] else 0.0
        total_ecs += loss_ecs.item() if training_settings["ECS"] else 0.0
        total_dab += loss_dab.item() if training_settings["DAB"] else 0.0
        total_adv_E += loss_adv_E.item() if training_settings["ADV"] else 0.0
        total_adv_D += loss_adv_D.item() if training_settings["ADV"] else 0.0
        total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
        total_mvc_zero_log_prob += (
            loss_mvc_zero_log_prob.item() if hyperparameters["MVC"] and explicit_zero_prob else 0.0
        )
        total_error += error_rate
        
        if batch % training_settings["log_interval"] == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / training_settings["log_interval"]
            cur_loss = total_loss / training_settings["log_interval"]
            cur_mse = total_mse / training_settings["log_interval"]
            cur_cls = total_cls / training_settings["log_interval"] if training_settings["CLS"] else 0.0
            cur_cce = total_cce / training_settings["log_interval"] if training_settings["CCE"] else 0.0
            cur_mvc = total_mvc / training_settings["log_interval"] if hyperparameters["MVC"] else 0.0
            cur_ecs = total_ecs / training_settings["log_interval"] if training_settings["ECS"] else 0.0
            cur_dab = total_dab / training_settings["log_interval"] if training_settings["DAB"] else 0.0
            cur_adv_E = total_adv_E / training_settings["log_interval"] if training_settings["ADV"] else 0.0
            cur_adv_D = total_adv_D / training_settings["log_interval"] if training_settings["ADV"] else 0.0
            cur_zero_log_prob = (
                total_zero_log_prob / training_settings["log_interval"] if explicit_zero_prob else 0.0
            )
            cur_mvc_zero_log_prob = (
                total_mvc_zero_log_prob / training_settings["log_interval"]
                if hyperparameters["MVC"] and explicit_zero_prob
                else 0.0
            )
            cur_error = total_error / training_settings["log_interval"]
            # ppl = math.exp(cur_loss)
            scgpt.logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if training_settings["MLM"] else "")
                + (f"cls {cur_cls:5.2f} | " if training_settings["CLS"] else "")
                + (f"err {cur_error:5.2f} | " if training_settings["CLS"] else "")
                + (f"cce {cur_cce:5.2f} |" if training_settings["CCE"] else "")
                + (f"mvc {cur_mvc:5.2f} |" if hyperparameters["MVC"] else "")
                + (f"ecs {cur_ecs:5.2f} |" if training_settings["ECS"] else "")
                + (f"dab {cur_dab:5.2f} |" if training_settings["DAB"] else "")
                + (f"adv_E {cur_adv_E:5.2f} |" if training_settings["ADV"] else "")
                + (f"adv_D {cur_adv_D:5.2f} |" if training_settings["ADV"] else "")
                + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "")
                + (
                f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                if hyperparameters["MVC"] and explicit_zero_prob
                else ""
                )
            )
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            total_mvc = 0
            total_ecs = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_zero_log_prob = 0
            total_mvc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()


def evaluate(
    model,
    loader,
    device,
    vocab,
    pad_token,
    hyperparameters,
    training_settings,
    criterion_cls,
    criterion_dab,
    return_raw
):
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            if not return_raw:
                celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=hyperparameters["amp"]):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if hyperparameters["DSBN"] else None,
                    CLS=training_settings["CLS"],  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
                output_values = output_dict["cls_output"]
                print(output_values)
                if not return_raw:
                    loss = criterion_cls(output_values, celltype_labels)
                    if training_settings["DAB"]:
                        loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
        
            if return_raw:
                preds = output_values.argmax(1).cpu().numpy()
                predictions.append(preds)
            else:
                total_loss += loss.item() * len(input_gene_ids)
                accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_dab += loss_dab.item() * len(input_gene_ids) if training_settings["DAB"] else 0.0
                total_num += len(input_gene_ids)

    if return_raw:
      return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_error / total_num


def test(
    model, 
    adata,
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
):
    all_counts = (
        adata.layers["X_binned"].A
        if issparse(adata.layers["X_binned"])
        else adata.layers["X_binned"]
    )

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_test = scgpt.tokenizer.tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=hyperparameters["include_zero_gene"],
    )

    input_values_test = scgpt.tokenizer.random_mask_value(
        tokenized_test["values"],
        mask_ratio=hyperparameters["mask_ratio"],
        mask_value=mask_value,
        pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
    }

    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=hyperparameters["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), hyperparameters["batch_size"] // 2),
        pin_memory=True,
    )

    model.eval()
    predictions = evaluate(
        model,
        test_loader,
        device,
        vocab,
        pad_token,
        hyperparameters,
        training_settings,
        criterion_cls,
        criterion_dab,
        return_raw=True
    )

    return predictions
