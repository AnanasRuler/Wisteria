"""Dump model embeddings for VEP classification task.

"""

import argparse
import os
from functools import partial
from os import path as osp
from typing import Dict, Iterable, Optional
import caduceus
import enformer_pytorch
import fsspec
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from sklearn import preprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, DefaultDataCollator


from src.dataloaders.utils.rc import string_reverse_complement
from src.utils.train import get_logger

WINDOW_SIZE_BP = 1536
log = get_logger(__name__)


class DNAEmbeddingModel(nn.Module):
    """Wrapper around HF model.

    Args:
        model_name_or_path: str, path to HF model.
    """
    def __init__(
            self,
            model_name_or_path: str,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        # Enformer uses different library for loading
        if "enformer" in model_name_or_path.lower():
            self.backbone = enformer_pytorch.from_pretrained(
                model_name_or_path,
                use_tf_gamma=False,
                use_checkpointing=True
            )
        # NT model is not compatible with AutoModel class
        elif "nucleotide-transformer" in model_name_or_path.lower():
            # NT LM `backbone` is under the `.esm` attribute
            self.backbone = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True).esm
        else:
            self.backbone = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    def forward(self, input_ids):
        """Backbone forward pass to retrieve last_hidden_state."""
        if "enformer" in self.model_name_or_path.lower():
            # Enformer forward pass has different signature
            return self.backbone(input_ids, return_embeddings=True)[1]
        return self.backbone(input_ids, output_hidden_states=True).hidden_states[-1]

class EnformerTokenizer:
    """Enformer tokenizer."""
    # Order is important here! (See: https://github.com/lucidrains/enformer-pytorch?tab=readme-ov-file#usage)
    pad_token = "P"  # Padding token should be a character to avoid issues with tokenization
    encode_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, pad_token: -1}

    @classmethod
    def encode(
            cls, seq: str, max_length: Optional[int] = None, truncation: Optional[bool] = False
    ) -> Iterable[int]:
        """Convert bp to token ids."""
        if max_length is not None:
            assert max_length >= 0, "max_length should be a positive integer."
            if len(seq) < max_length:
                seq = seq + cls.pad_token * (max_length - len(seq))
            elif truncation:
                seq = seq[:max_length]
        return [cls.encode_map[bp] for bp in seq.upper()]

    @classmethod
    def batch_encode_plus(
            cls, seqs: Iterable[str], max_length: Optional[int] = None, truncation: Optional[bool] = False,
            **kwargs,  # ensures compatibility with HF tokenizer-like API
    ) -> Dict[str, Iterable[Iterable[int]]]:
        """Batch encode sequences using HF tokenizer-like API."""
        input_ids = [cls.encode(seq, max_length=max_length, truncation=truncation) for seq in seqs]
        return {"input_ids": input_ids}


def setup_distributed():
    """Set environment variables for distributed runs."""
    dist.init_process_group("nccl")


def cleanup_distributed():
    """Clean up processes from distributed runs."""
    dist.destroy_process_group()


def fsspec_exists(filename):
    """Check if file exists in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


# Processing functions
def recast_chromosome_tissue_dist2TSS(examples):
    """Recast chromosome to int."""
    return {
        "chromosome": -1 if examples["chromosome"] == "X" else int(examples["chromosome"]),
        "tissue": examples["tissue"],
        "distance_to_nearest_tss": examples["distance_to_nearest_tss"]
    }


def tokenize_variants(examples, tokenizer, max_length: int):
    """Tokenize sequence.

    Args:
        examples: (batch of) items from the dataset.
        tokenizer: AutoTokenizer.
        max_length: int.
    Returns:
        dict with values as list of token ids.
    """

    ref_tokenized = tokenizer.batch_encode_plus(
        examples["ref_forward_sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    alt_tokenized = tokenizer.batch_encode_plus(
        examples["alt_forward_sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    ref_rc_tokenized = tokenizer.batch_encode_plus(
        [string_reverse_complement(seq) for seq in examples["ref_forward_sequence"]],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    alt_rc_tokenized = tokenizer.batch_encode_plus(
        [string_reverse_complement(seq) for seq in examples["alt_forward_sequence"]],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )

    return {
        "ref_input_ids": ref_tokenized["input_ids"],
        "alt_input_ids": alt_tokenized["input_ids"],
        "ref_rc_input_ids": ref_rc_tokenized["input_ids"],
        "alt_rc_input_ids": alt_rc_tokenized["input_ids"],
    }


def find_variant_idx(examples):
    """Find token location that differs between reference and variant sequence.

    Args:
        examples: items from the dataset (not batched).
    Returns:
        dict with values index of difference.
    """
    # Guess that variant is at halfway point
    idx = len(examples["ref_input_ids"]) // 2
    if examples["ref_input_ids"][idx] == examples["alt_input_ids"][idx]:
        # If no, loop through sequence and find variant location
        idx = -1
        for i, (ref, alt) in enumerate(zip(examples["ref_input_ids"], examples["alt_input_ids"])):
            if ref != alt:
                idx = i
    # Same as above, but for reverse complement
    rc_idx = len(examples["ref_rc_input_ids"]) // 2 - 1
    if examples["ref_rc_input_ids"][rc_idx] == examples["alt_rc_input_ids"][rc_idx]:
        rc_idx = -1
        for i, (ref, alt) in enumerate(zip(examples["ref_rc_input_ids"], examples["alt_rc_input_ids"])):
            if ref != alt:
                rc_idx = i
    return {"variant_idx": idx, "rc_variant_idx": rc_idx}


def prepare_dataset(args, tokenizer):
    """Prepare or load the tokenized dataset."""
    # Data Preprocessing
    num_tokens = args.seq_len // args.bp_per_token

    # Load data
    cache_dir = osp.join(
        os.getenv("HF_HOME"), "datasets", "InstaDeepAI___genomics-long-range-benchmark",
        "variant_effect_gene_expression", f"seqlen={args.seq_len}"
    )
    if "nucleotide-transformer" in args.model_name_or_path.lower():  # NT uses 6-mers, so tokenization is different
        preprocessed_cache_file = osp.join(cache_dir, "6mer_token_preprocessed")

    elif "enformer" in args.model_name_or_path.lower():
        # Enformer tokenization requires having vocab of just `A,C,G,T,N` (in that order)
        preprocessed_cache_file = osp.join(cache_dir, "enformer_char_token_preprocessed")
    else:
        preprocessed_cache_file = osp.join(cache_dir, "char_token_preprocessed")
    log.warning(f"Cache dir: {cache_dir}")
    log.warning(f"Cache dir preprocessed: {preprocessed_cache_file}")

    if not fsspec_exists(preprocessed_cache_file):
        if dist.get_rank() == 0:
            dataset = load_dataset(
                "InstaDeepAI/genomics-long-range-benchmark",
                task_name="variant_effect_causal_eqtl",
                sequence_length=args.seq_len,
                load_from_cache=False,
            )
            log.warning("Dataset loaded. Cached to disk:")
            log.warning(osp.dirname(list(dataset.cache_files.values())[0][0]["filename"]))
            try:
                del dataset["validation"]  # `validation` split is empty
            except KeyError:
                pass

            # Process data
            dataset = dataset.filter(
                lambda example: example["ref_forward_sequence"].count('N') < 0.005 * args.seq_len,
                desc="Filter N's"
            )
            dataset = dataset.map(
                recast_chromosome_tissue_dist2TSS,
                remove_columns=["chromosome", "tissue", "distance_to_nearest_tss"],
                desc="Recast chromosome"
            )
            dataset = dataset.map(
                partial(tokenize_variants, tokenizer=tokenizer, max_length=num_tokens),
                batch_size=1000,
                batched=True,
                remove_columns=["ref_forward_sequence", "alt_forward_sequence"],
                desc="Tokenize"
            )
            dataset = dataset.map(find_variant_idx, desc="Find variant idx")
            dataset.save_to_disk(preprocessed_cache_file)
    dist.barrier()  # Processes need to wait for dataset to be saved to disk (if not already done)
    dataset = load_from_disk(preprocessed_cache_file)
    log.warning(f"Loaded preprocessed dataset from {preprocessed_cache_file}")
    log.warning(dataset)
    return dataset


def get_backbone_model(args, device):
    """Get the backbone model."""

    model = DNAEmbeddingModel(
        model_name_or_path=args.model_name_or_path,
    )
    model.eval()
    return DDP(model.to(device))


def dump_embeddings(args, dataset, model, device):
    """Dump embeddings to disk with intermediate saving."""

    def extract_embeddings(item_ref, item_alt, variant_idx):
        layer_metrics = {}
        if "enformer" in args.model_name_or_path.lower():
            window_size = WINDOW_SIZE_BP // 128
            variant_idx = torch.ones_like(variant_idx) * item_ref.size(1) // 2
        else:
            window_size = WINDOW_SIZE_BP // args.bp_per_token

        start, end = -window_size // 2, window_size // 2 + 1
        expanded_indices = torch.arange(start, end, device=item_ref.device).unsqueeze(0) + \
                           variant_idx.unsqueeze(1).to(item_ref.device)
        expanded_indices = torch.clamp(expanded_indices, 0, item_ref.size(1) - 1)

        tokens_window_ref = torch.gather(item_ref, 1,
            expanded_indices.unsqueeze(-1).expand(-1, -1, item_ref.size(2))).mean(dim=1)
        tokens_window_alt = torch.gather(item_alt, 1,
            expanded_indices.unsqueeze(-1).expand(-1, -1, item_ref.size(2))).mean(dim=1)
        layer_metrics["concat_avg_ws"] = torch.cat([tokens_window_ref, tokens_window_alt], dim=-1)
        return layer_metrics

    embeds_path = osp.join(args.downstream_save_dir, args.name)
    os.makedirs(embeds_path, exist_ok=True)

    dataloader_params = {
        "batch_size": args.embed_dump_batch_size,
        "collate_fn": DefaultDataCollator(return_tensors="pt"),
        "num_workers": args.num_workers,
        "pin_memory": False,
        "shuffle": False,
        "drop_last": True
    }

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(dataset["test"]["tissue"])
    dataset["train"] = dataset["train"].add_column("tissue_embed", label_encoder.transform(dataset["train"]["tissue"]))
    dataset["test"] = dataset["test"].add_column("tissue_embed", label_encoder.transform(dataset["test"]["tissue"]))

    save_every_batches = 100
    fs = fsspec.filesystem("file")

    def find_last_existing_batch(split_name, rank, total_batches):
        """使用二分查找确定最后一个存在的批次"""
        low, high = 0, total_batches - 1
        last_existing = -1
        
        while low <= high:
            mid = (low + high) // 2
            partial_fname = f"{split_name}_partial_{rank}_{mid}.pt"
            partial_path = osp.join(embeds_path, partial_fname)
            
            if fs.exists(partial_path):
                last_existing = mid
                low = mid + 1  # 检查更大的批次
            else:
                high = mid - 1  # 检查更小的批次
                
        return last_existing

    for split_name, split in dataset.items():
        sampler = DistributedSampler(split, shuffle=dataloader_params["shuffle"], drop_last=dataloader_params["drop_last"])
        dl = DataLoader(split, **dataloader_params, sampler=sampler)
        total_batches = len(dl)
        rank = dist.get_rank()
        
        # 使用二分查找快速确定最后一个存在的批次
        last_existing = find_last_existing_batch(split_name, rank, total_batches)
        
        # 如果所有批次都已存在，则跳过这个split的处理
        if last_existing == total_batches - 1:
            print(f"[RANK {rank}] All batches already exist for {split_name}, skipping...")
            continue

        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), desc=f"[RANK {rank}] Embedding {split_name}",
                                   disable=rank != 0):
            # 跳过已经存在的批次
            if batch_idx <= last_existing:
                continue

            partial_fname = f"{split_name}_partial_{rank}_{batch_idx}.pt"
            partial_path = osp.join(embeds_path, partial_fname)

            storage_dict = {
                "concat_avg_ws": [],
                "rc_concat_avg_ws": [],
                "chromosome": [],
                "labels": [],
                "distance_to_nearest_tss": [],
                "tissue_embed": [],
            }

            for key in ["chromosome", "labels", "distance_to_nearest_tss", "tissue_embed"]:
                storage_dict[key].append(batch[key].to("cpu", non_blocking=True))

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                output_alt = model(batch["alt_input_ids"].to(device))
                output_ref = model(batch["ref_input_ids"].to(device))
                if args.rcps:
                    num_channels = output_alt.size(-1)
                    output_alt_rc = output_alt[..., num_channels // 2:].flip(dims=[1, 2])
                    output_ref_rc = output_ref[..., num_channels // 2:].flip(dims=[1, 2])
                    output_alt = output_alt[..., :num_channels // 2]
                    output_ref = output_ref[..., :num_channels // 2]
                else:
                    output_alt_rc = model(batch["alt_rc_input_ids"].to(device)).flip(dims=[1])
                    output_ref_rc = model(batch["ref_rc_input_ids"].to(device)).flip(dims=[1])

            metrics = extract_embeddings(output_ref, output_alt, batch["variant_idx"])
            for key in metrics:
                storage_dict[key].append(metrics[key].to("cpu", non_blocking=True))

            metrics_rc = extract_embeddings(output_ref_rc, output_alt_rc, batch["variant_idx"])
            for key in metrics_rc:
                storage_dict[f"rc_{key}"].append(metrics_rc[key].to("cpu", non_blocking=True))

            # 保存当前批次
            storage_dict = concat_storage_dict_values(storage_dict)
            with fsspec.open(partial_path, "wb") as f:
                torch.save(storage_dict, f)

            if batch_idx % save_every_batches == 0:
                print(f"[RANK {rank}] Saved intermediate batch {batch_idx}.")

        print(f"[RANK {rank}] Finished all batches for {split_name}")




def combine_embeddings(embeds_path: str):
    """Combine all partial embedding files into one file per split (train/test/val).

    Args:
        embeds_path (str): Path to the directory where partial embedding files are saved.
    """
    import glob
    import torch

    os.makedirs(embeds_path, exist_ok=True)
    print(f"Combining embeddings in {embeds_path}...")

    all_partial_files = glob.glob(osp.join(embeds_path, "*_partial_*.pt"))
    if not all_partial_files:
        print("No partial files found.")
        return

    # Group by split_name (e.g., train/test/val)
    split_file_map = {}
    for f in all_partial_files:
        basename = osp.basename(f)
        split_name = basename.split("_partial_")[0]
        split_file_map.setdefault(split_name, []).append(f)

    for split_name, file_list in split_file_map.items():
        combined_path = osp.join(embeds_path, f"{split_name}_embeds_combined.pt")
        if osp.exists(combined_path):
            print(f"Combined file for {split_name} already exists at {combined_path}, skipping.")
            continue

        print(f"Combining {len(file_list)} files for split '{split_name}'...")

        combined = None
        for file_path in tqdm(sorted(file_list), desc=f"Loading {split_name}"):
            with fsspec.open(file_path, "rb") as f:
                part = torch.load(f, map_location="cpu")
                if combined is None:
                    combined = {k: [v] for k, v in part.items()}
                else:
                    for k, v in part.items():
                        combined[k].append(v)

        # Concatenate each list of tensors into one tensor
        for k in combined:
            combined[k] = torch.cat(combined[k], dim=0)

        with fsspec.open(combined_path, "wb") as f:
            torch.save(combined, f)
        print(f"[DONE] Saved combined file to {combined_path}")





def main(args):
    """Main entry point."""
    # Reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # Init distributed
    log.warning("Initializing distributed...")
    dist.init_process_group("nccl")
    print(f"[RANK {dist.get_rank()}] Distributed initialized: rank {dist.get_rank()}")  # All processes print this
    # Setup device
    device = torch.device(f"cuda:{dist.get_rank()}")
    print(f"[RANK {dist.get_rank()}] Using device: {device}.")  # All processes print this

    # Init tokenizer
    if "enformer" in args.model_name_or_path.lower():
        # Enformer tokenization requires having vocab of just `A,C,G,T,N` (in that order)
        tokenizer = EnformerTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Get dataset
    dist.barrier()
    dataset = prepare_dataset(args, tokenizer)

    # Get model
    dist.barrier()
    model = get_backbone_model(args, device)
    log.warning("Model loaded.")

    # Dump embeddings
    dist.barrier()
    dump_embeddings(args, dataset, model, device)

    # Combine embeddings into single file
    dist.barrier()
    cleanup_distributed()
    

    def is_main_process():
        return not dist.is_initialized() or dist.get_rank() == 0

    # 在你的 main 函数中这样调用
    if is_main_process():
        combine_embeddings(osp.join(args.downstream_save_dir, args.name))


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq_len", type=int, default=131072,
                        help="Sequence length (in bp)..")
    parser.add_argument("--bp_per_token", type=int, default=1,
                        help="Number of base pairs per token.")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--downstream_save_dir", type=str, default="./outputs/downstream/vep_embeddings",
                        help="Directory to save downstream task.")
    parser.add_argument("--name", type=str, default=None, help="Embeddings model name.")
    parser.add_argument("--rcps", default=False, action="store_true", help="Use RCPS.")
    parser.add_argument("--no-rcps", dest="rcps", action="store_false", help="Do not use RCPS.")
    parser.add_argument("--embed_dump_batch_size", type=int, default=1,
                        help="Batch size for embedding dump.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
    opts, _ = parser.parse_known_args()
    log.warning("*** Args ************************")
    for k, v in vars(opts).items():
        log.warning(f"  - {k}: {v}")
    log.warning("******************************\n")

    main(opts)
