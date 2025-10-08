import argparse
import os
import random
from typing import List, Tuple

import tensorflow as tf


def read_records(input_glob: str) -> List[bytes]:
    files = tf.io.gfile.glob(input_glob)
    if not files:
        raise FileNotFoundError(f"No TFRecord files matched: {input_glob}")
    out: List[bytes] = []
    for raw in tf.data.TFRecordDataset(files):
        out.append(bytes(raw.numpy()))
    return out


def try_get_label(example: tf.train.Example) -> int:
    feature = example.features.feature
    for key in ("image/class/label", "image/label", "label"):
        if key in feature and feature[key].int64_list.value:
            return 1 if int(feature[key].int64_list.value[0]) == 1 else 0
    raise KeyError("Label not found in example (tried image/class/label, image/label, label)")


def split_by_label(records: List[bytes]) -> Tuple[List[bytes], List[bytes]]:
    crystals: List[bytes] = []
    non_crystals: List[bytes] = []
    for raw in records:
        ex = tf.train.Example()
        ex.ParseFromString(raw)
        label = try_get_label(ex)
        (crystals if label == 1 else non_crystals).append(raw)
    return crystals, non_crystals


def balance(crystals: List[bytes], non_crystals: List[bytes], seed: int) -> List[bytes]:
    random.seed(seed)
    k = min(len(crystals), len(non_crystals))
    crystals_b = random.sample(crystals, k=k) if len(crystals) > k else crystals
    non_b = random.sample(non_crystals, k=k) if len(non_crystals) > k else non_crystals
    merged = crystals_b + non_b
    random.shuffle(merged)
    return merged


def write_sharded(records: List[bytes], output_dir: str, prefix: str, shard_size: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    num_shards = max(1, (len(records) + shard_size - 1) // shard_size)
    for i in range(num_shards):
        start = i * shard_size
        end = min((i + 1) * shard_size, len(records))
        shard = os.path.join(output_dir, f"{prefix}-{i:05d}-of-{num_shards:05d}")
        with tf.io.TFRecordWriter(shard) as w:
            for r in records[start:end]:
                w.write(r)
        print(f"Wrote {shard} with {end - start} records")


def main() -> None:
    p = argparse.ArgumentParser(description="Create balanced TFRecords from a TFRecord split")
    p.add_argument("--input_glob", required=True, help="e.g. train/train-* or test/test-*")
    p.add_argument("--output_dir", required=True, help="e.g. train_balanced or test_balanced")
    p.add_argument("--prefix", default="balanced", help="output shard name prefix")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_size", type=int, default=1000)
    args = p.parse_args()

    print(f"Reading: {args.input_glob}")
    records = read_records(args.input_glob)
    print(f"Total records: {len(records):,}")

    crystals, non_crystals = split_by_label(records)
    print(f"Class counts → crystals: {len(crystals):,}, not-crystals: {len(non_crystals):,}")
    if not crystals or not non_crystals:
        raise RuntimeError("One class has zero samples; cannot balance.")

    balanced = balance(crystals, non_crystals, args.seed)
    print(f"Balanced total: {len(balanced):,} (each class: {len(balanced)//2:,})")

    print(f"Writing to: {args.output_dir}")
    write_sharded(balanced, args.output_dir, args.prefix, args.shard_size)
    print("✅ Done")


if __name__ == "__main__":
    main()
