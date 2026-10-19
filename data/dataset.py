import csv
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter
from typing import Tuple

from data.tokenizer import encode, pad
from config import MAX_LEN, BATCH_SIZE


class PairedDataset(Dataset):
    def __init__(self, csv_path: str, max_len: int = MAX_LEN):
        self.max_len = max_len
        self.samples: list[tuple[list[int], list[int], int]] = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                inp = row['input_seq'].strip()
                tgt = row['target_seq'].strip()
                cond = int(row['cond'])
                inp_tok = pad(encode(inp), max_len)
                tgt_tok = pad(encode(tgt), max_len)
                self.samples.append((inp_tok, tgt_tok, cond))

        print(f'PairedDataset: {len(self.samples)} samples')
        cond_counts = Counter(cond for _, _, cond in self.samples)
        for c in sorted(cond_counts):
            print(f'  cond={c}: {cond_counts[c]:>6}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp_tok, tgt_tok, cond = self.samples[idx]
        return (torch.tensor(inp_tok, dtype=torch.long),
                torch.tensor(tgt_tok, dtype=torch.long),
                torch.tensor(cond, dtype=torch.long))


def create_loaders(csv_path: str, batch_size: int = BATCH_SIZE,
                   val_split: float = 0.1) -> Tuple[DataLoader, DataLoader]:
    """Train/val loaders split **by parent sequence**, not by row.

    One parent now yields several stapled variants (cond 2/3/4).  Splitting
    by row would put some variants of a parent in train and the rest in val,
    so the model could score well by memorising that parent — val_loss would
    be optimistic and checkpoint selection would follow it.
    """
    dataset = PairedDataset(csv_path)
    n = len(dataset)

    groups: dict[tuple, list[int]] = {}
    for idx, (inp, _, _) in enumerate(dataset.samples):
        groups.setdefault(tuple(inp), []).append(idx)

    keys = sorted(groups)
    rng = random.Random(42)
    rng.shuffle(keys)

    n_val_groups = max(1, int(len(keys) * val_split)) if len(keys) > 1 else 0
    val_keys = set(keys[:n_val_groups])

    train_idx = [i for k in keys if k not in val_keys for i in groups[k]]
    val_idx = [i for k in keys if k in val_keys for i in groups[k]]

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx),
                            batch_size=batch_size, shuffle=False)
    print(f'Train: {len(train_idx)} ({len(keys) - len(val_keys)} parents), '
          f'Val: {len(val_idx)} ({len(val_keys)} parents)')
    return train_loader, val_loader
