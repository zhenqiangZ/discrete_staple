"""训练两个 transformer-based 预测模型: AMP 活性 & 毒性.

AMP 模型用 data/peptide_data.csv (label=1 有活性).
毒性模型用 toxdata/ 下多源数据 (label=1 有毒).

架构: Embedding → TransformerEncoder × 2 → mean pooling → Linear(1)
输出: checkpoints/amp_classifier_best.pt, checkpoints/toxicity_best.pt
"""

import os, csv
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ================================================================
# 氨基酸映射 (与 task.py 一致)
# ================================================================
AMINO_ACIDS = {aa: i + 1 for i, aa in enumerate(
    ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
     'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
     'S5', 'R8'])}
PAD_TOKEN = 0
VOCAB_SIZE = 23

def tokenize(seq: str) -> torch.Tensor:
    tokens = []
    i = 0
    while i < len(seq):
        if i + 2 <= len(seq) and seq[i:i+2] in AMINO_ACIDS:
            tokens.append(AMINO_ACIDS[seq[i:i+2]])
            i += 2
        else:
            tokens.append(AMINO_ACIDS.get(seq[i], PAD_TOKEN))
            i += 1
    return torch.tensor(tokens, dtype=torch.long)

def pad_sequence(seq: torch.Tensor, max_len: int) -> torch.Tensor:
    if len(seq) >= max_len:
        return seq[:max_len]
    return F.pad(seq, (0, max_len - len(seq)), value=PAD_TOKEN)


# ================================================================
# 模型
# ================================================================

class TransformerClassifier(nn.Module):
    """TransformerEncoder 二分类器.

    对序列做 mean pooling, 输出 logit.
    """

    def __init__(self, vocab_size: int = 23, hidden_dim: int = 256,
                 num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.classifier(h)


# ================================================================
# AUC
# ================================================================

def compute_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranked = sorted(zip(scores.tolist(), labels.tolist()), key=lambda x: x[0])
    sum_pos_rank = sum(rank for rank, (_, label) in enumerate(ranked, 1) if label == 1)
    return (sum_pos_rank - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ================================================================
# AMP 数据加载
# ================================================================

def load_amp_data(csv_path: str = 'data/peptide_data.csv',
                  max_length: int = 50, val_split: float = 0.2,
                  test_split: float = 0.1, seed: int = 42):
    seqs, labels_list = [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seqs.append(row['sequence'].strip())
            labels_list.append(int(row['label']))

    tokens = torch.stack([pad_sequence(tokenize(s), max_length) for s in seqs])
    labels_t = torch.tensor(labels_list, dtype=torch.float32)
    n = len(tokens)

    all_idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    test_idx = all_idx[:n_test]
    val_idx = all_idx[n_test:n_test + n_val]
    train_idx = all_idx[n_test + n_val:]

    train_loader = DataLoader(
        TensorDataset(tokens[train_idx], labels_t[train_idx].long()),
        batch_size=64, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(tokens[val_idx], labels_t[val_idx].long()),
        batch_size=256, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(tokens[test_idx], labels_t[test_idx].long()),
        batch_size=256, shuffle=False)

    n_pos = sum(labels_list)
    n_neg = n - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])
    print(f"AMP data: {n} seqs | pos={n_pos}, neg={n_neg} | "
          f"pos_weight={pos_weight.item():.2f}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    return train_loader, val_loader, test_loader, pos_weight


# ================================================================
# 毒性数据加载 (复用 tox_pred.py 逻辑)
# ================================================================

VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

def _read_csv(path: str, label: int) -> List[Tuple[str, int]]:
    import pandas as pd
    df = pd.read_csv(path, header=None)
    seqs = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
    return [(s, label) for s in seqs if s]

def _read_fasta_like(path: str, pos_label: str = None, neg_label: str = None):
    with open(path, 'r') as f:
        lines = f.readlines()
    result = []
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        header = lines[i].strip()
        seq = lines[i + 1].strip().upper()
        if not seq:
            continue
        if pos_label is not None and pos_label in header:
            result.append((seq, 1))
        elif neg_label is not None and neg_label in header:
            result.append((seq, 0))
    return result

def _read_captp(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
    result = []
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        header = lines[i]
        seq = lines[i + 1].strip().upper()
        if not seq:
            continue
        label_char = header[-2]
        if label_char == '1':
            result.append((seq, 1))
        elif label_char == '0':
            result.append((seq, 0))
    return result

def _read_toxtel(path: str):
    return _read_fasta_like(path, pos_label='>pos', neg_label='>neg')

def _read_btl(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    seq_col = 'sequence' if 'sequence' in df.columns else df.columns[1]
    label_col = 'label' if 'label' in df.columns else df.columns[2]
    result = []
    for _, row in df.iterrows():
        seq = str(row[seq_col]).strip().upper()
        label = int(row[label_col])
        if seq:
            result.append((seq, label))
    return result

def aggregate_toxicity(toxdata_dir: str = 'toxdata'):
    all_data = []
    all_data.extend(_read_csv(f'{toxdata_dir}/train_pos.csv', 1))
    all_data.extend(_read_csv(f'{toxdata_dir}/train_neg.csv', 0))
    all_data.extend(_read_csv(f'{toxdata_dir}/test_pos.csv', 1))
    all_data.extend(_read_csv(f'{toxdata_dir}/test_neg.csv', 0))
    print(f"  [CSV/toxdb]  {len(all_data)}")

    txp = _read_fasta_like(f'{toxdata_dir}/TXP.txt', pos_label='TXP')
    non_txp = _read_fasta_like(f'{toxdata_dir}/Non-TXP.txt', neg_label='non-toxic')
    all_data.extend(txp)
    all_data.extend(non_txp)
    print(f"  [TXP]       {len(txp)} pos, [Non-TXP] {len(non_txp)} neg")

    for fname in ['train_data.txt', 'test1.txt', 'test2.txt']:
        captp = _read_captp(f'{toxdata_dir}/captp/{fname}')
        all_data.extend(captp)
        print(f"  [CAPTP/{fname}] {len(captp)}")

    toxtel_train = _read_toxtel(f'{toxdata_dir}/toxtel/train.fasta')
    toxtel_test = _read_toxtel(f'{toxdata_dir}/toxtel/test.fasta')
    all_data.extend(toxtel_train)
    all_data.extend(toxtel_test)
    print(f"  [Toxtel]    train {len(toxtel_train)}, test {len(toxtel_test)}")

    btl = _read_btl(f'{toxdata_dir}/toxbtl/peptide.csv')
    all_data.extend(btl)
    print(f"  [BTL]       {len(btl)}")
    return all_data

def filter_and_deduplicate(all_data: List[Tuple[str, int]]):
    print(f"  Raw total: {len(all_data)}")
    valid = []
    illegal_chars = set()
    for seq, label in all_data:
        seq_clean = seq.strip().upper()
        if not seq_clean:
            continue
        bad = set(seq_clean) - VALID_AA
        if bad:
            illegal_chars.update(bad)
            continue
        if len(seq_clean) < 4 or len(seq_clean) > 50:
            continue
        valid.append((seq_clean, label))

    print(f"  After filter: {len(valid)}")
    if illegal_chars:
        print(f"  Illegal chars: {sorted(illegal_chars)}")

    seen = {}
    conflict_seqs = set()
    for seq, label in valid:
        if seq in seen and seen[seq] != label:
            conflict_seqs.add(seq)
        if seq not in seen:
            seen[seq] = label

    deduped = []
    seen_once = {}
    for seq, label in valid:
        if seq in conflict_seqs:
            continue
        if seq in seen_once:
            continue
        seen_once[seq] = label
        deduped.append((seq, label))

    print(f"  After dedup: {len(deduped)} ({len(valid) - len(deduped)} removed)")
    if conflict_seqs:
        print(f"  Conflicts removed: {len(conflict_seqs)}")

    seqs = [s for s, _ in deduped]
    labels = [l for _, l in deduped]
    pos = sum(labels)
    neg = len(labels) - pos
    print(f"  Final: {len(seqs)} ({pos} toxic, {neg} non-toxic)")
    return seqs, labels

def create_toxicity_loaders(seqs, labels, val_split=0.1, test_split=0.1,
                             batch_size=64, max_len=50, seed=42):
    tokens = torch.stack([pad_sequence(tokenize(s), max_len) for s in seqs])
    labels_t = torch.tensor(labels, dtype=torch.long)

    n = len(tokens)
    all_idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    test_idx = all_idx[:n_test]
    val_idx = all_idx[n_test:n_test + n_val]
    train_idx = all_idx[n_test + n_val:]

    train_loader = DataLoader(TensorDataset(tokens[train_idx], labels_t[train_idx]),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(tokens[val_idx], labels_t[val_idx]),
                            batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(tokens[test_idx], labels_t[test_idx]),
                             batch_size=256, shuffle=False)

    n_pos = sum(labels)
    n_neg = n - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  pos_weight: {pos_weight.item():.2f}")
    return train_loader, val_loader, test_loader, pos_weight


# ================================================================
# 训练循环
# ================================================================

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: torch.Tensor,
    vocab_size: int = 23,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_epochs: int = 30,
    lr: float = 2e-4,
    save_path: str = 'checkpoints/model_best.pt',
    name: str = 'Model',
    pretrain_path: str = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    model = TransformerClassifier(vocab_size=vocab_size, hidden_dim=hidden_dim,
                                  num_layers=num_layers).to(device)

    if pretrain_path and os.path.exists(pretrain_path):
        print(f"  Loading pretrain from {pretrain_path}...")
        sd = torch.load(pretrain_path, map_location=device, weights_only=True)
        if sd['embed.weight'].shape[0] != vocab_size:
            new_embed = model.embed.weight.data.clone()
            new_embed[:sd['embed.weight'].shape[0]] = sd['embed.weight']
            sd['embed.weight'] = new_embed
        model.load_state_dict(sd, strict=False)

    pos_weight = pos_weight.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print(f"{name} Training")
    print(f"  hidden_dim={hidden_dim}, layers={num_layers}, params={n_params:,}")
    print(f"  epochs={num_epochs}, lr={lr}")
    print(f"{'='*50}")

    best_val_auc = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device).float()
            logits = model(x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1]
                logits = model(x).squeeze(-1)
                val_logits.append(logits.cpu())
                val_labels.append(y)
        val_logits = torch.cat(val_logits)
        val_labels = torch.cat(val_labels)
        val_probs = torch.sigmoid(val_logits)
        val_auc = compute_auc(val_labels, val_probs)
        val_acc = ((val_probs > 0.5) == val_labels).float().mean().item()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)

        if epoch == 1 or epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}/{num_epochs} | loss={np.mean(train_losses):.4f} "
                  f"| val_auc={val_auc:.4f} | val_acc={val_acc:.2%}")

    print(f"\n  {name} complete | best val AUC={best_val_auc:.4f}")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model


# ================================================================
# 评估
# ================================================================

def evaluate(model, test_loader, name='Test'):
    device = next(model.parameters()).device
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1]
            logits.append(model(x).squeeze(-1).cpu())
            labels.append(y)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    probs = torch.sigmoid(logits)
    auc = compute_auc(labels, probs)
    acc = ((probs > 0.5) == labels).float().mean().item()
    print(f"  {name}: AUC={auc:.4f}, Acc={acc:.2%}, "
          f"Samples={len(labels)} ({labels.sum().item()} pos, "
          f"{len(labels) - labels.sum().item()} neg)")
    return auc, acc


# ================================================================
# 主入口
# ================================================================

def main():
    print("=" * 60)
    print("Training Two Transformer Models: AMP & Toxicity")
    print("=" * 60)

    # --- AMP 模型 ---
    print("\n" + "=" * 50)
    print("MODEL 1: AMP Activity Prediction")
    print("=" * 50)
    print("\n[1] Loading AMP data...")
    train_loader, val_loader, test_loader, pos_weight = load_amp_data()
    print("\n[2] Training AMP classifier...")
    amp_model = train_model(
        train_loader, val_loader, pos_weight,
        save_path='checkpoints/amp_classifier_best.pt',
        name='AMP Classifier')
    print("\n[3] Evaluating AMP classifier...")
    evaluate(amp_model, test_loader, 'AMP Test')

    # --- 毒性模型 ---
    print("\n" + "=" * 50)
    print("MODEL 2: Toxicity Prediction")
    print("=" * 50)
    print("\n[1] Aggregating toxicity data...")
    all_data = aggregate_toxicity()
    print(f"  Total: {len(all_data)}")
    print("\n[2] Filtering & deduplicating...")
    seqs, labels = filter_and_deduplicate(all_data)
    print("\n[3] Creating splits...")
    train_loader, val_loader, test_loader, pos_weight = create_toxicity_loaders(seqs, labels)
    print("\n[4] Training toxicity model (transfer from AMP)...")
    tox_model = train_model(
        train_loader, val_loader, pos_weight,
        save_path='checkpoints/toxicity_best.pt',
        name='Toxicity Classifier',
        pretrain_path='checkpoints/amp_classifier_best.pt')
    print("\n[5] Evaluating toxicity model...")
    evaluate(tox_model, test_loader, 'Toxicity Test')

    print("\n" + "=" * 60)
    print("Done. Models saved to checkpoints/")
    print("  - checkpoints/amp_classifier_best.pt")
    print("  - checkpoints/toxicity_best.pt")
    print("=" * 60)


if __name__ == '__main__':
    main()
