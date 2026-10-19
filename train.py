import os
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from config import (
    PAIRED_CSV, CKPT_DIR, LOG_DIR,
    EPOCHS, BATCH_SIZE, LR, MASK_RATE_MAX,
)
from data.dataset import create_loaders
from model.network import StapleDiffusionNetwork
from model.diffusion import random_mask, random_mask_rate

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_staple_correct = 0
    total_staple_tokens = 0

    for inp, tgt, cond in loader:
        inp = inp.to(device)
        tgt = tgt.to(device)
        cond = cond.to(device)

        B = inp.size(0)
        mask_rate = random_mask_rate(B, MASK_RATE_MAX)
        
        inp_noisy, mask = random_mask(inp, mask_rate)

        logits = model(inp_noisy, cond)
        
        pred = logits.argmax(dim=-1)

        mask_correct = (pred[mask] == tgt[mask]).float().sum().item()
        total_correct += mask_correct

        is_staple = (tgt == 21) | (tgt == 22)
        staple_mask = mask & is_staple
        total_staple_tokens += staple_mask.sum().item()
        if staple_mask.any():
            staple_correct = (pred[staple_mask] == tgt[staple_mask]).float().sum().item()
            total_staple_correct += staple_correct

        loss = F.cross_entropy(logits[mask], tgt[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()

    staple_acc = total_staple_correct / total_staple_tokens if total_staple_tokens > 0 else 0.0
    return total_loss / total_tokens, total_correct / total_tokens, staple_acc


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_staple_correct = 0
    total_staple_tokens = 0

    with torch.no_grad():
        for inp, tgt, cond in loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            cond = cond.to(device)

            B = inp.size(0)
            mask_rate = random_mask_rate(B, MASK_RATE_MAX)
        
            inp_noisy, mask = random_mask(inp, mask_rate)

            logits = model(inp_noisy, cond)
            loss = F.cross_entropy(logits[mask], tgt[mask])

            pred = logits.argmax(dim=-1)
            correct = (pred[mask] == tgt[mask]).float().sum().item()

            is_staple = (tgt == 21) | (tgt == 22)
            staple_mask = mask & is_staple
            total_staple_tokens += staple_mask.sum().item()
            if staple_mask.any():
                staple_correct = (pred[staple_mask] == tgt[staple_mask]).float().sum().item()
                total_staple_correct += staple_correct

            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
            total_correct += correct

    staple_acc = total_staple_correct / total_staple_tokens if total_staple_tokens > 0 else 0.0
    return total_loss / total_tokens, total_correct / total_tokens, staple_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--data', type=str, default=PAIRED_CSV,
                        help='paired CSV to train on. Point this at a train-only file '
                             '(see TRAINING_PLAN.md) if you want a real held-out eval.')
    parser.add_argument('--out_dir', type=str, default=CKPT_DIR,
                        help='checkpoint output directory (best.pt is written here)')
    args = parser.parse_args()

    epochs = args.epochs or EPOCHS
    batch_size = args.batch_size or BATCH_SIZE
    lr = args.lr or LR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR) if HAS_TB else None

    print(f'Data: {args.data}')
    train_loader, val_loader = create_loaders(args.data, batch_size)

    model = StapleDiffusionNetwork().to(device)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_staple_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_staple_acc = validate(model, val_loader, device)
        elapsed = time.time() - t0

        if writer is not None:
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/val', val_loss, epoch)
            writer.add_scalar('acc/train', train_acc, epoch)
            writer.add_scalar('acc/val', val_acc, epoch)
            writer.add_scalar('staple_acc/train', train_staple_acc, epoch)
            writer.add_scalar('staple_acc/val', val_staple_acc, epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch {epoch:3d} | '
              f'train_loss={train_loss:.4f} val_loss={val_loss:.4f} '
              f'train_acc={train_acc:.4f} val_acc={val_acc:.4f} | '
              f'T_sacc={train_staple_acc:.4f} V_sacc={val_staple_acc:.4f} | {elapsed:.1f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }
            torch.save(ckpt, os.path.join(args.out_dir, 'best.pt'))
            print(f'  -> saved best checkpoint (val_loss={val_loss:.4f})')

    if writer is not None:
        writer.close()
    print(f'Done. Best val_loss={best_val_loss:.4f}')


if __name__ == '__main__':
    main()
