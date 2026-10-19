import argparse
import torch
import torch.nn.functional as F

from config import CKPT_DIR, MAX_LEN
from data.tokenizer import encode, pad, decode_pretty
from model.network import StapleDiffusionNetwork
from model.diffusion import random_mask


def get_probs(model, amp_tokens, cond, mask_rate, device):
    B, L = amp_tokens.shape
    mask_rate_t = torch.full((B,), mask_rate)
    inp_noisy, mask = random_mask(amp_tokens, mask_rate_t)
    logits = model(inp_noisy, cond)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    p_s5 = probs[:, :, 21]
    p_r8 = probs[:, :, 22]
    return p_s5, p_r8, probs, log_probs


def build_candidates_cond2(amp_tokens, log_probs):
    B, L = amp_tokens.shape
    is_pad = (amp_tokens == 0)
    candidates = []

    for i in range(B):
        for a in range(L - 4):
            if is_pad[i, a] or is_pad[i, a + 4]:
                continue
            pair = (a, a + 4)
            conf = (log_probs[i, a, 21] + log_probs[i, a + 4, 21]).item()
            candidates.append((i, pair, conf))
    return candidates


def build_candidates_cond3(amp_tokens, log_probs):
    B, L = amp_tokens.shape
    is_pad = (amp_tokens == 0)
    candidates = []

    for i in range(B):
        for a in range(L - 7):
            if is_pad[i, a] or is_pad[i, a + 7]:
                continue
            pair = (a, a + 7)
            conf = (log_probs[i, a, 21] + log_probs[i, a + 7, 22]).item()
            candidates.append((i, pair, conf))
    return candidates


def build_candidates_cond4(amp_tokens, log_probs):
    B, L = amp_tokens.shape
    is_pad = (amp_tokens == 0)
    candidates = []

    for i in range(B):
        valid_pos = (~is_pad[i]).nonzero(as_tuple=True)[0].tolist()
        seen = set()
        for a1 in valid_pos:
            for gap1 in [4, 7]:
                j1 = a1 + gap1
                if j1 >= L or is_pad[i, j1]:
                    continue
                tok1, tok2 = (21, 21) if gap1 == 4 else (21, 22)
                for a2 in valid_pos:
                    if a2 == a1:
                        continue
                    for gap2 in [4, 7]:
                        j2 = a2 + gap2
                        if j2 >= L or is_pad[i, j2]:
                            continue
                        if len({a1, j1, a2, j2}) < 4:
                            continue
                        quad = (a1, j1, a2, j2)
                        key = tuple(sorted([a1, j1, a2, j2]))
                        if key in seen:
                            continue
                        seen.add(key)
                        tok3, tok4 = (21, 21) if gap2 == 4 else (21, 22)
                        conf = (
                            log_probs[i, a1, tok1].item()
                            + log_probs[i, j1, tok2].item()
                            + log_probs[i, a2, tok3].item()
                            + log_probs[i, j2, tok4].item()
                        )
                        candidates.append((i, quad, (tok1, tok2, tok3, tok4), conf))
    return candidates


@torch.no_grad()
def score_candidates(model, amp_tokens, cond_tensor, candidates, cand_type, device):
    if not candidates:
        return []

    if cand_type == 'cond4':
        cands = candidates  # list of (i, quad, config, conf)
        seqs = []
        for i, quad, config, _ in cands:
            seq = amp_tokens[i:i+1].clone()
            pos = [quad[0], quad[1], quad[2], quad[3]]
            tok = [config[0], config[1], config[2], config[3]]
            for p, t in zip(pos, tok):
                seq[0, p] = t
            seqs.append(seq)
        seqs = torch.cat(seqs, dim=0)
        conds = cond_tensor[0].unsqueeze(0).repeat(len(seqs), 1).squeeze(1)
        logits = model(seqs, conds)
        probs = F.softmax(logits, dim=-1)
        for idx, (bi, quad, config, _) in enumerate(cands):
            pos = [quad[0], quad[1], quad[2], quad[3]]
            tok = [config[0], config[1], config[2], config[3]]
            score = sum(probs[idx, p, t].item() for p, t in zip(pos, tok))
            cands[idx] = (bi, quad, config, score)
        return sorted(cands, key=lambda x: x[3], reverse=True)

    cands = candidates  # list of (i, pair, conf)
    seqs = []
    for i, pair, _ in cands:
        seq = amp_tokens[i:i+1].clone()
        a, b = pair
        seq[0, a] = 21
        seq[0, b] = 22 if cond_tensor[0].item() == 3 else 21
        seqs.append(seq)
    seqs = torch.cat(seqs, dim=0)
    conds = cond_tensor[0].unsqueeze(0).repeat(len(seqs), 1).squeeze(1)
    logits = model(seqs, conds)
    probs = F.softmax(logits, dim=-1)
    for idx, (i, pair, _) in enumerate(cands):
        a, b = pair
        tok_a = 21
        tok_b = 22 if cond_tensor[0].item() == 3 else 21
        score = probs[idx, a, tok_a].item() + probs[idx, b, tok_b].item()
        cands[idx] = (i, pair, score)
    return sorted(cands, key=lambda x: x[2], reverse=True)


def main():
    parser = argparse.ArgumentParser(description='基于 AMP 序列设计订书肽（候选排序）')
    parser.add_argument('seq', type=str, help='AMP 氨基酸序列')
    parser.add_argument('--cond', type=int, required=True, choices=[2, 3, 4])
    parser.add_argument('--mask_rate', type=float, default=0.1)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--top_n', type=int, default=3)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    top_k = args.top_k or {2: 5, 3: 5, 4: 10}[args.cond]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = StapleDiffusionNetwork().to(device)
    model.eval()

    ckpt_path = args.ckpt or f'{CKPT_DIR}/best.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint: {ckpt_path} (epoch={ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.4f})')

    amp_tokens = torch.tensor([pad(encode(args.seq), MAX_LEN)], dtype=torch.long, device=device)
    cond_tensor = torch.full((1,), args.cond, dtype=torch.long, device=device)

    cond_names = {2: 'i,i+4 (S5+S5)', 3: 'i,i+7 (S5+R8)', 4: 'double (S5/R8 mixed)'}
    print(f'\nAMP: {args.seq}')
    print(f'Cond: {args.cond} ({cond_names[args.cond]})')
    print(f'mask_rate={args.mask_rate}, top_n={args.top_n}')

    _, _, _, log_probs = get_probs(model, amp_tokens, cond_tensor, args.mask_rate, device)

    if args.cond == 2:
        cands = build_candidates_cond2(amp_tokens, log_probs)
    elif args.cond == 3:
        cands = build_candidates_cond3(amp_tokens, log_probs)
    else:
        cands = build_candidates_cond4(amp_tokens, log_probs)

    cands.sort(key=lambda x: x[-1], reverse=True)

    print(f'\nResults (top {min(args.top_n, len(cands))}):\n')
    for rank, cand in enumerate(cands[:args.top_n], 1):
        if args.cond == 4:
            i, quad, config, conf = cand
            seq = amp_tokens[i:i+1].clone()
            for p, t in zip([quad[0], quad[1], quad[2], quad[3]], config):
                seq[0, p] = t
            pos_str = f'({quad[0]},{quad[1]}),({quad[2]},{quad[3]})'
            tok_str = f'[{decode_pretty([config[0]])}][{decode_pretty([config[1]])}]/[{decode_pretty([config[2]])}][{decode_pretty([config[3]])}]'
        else:
            i, pair, conf = cand
            seq = amp_tokens[i:i+1].clone()
            a, b = pair
            tok_a, tok_b = (21, 22) if args.cond == 3 else (21, 21)
            seq[0, a], seq[0, b] = tok_a, tok_b
            pos_str = f'{pair}'
            tok_str = f'[{decode_pretty([tok_a])}][{decode_pretty([tok_b])}]'

        pretty = decode_pretty(seq[0].tolist())
        print(f'  #{rank}  {pretty}  confidence={conf:.4f}  pos={pos_str}  tokens={tok_str}')


if __name__ == '__main__':
    main()
