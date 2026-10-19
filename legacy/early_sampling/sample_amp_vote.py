# python sample_amp_vote.py WIQPATAVIPYVRYL --cond 2 --mask_rate 0.1 --temp 1.5 --count 5

import argparse
import torch
import torch.nn.functional as F

from config import CKPT_DIR, MAX_LEN, MASK_TOKEN, PAD_TOKEN
from data.tokenizer import encode, pad, decode
from model.network import StapleDiffusionNetwork


@torch.no_grad()
def design(model, amp_tokens: torch.Tensor, cond: torch.Tensor,
           mask_rate: float = 0.15, temperature: float = 0.5) -> torch.Tensor:
    B, L = amp_tokens.shape
    is_pad = (amp_tokens == PAD_TOKEN)
    results = []

    for i in range(B):
        valid_pos = (~is_pad[i]).nonzero(as_tuple=True)[0]
        n_valid = valid_pos.numel()
        chunk_size = max(1, int(n_valid * mask_rate))
        shuffled = valid_pos[torch.randperm(n_valid, device=amp_tokens.device)]
        chunks = [shuffled[j:j+chunk_size] for j in range(0, n_valid, chunk_size)]

        preds = {}
        for chunk in chunks:
            x_masked = amp_tokens[i:i+1].clone()
            x_masked[:, chunk] = MASK_TOKEN

            logits = model(x_masked, cond[i:i+1])
            probs = F.softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, L)
            # print(tokens)

            for pos in chunk.tolist():
                preds[pos] = tokens[0, pos]

        final = amp_tokens[i].clone()
        for pos, token in preds.items():
            final[pos] = token
        results.append(final)

    return torch.stack(results)


def main():
    parser = argparse.ArgumentParser(description='基于 AMP 序列设计订书肽（逐块预测拼接）')
    parser.add_argument('seq', type=str, help='AMP 氨基酸序列')
    parser.add_argument('--cond', type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='每块 mask 比例 (占非PAD长度)')
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = StapleDiffusionNetwork().to(device)
    model.eval()

    ckpt_path = args.ckpt or f'{CKPT_DIR}/best.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint: {ckpt_path} (epoch={ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.4f})')

    amp_tokens = pad(encode(args.seq), MAX_LEN)
    amp_tensor = torch.tensor([amp_tokens], dtype=torch.long, device=device)
    amp_tensor = amp_tensor.repeat(args.count, 1)

    cond_tensor = torch.full((args.count,), args.cond, dtype=torch.long, device=device)

    cond_names = {1: 'self', 2: 'i,i+4', 3: 'i,i+7', 4: 'double'}
    print(f'\nAMP: {args.seq}')
    print(f'Cond: {args.cond} ({cond_names[args.cond]})')

    if args.cond == 1:
        print('  cond=1: identity (no change)')
        tokens = amp_tensor
    else:
        n_valid = sum(1 for t in amp_tokens if t != PAD_TOKEN)
        print(f'  mask_rate={args.mask_rate}, seq_len={n_valid}, '
              f'chunks=~{n_valid // max(1, int(n_valid * args.mask_rate))}')
        tokens = design(model, amp_tensor, cond_tensor,
                        args.mask_rate, args.temp)

    print(f'\nResults ({args.count} samples):\n')
    for i in range(args.count):
        seq = decode(tokens[i].tolist())
        print(f'  {i+1}. {seq}')


if __name__ == '__main__':
    main()
