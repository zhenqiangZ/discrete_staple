#  python sample_based_amp.py WIQPATAVIPYVRYL --cond 2 --temp 1.5 --steps 1


import argparse
import torch
import torch.nn.functional as F

from config import MAX_LEN, TEMPERATURE
from data.tokenizer import encode, pad, decode
from model.network import StapleDiffusionNetwork


@torch.no_grad()
def predict_from_amp(model, amp_tokens: torch.Tensor, cond: torch.Tensor,
                     temperature: float = TEMPERATURE) -> torch.Tensor:
    logits = model(amp_tokens, cond)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view_as(amp_tokens)


@torch.no_grad()
def refine_with_confidence(model, amp_tokens: torch.Tensor, cond: torch.Tensor,
                           steps: int = 5, temperature: float = TEMPERATURE) -> torch.Tensor:
    x = amp_tokens.clone()
    B, L = x.shape

    for _ in range(steps):
        logits = model(x, cond)
        probs = F.softmax(logits / temperature, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        confidence, x_new = log_probs.max(dim=-1)

        num_update = max(1, L // (steps * 2))
        for i in range(B):
            _, idx = confidence[i].sort()
            update_pos = idx[:num_update]
            x[i, update_pos] = torch.multinomial(probs[i, update_pos], 1).squeeze(-1)

    return x


def main():
    parser = argparse.ArgumentParser(description='基于给定 AMP 序列设计订书肽')
    parser.add_argument('seq', type=str, help='AMP 氨基酸序列')
    parser.add_argument('--cond', type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--temp', type=float, default=TEMPERATURE)
    parser.add_argument('--steps', type=int, default=5,
                        help='精炼步数 (1=单次直接预测)')
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = StapleDiffusionNetwork().to(device)
    model.eval()

    # ckpt_path = args.ckpt or f'{CKPT_DIR}/best.pt'
    # ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # model.load_state_dict(ckpt['model_state_dict'])
    # print(f'Loaded checkpoint: {ckpt_path} (epoch={ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.4f})')

    ckpt_path = args.ckpt or f'./checkpoints/rl_grpo/rl_grpo_best.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint: {ckpt_path}')
    
    amp_tokens = pad(encode(args.seq), MAX_LEN)
    amp_tensor = torch.tensor([amp_tokens], dtype=torch.long, device=device)
    amp_tensor = amp_tensor.repeat(args.count, 1)

    cond_tensor = torch.full((args.count,), args.cond, dtype=torch.long, device=device)

    cond_names = {1: 'self', 2: 'i,i+4', 3: 'i,i+7', 4: 'double'}
    print(f'\nAMP: {args.seq}')
    print(f'Cond: {args.cond} ({cond_names[args.cond]})')

    with torch.no_grad():
        if args.steps <= 1 or args.cond == 1:
            tokens = predict_from_amp(model, amp_tensor, cond_tensor, args.temp)
        else:
            tokens = refine_with_confidence(model, amp_tensor, cond_tensor,
                                             args.steps, args.temp)

    print(f'Results ({args.count} samples):\n')
    for i in range(args.count):
        seq = decode(tokens[i].tolist())
        print(f'  {i+1}. {seq}')


if __name__ == '__main__':
    main()
