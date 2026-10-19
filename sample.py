import argparse
import torch

from config import CKPT_DIR, TEMPERATURE, NUM_SAMPLE_STEPS
from data.tokenizer import decode
from model.network import StapleDiffusionNetwork
from model.sampling import (
    DEFAULT_SAMPLE_LEN, sample, sample_autoregressive_final,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cond', type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--temp', type=float, default=TEMPERATURE)
    parser.add_argument('--steps', type=int, default=NUM_SAMPLE_STEPS)
    parser.add_argument('--max_len', type=int, default=DEFAULT_SAMPLE_LEN,
                        help='generated sequence length (residues)')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--method', type=str, default='iterative', choices=['iterative', 'one_shot'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = StapleDiffusionNetwork().to(device)
    model.eval()

    ckpt_path = args.ckpt or f'{CKPT_DIR}/best.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint: {ckpt_path} (epoch={ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.4f})')

    cond_names = {1: 'self', 2: 'i,i+4', 3: 'i,i+7', 4: 'double'}
    print(f'\nGenerating {args.count} samples (cond={args.cond}, {cond_names.get(args.cond, "?")})')
    print(f'  temperature={args.temp}, steps={args.steps}, method={args.method}')

    cond = torch.full((args.count,), args.cond, dtype=torch.long, device=device)

    with torch.no_grad():
        if args.method == 'iterative':
            tokens = sample(model, cond, steps=args.steps,
                            temperature=args.temp, max_len=args.max_len,
                            device=device)
        else:
            tokens = sample_autoregressive_final(model, cond,
                                                  temperature=args.temp,
                                                  max_len=args.max_len,
                                                  device=device)

    print()
    for i in range(args.count):
        seq = decode(tokens[i].tolist())
        print(f'  {i+1}. {seq}')


if __name__ == '__main__':
    main()
