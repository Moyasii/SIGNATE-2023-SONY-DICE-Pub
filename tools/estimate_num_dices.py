import argparse
from pathlib import Path
import numpy as np


def main(args: argparse.Namespace) -> None:
    # 画素値が2以上のピクセル数の分布を確認すると、
    # 2つの山があり100程度に閾値を設けることでサイコロの数を推定できることがわかる
    x_train = np.load(args.x_train_path)
    count_bright_pixels_per_dice = np.sum(x_train > 1, axis=1)
    num_dices = np.where(count_bright_pixels_per_dice < 100, 1, 2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'X_train_num_dices', num_dices)
    print(f'save {(output_dir / "X_train_num_dices.npy").as_posix()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('x_train_path', type=str)
    parser.add_argument('--output_dir', type=str, default='data')
    main(parser.parse_args())
