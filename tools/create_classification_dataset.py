from typing import List, Optional, Dict

import random
import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans


def create_train_dataset(x_train: np.ndarray,
                         y_train_center: np.ndarray,
                         y_train_center_target: np.ndarray,
                         output_root: Path,
                         crop_half: int
                         ) -> None:
    output_dirs = [output_root / f'{i}' for i in range(6)]
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)

    for no, (image, centers, targets) in tqdm(enumerate(zip(x_train, y_train_center, y_train_center_target)), total=len(x_train)):
        mask = targets != 255
        centers = centers[mask]
        targets = targets[mask]
        for i, (center, target) in enumerate(zip(centers, targets)):
            x, y = center
            target -= 1
            left = np.clip(int(round(x - crop_half)), 0, 20)
            top = np.clip(int(round(y - crop_half)), 0, 20)
            right = np.clip(int(round(x + crop_half)), 0, 20)
            bottom = np.clip(int(round(y + crop_half)), 0, 20)
            crop_image = image[top:bottom, left:right]
            cv2.imwrite((output_dirs[target] / f'{no:08d}_{i}.png').as_posix(), crop_image)


def pred_num_dices_test(predictions_table: Dict[str, np.ndarray],
                        score_th: float,
                        output_dir: Optional[Path] = None,
                        ) -> np.ndarray:
    data = dict()
    for model, predictions in predictions_table.items():
        num_centers = []
        for p in predictions:
            trusted_p_by_threshold = p[p[..., 3] >= score_th]
            num_centers.append(int(np.sum(trusted_p_by_threshold[..., 0])))
        data[model] = num_centers
    test_num_dices_df = pd.DataFrame(data=data)
    max_frequency = test_num_dices_df.apply(lambda row: Counter(row).most_common(1)[0][1], axis=1)
    pred = test_num_dices_df.apply(lambda row: Counter(row).most_common(1)[0][0], axis=1).to_numpy()
    test_num_dices_df['max_frequency'] = max_frequency
    test_num_dices_df['pred'] = pred

    if output_dir is not None:
        test_num_dices_df.to_csv(output_dir / 'num_dices_test.csv', index=False)
    return pred


def pred_centers_test(x_test: np.ndarray,
                      num_dices: np.ndarray,
                      predictions_table: Dict[str, np.ndarray],
                      score_th: float
                      ) -> List[float]:
    centers_test = []
    for no in range(len(x_test)):
        num_dice = num_dices[no]
        prediction_list = []
        for _, predictions in predictions_table.items():
            # スコアの低い認識結果を破棄する
            prediction = predictions[no][predictions[no][..., 3] > score_th]
            prediction = prediction[:num_dice]
            if len(prediction) == num_dice:
                # 数があっている場合は採用する
                prediction_list.append(prediction)

        if len(prediction_list) <= 0:
            # 正解の予測結果がないのでスキップする
            centers_test.append([])
            continue

        # x, y座標をクラスタリング
        predictions = np.vstack(prediction_list)
        kmeans_model = KMeans(n_clusters=num_dice, random_state=42, n_init='auto')
        kmeans_model.fit(predictions[:, 1:3])
        clusters = [predictions[kmeans_model.labels_ == i] for i in range(num_dice)]
        num_points = [len(c) for c in clusters]
        if len(set(num_points)) != 1:
            # クラスタ分割が等分されない場合はおかしな位置に予測結果が出ている可能性があるためスキップする
            centers_test.append([])
            continue

        centers = []
        for cluster in clusters:
            #  1: 正規化y座標, 2: 正規化x座標
            x = np.clip((np.mean(cluster[..., 2]) * 40 + 2.0) / 2, 0.0, 20.0)
            y = np.clip((np.mean(cluster[..., 1]) * 40 + 2.0) / 2, 0.0, 20.0)
            centers.append([x, y])
        centers_test.append(centers)

    return centers_test


def create_test_dataset(x_test: np.ndarray,
                        centers_test: np.ndarray,
                        output_dir: Path,
                        crop_half: int,
                        ) -> None:
    for no, image in tqdm(enumerate(x_test), total=len(x_test)):
        for i, (x, y) in enumerate(centers_test[no]):
            left = np.clip(int(round(x - crop_half)), 0, 20)
            top = np.clip(int(round(y - crop_half)), 0, 20)
            right = np.clip(int(round(x + crop_half)), 0, 20)
            bottom = np.clip(int(round(y + crop_half)), 0, 20)
            crop_image = image[top:bottom, left:right]
            cv2.imwrite((output_dir / f'{no:08d}_{i}.png').as_posix(), crop_image)


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    print('データセットを自動で作成します')

    output_dir = Path(args.output_dir)
    output_train_dir = output_dir / 'train'
    output_train_dir.mkdir(parents=True, exist_ok=False)
    output_test_dir = output_dir / 'test'
    output_test_dir.mkdir(parents=True, exist_ok=False)

    x_train = np.load(args.x_train_path).reshape(-1, 20, 20)
    y_train_center = np.load(args.y_train_center_path)
    y_train_center_target = np.load(args.y_train_center_target_path)

    create_train_dataset(x_train, y_train_center, y_train_center_target,
                         output_train_dir, args.crop_half)

    predictions_table = dict()
    if args.center_pred_root_list is not None:
        for i, center_pred_root in enumerate(args.center_pred_root_list):
            center_pred_root = Path(center_pred_root)
            for predictions_npy_path in sorted(center_pred_root.glob('**/predictions.npy')):
                model = predictions_npy_path.parts[-4]
                predictions = np.load(predictions_npy_path)
                predictions[..., 0] += 1
                predictions_table[f'{i}_{model}'] = predictions

    x_test = np.load(args.x_test_path).reshape(-1, 20, 20)
    num_dices_test = pred_num_dices_test(predictions_table, args.score_th, output_dir=output_dir)
    centers_test = pred_centers_test(x_test, num_dices_test, predictions_table, args.score_th)
    create_test_dataset(x_test, centers_test, output_test_dir, args.crop_half)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('x_train_path', type=str)
    parser.add_argument('y_train_center_path', type=str)
    parser.add_argument('y_train_center_target_path', type=str)
    parser.add_argument('x_test_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--center_pred_root_list', nargs="*", type=str, default=None)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--crop_half', type=int, default=7)
    parser.add_argument('--score_th', type=float, default=0.25)
    main(parser.parse_args())
