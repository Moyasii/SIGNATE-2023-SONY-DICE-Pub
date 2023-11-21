from typing import List, Tuple, Optional, Dict

import copy
import random
import pickle
import argparse
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from scipy.stats import mode
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class Dice:
    no: int
    image: np.ndarray
    targets: List[int]
    bboxes: List[List[float]]
    centers: List[List[float]]


@dataclass(frozen=True)
class CroppedDice:
    no: int
    image: np.ndarray
    target: int
    mask: np.ndarray
    height: int
    width: int
    area: int


def get_dice_polygon(image_origin: np.ndarray,
                     bin_threshold: int = 1,
                     area_limit: List[int] = [30, 85],
                     ) -> List[np.ndarray]:
    image = copy.deepcopy(image_origin)

    _, image = cv2.threshold(image, bin_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    contours = [c for c, a in zip(contours, areas) if (area_limit[0] < a < area_limit[1])]
    return contours


def get_bbox_and_center(contours: List[np.ndarray]) -> Tuple[List[float]]:
    bboxes = []
    centers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, w, h])
        centers.append([x + w/2, y + h/2])

    return bboxes, centers


def adjust_center(image: np.ndarray,
                  src_centers: List[float]
                  ) -> Tuple[List[float]]:
    num_dices = len(src_centers)
    contours = get_dice_polygon(image, bin_threshold=64)

    def calc_distance(point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    if len(contours) == num_dices:
        bboxes, dst_centers = get_bbox_and_center(contours)
        distances = np.array([[calc_distance(a, b) for b in dst_centers] for a in src_centers])
        indices = np.argmin(distances, axis=1)
        adjust_centers = [dst_centers[dst_idx] for dst_idx in indices]
        bboxes = [bboxes[dst_idx] for dst_idx in indices]
        return adjust_centers, bboxes
    else:
        return src_centers, [[-1.0, -1.0, -1.0, -1.0] for i in range(num_dices)]


def get_dices(x_train: np.ndarray,
              y_train: np.ndarray,
              num_dices: np.ndarray,
              image_size: int,
              predictions_table: Optional[Dict[str, np.ndarray]] = None,
              score_th: float = 0.25,
              ) -> List[Dice]:
    dices: List[Dice] = []
    kmeans_model = KMeans(n_clusters=2, random_state=10, n_init='auto')

    # num_preds_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for no, (image, target, num_dice) in tqdm(enumerate(zip(x_train, y_train, num_dices)), total=len(x_train)):
        image = cv2.resize(image.reshape(20, 20), (image_size, image_size), interpolation=cv2.INTER_NEAREST)

        if (num_dice == 1):
            contours = get_dice_polygon(image)
            if len(contours) == 1:
                # ポリゴンとダイスの数が一致している場合はBBoxを自動生成する
                bboxes, centers = get_bbox_and_center(contours)
                dices.append(Dice(no, image, [int(target)], bboxes, centers))
        else:
            prediction_list = []
            for _, predictions in predictions_table.items():
                # スコアの低い認識結果を破棄する
                prediction = predictions[no][predictions[no][..., 3] > score_th]
                # 必ず2個映っているので認識結果から2つ取得する
                prediction = prediction[:2]
                pred_dice_result = np.sum(prediction[..., 0])
                if len(prediction) == 2 and pred_dice_result == target:
                    # 予測したサイコロの数と予測値があっている場合は採用する
                    prediction_list.append(prediction)

            # num_preds_dist[len(prediction_list)] += 1
            if len(prediction_list) < 2:
                # 正解の予測結果が十分にないのでスキップする
                continue

            predictions = np.vstack(prediction_list)
            kmeans_model.fit(predictions[:, 1:3])
            cluster_1 = predictions[kmeans_model.labels_ == 0]
            cluster_2 = predictions[kmeans_model.labels_ == 1]
            if len(cluster_1) != len(cluster_2):
                # クラスタ分割が等分されない場合はおかしな位置に予測結果が出ている可能性があるためスキップする
                continue

            # クラスタごとに頻出のクラスラベルの結果のみを保持する
            cluster_1_label, cluster_1_mode_count = mode(cluster_1[..., 0].astype(np.int64), keepdims=False)
            cluster_2_label, cluster_2_mode_count = mode(cluster_2[..., 0].astype(np.int64), keepdims=False)
            if cluster_1_mode_count < (len(cluster_1) / 2) or cluster_2_mode_count < (len(cluster_2) / 2):
                # 頻出のクラスラベルが全体の半分未満の場合は予測結果がばらけていて信頼性を欠くのでスキップする
                continue

            # クラスラベルの予測に失敗した情報を削除する
            cluster_1 = cluster_1[cluster_1[..., 0] == cluster_1_label]
            cluster_2 = cluster_2[cluster_2[..., 0] == cluster_2_label]

            targets = []
            centers = []
            for cluster, dice_result in zip([cluster_1, cluster_2], [cluster_1_label, cluster_2_label]):
                targets.append(dice_result)
                #  1: 正規化y座標, 2: 正規化x座標
                x = np.clip((np.mean(cluster[..., 2]) * 40 + 2.0) / 2, 0.0, float(image_size))
                y = np.clip((np.mean(cluster[..., 1]) * 40 + 2.0) / 2, 0.0, float(image_size))
                centers.append([x, y])

            centers, bboxes = adjust_center(image, centers)
            dices.append(Dice(no, image, targets, bboxes, centers))

    # print(num_preds_dist)
    return dices


def get_cropped_dices(x_train: np.ndarray,
                      y_train: np.ndarray,
                      num_dices: np.ndarray,
                      image_size: int,
                      ) -> List[CroppedDice]:
    cropped_dices: List[CroppedDice] = []

    for no, (image, target, num_dice) in enumerate(zip(x_train, y_train, num_dices)):
        image = cv2.resize(image.reshape(20, 20), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        if num_dice != 1:
            # サイコロが2つ映っている場合は処理をスキップする
            continue

        contours = get_dice_polygon(image)
        if len(contours) != 1:
            # ポリゴンが1つでない場合は処理をスキップする
            continue

        bboxes, _ = get_bbox_and_center(contours)
        bbox = bboxes[0]
        dict_image = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        dice_mask = np.where(dict_image > 1, 255, 0).astype(np.uint8)
        height, width = dict_image.shape
        cropped_dice = CroppedDice(no=no,
                                   image=dict_image,
                                   target=target,
                                   mask=dice_mask,
                                   height=height,
                                   width=width,
                                   area=height * width)
        cropped_dices.append(cropped_dice)

    cropped_dices = sorted(cropped_dices, key=lambda x: x.height)
    return cropped_dices


def get_two_dices(cropped_dices: List[CroppedDice],
                  image_size: int,
                  retry: int = 1000,
                  start_no: int = 200000,
                  ) -> List[Dice]:
    composite_dices: List[Dice] = []

    transform = A.Compose([
        A.RandomRotate90(p=1.0),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['bbox_targets']), keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_targets']))

    cropped_dices_work = copy.deepcopy(cropped_dices)
    random.shuffle(cropped_dices_work)

    # TODO: マルチプロセス化
    no = start_no
    while len(cropped_dices_work) > 0:
        retry_count = 0
        target_cropped_dices = [cropped_dices_work.pop() for _ in range(2) if len(cropped_dices_work) > 0]

        while True:
            success_count = 0
            composite_image = np.ones((image_size, image_size), dtype=np.uint8)
            bboxes = []
            centers = []
            targets = []

            # ランダム配置
            for target_cropped_dice in target_cropped_dices:
                target_image = target_cropped_dice.image.copy()
                mask = target_cropped_dice.mask
                target_h, target_w = target_image.shape[:2]
                top = np.random.randint(0, image_size - target_h + 1)
                left = np.random.randint(0, image_size - target_w + 1)
                bottom = top + target_h
                right = left + target_w
                roi = composite_image[top:bottom, left:right]
                if np.all(roi[mask == 255] == 1):
                    success_count += 1
                    composite_image[top:bottom, left:right][mask == 255] = target_image[mask == 255]
                    bboxes.append([left, top, right-left, bottom-top])
                    centers.append([(left + right) / 2, (top + bottom) / 2])
                    targets.append(target_cropped_dice.target)

            if success_count == len(target_cropped_dices):
                transformed = transform(image=composite_image, bboxes=bboxes, keypoints=centers,
                                        bbox_targets=targets, keypoint_targets=targets)
                transformed_composite_image = transformed['image']
                transformed_targets = transformed['keypoint_targets']
                transformed_bboxes = transformed['bboxes']
                transformed_centers = transformed['keypoints']
                # composite_dices.append(Dice(no, transformed_composite_image, transformed_targets, transformed_bboxes))
                composite_dices.append(Dice(no, transformed_composite_image, transformed_targets,
                                       transformed_bboxes, transformed_centers))
                no += 1
                break
            else:
                retry_count += 1
                if retry_count > retry:
                    break

    return composite_dices


def calc_margins(image: np.ndarray) -> List[int]:
    _, bin_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    y, x = np.where(bin_image == 255)

    left = np.min(x)
    right = np.max(x)
    top = np.min(y)
    bottom = np.max(y)

    # 余白を計算
    left_margin = left
    right_margin = bin_image.shape[1] - right - 1
    top_margin = top
    bottom_margin = bin_image.shape[0] - bottom - 1

    return [left_margin, right_margin, top_margin, bottom_margin]


def shift_image(image: np.ndarray,
                shift_ratio: float = 0.75
                ) -> np.ndarray:
    src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
    dst = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)

    shift_pixel_x = 0
    shift_pixel_y = 0
    left_margin, right_margin, top_margin, bottom_margin = calc_margins(image)
    if (left_margin + right_margin) > (top_margin + bottom_margin):
        # 左右の余白が大きい場合は左右のどちらかに寄せる
        if np.random.rand() > 0.5:
            # 右方向にずらす
            shift_pixel_x = np.random.randint(int(right_margin * shift_ratio), right_margin + 1)
        else:
            # 左方向にずらす
            shift_pixel_x = np.random.randint(-left_margin, -int(left_margin * shift_ratio) + 1)
    else:
        # 上下の余白が大きい場合は上下のどちらかに寄せる
        if np.random.rand() > 0.5:
            # 下方向にずらす
            shift_pixel_y = np.random.randint(int(bottom_margin * shift_ratio), bottom_margin + 1)
        else:
            # 上方向にずらす
            shift_pixel_y = np.random.randint(-top_margin, -int(top_margin * shift_ratio) + 1)

    dst[:, 0] += shift_pixel_x
    dst[:, 1] += shift_pixel_y
    affine = cv2.getAffineTransform(src, dst)

    height, width = image.shape[:2]
    shifted_image = cv2.warpAffine(image, affine, (width, height))
    shifted_image[shifted_image == 0] = 1

    return shifted_image, shift_pixel_x, shift_pixel_y


def get_three_dices(dices: List[Dice],
                    cropped_dices: List[CroppedDice],
                    retry: int = 50,
                    start_no: int = 200000,
                    ) -> List[Dice]:

    three_dices: List[Dice] = []

    dice_transform = A.Compose([A.RandomRotate90(p=1.0)],
                               keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
    cropped_dice_transform = A.Compose([A.RandomRotate90(p=1.0)],
                                       additional_targets={'mask': 'image'})

    dices_work = copy.deepcopy(dices)
    dices_work = [d for d in dices_work if len(d.centers) == 2]
    cropped_dices_work = copy.deepcopy(cropped_dices)

    # TODO: マルチプロセス化
    no = start_no
    max_dices_length = max(len(dices_work), len(cropped_dices_work))
    pbar = tqdm(total=max_dices_length)
    prev_processed = 0
    while len(dices_work) > 0 and len(cropped_dices_work) > 0:
        dice = dices_work.pop()

        # 進捗表示
        min_dices_length = min(len(dices_work), len(cropped_dices_work))
        cur_processed = max_dices_length - min_dices_length
        pbar.update(cur_processed - prev_processed)
        prev_processed = cur_processed

        transformed = dice_transform(image=dice.image, keypoints=dice.centers, keypoint_labels=dice.targets)
        canvas_image = transformed['image']
        canvas_image_height, canvas_image_width = canvas_image.shape[:2]
        centers = transformed['keypoints']
        targets = transformed['keypoint_labels']

        canvas_image, shift_pixel_x, shift_pixel_y = shift_image(canvas_image)
        centers = [[c[0]+shift_pixel_x, c[1]+shift_pixel_y] for c in centers]

        for _ in range(retry):
            cropped_dice = random.sample(cropped_dices_work, 1)[0]
            transformed_cropped = cropped_dice_transform(image=cropped_dice.image, mask=cropped_dice.mask)
            overlay_image = transformed_cropped['image']
            overlay_mask = transformed_cropped['mask']
            overlay_target = cropped_dice.target
            overlay_image_height, overlay_image_width = overlay_image.shape[:2]

            transformed_image = copy.deepcopy(canvas_image)
            transformed_targets = copy.deepcopy(targets)
            transformed_centers = copy.deepcopy(centers)
            paste_bboxes = []
            for y in range(canvas_image_height - overlay_image_height + 1):
                for x in range(canvas_image_width - overlay_image_width + 1):
                    roi = transformed_image[y:y+overlay_image_height, x:x+overlay_image_width]
                    if np.all(roi[overlay_mask == 255] == 1):
                        paste_bboxes.append([x, y, x+overlay_image_width, y+overlay_image_height])

            if len(paste_bboxes) > 0:
                b = random.choice(paste_bboxes)
                transformed_image[b[1]:b[3], b[0]:b[2]][overlay_mask == 255] = overlay_image[overlay_mask == 255]
                transformed_targets.append(overlay_target)
                transformed_centers.append([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                three_dices.append(Dice(no, transformed_image, transformed_targets, [], transformed_centers))
                no += 1
                cropped_dices_work = [d for d in cropped_dices_work if d.no != cropped_dice.no]
                break

    return three_dices


def create_dataset(all_dices: List[Dice]) -> Tuple[np.ndarray]:
    x_train_list = []
    y_train_list = []
    y_train_center_list = []
    y_train_center_target_list = []

    for dice in all_dices:
        x_train = dice.image.reshape(-1)
        y_train = np.sum(dice.targets)
        y_train_center = []
        for i in range(3):
            y_train_center.append(dice.centers[i] if i < len(dice.centers) else [-1.0, -1.0])
        y_train_center_target = []
        for i in range(3):
            y_train_center_target.append(dice.targets[i] if i < len(dice.targets) else 255)

        x_train_list.append(x_train)
        y_train_list.append(y_train)
        y_train_center_list.append([y_train_center])
        y_train_center_target_list.append([y_train_center_target])

    x_train = np.vstack(x_train_list, dtype=np.uint8)
    y_train = np.stack(y_train_list, dtype=np.int64)
    y_train_center = np.vstack(y_train_center_list, dtype=np.float32)
    y_train_center_target = np.vstack(y_train_center_target_list, dtype=np.int64)

    return x_train, y_train, y_train_center, y_train_center_target


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    x_train = np.load(args.x_train_path)
    y_train = np.load(args.y_train_path)
    num_dices = np.load(args.num_dices_path)

    predictions_table = dict()
    if args.center_pred_root_list is not None:
        for i, center_pred_root in enumerate(args.center_pred_root_list):
            center_pred_root = Path(center_pred_root)
            for predictions_npy_path in sorted(center_pred_root.glob('**/predictions.npy')):
                model = predictions_npy_path.parts[-4]
                predictions = np.load(predictions_npy_path)
                predictions[..., 0] += 1
                predictions_table[f'{i}_{model}'] = predictions

    print('アノテーションを自動で作成します')
    dices = get_dices(x_train, y_train, num_dices, args.image_size, predictions_table)
    with open(output_dir / 'dices.pkl', mode='wb') as f:
        pickle.dump(dices, f)

    two_dices = []
    if args.two_dice:
        cropped_dices = get_cropped_dices(x_train, y_train, num_dices, args.image_size)
        with open(output_dir / 'cropped_dices.pkl', mode='wb') as f:
            pickle.dump(cropped_dices, f)
        print('画像同士を組み合わせてサイコロが2つ映った画像を作成中・・・（数分かかるので気長に待機してください）')
        two_dices = get_two_dices(cropped_dices, args.image_size)
        with open(output_dir / 'two_dices.pkl', mode='wb') as f:
            pickle.dump(two_dices, f)
    three_dices = []
    if args.three_dice:
        cropped_dices = get_cropped_dices(x_train, y_train, num_dices, args.image_size)
        with open(output_dir / 'cropped_dices.pkl', mode='wb') as f:
            pickle.dump(cropped_dices, f)
        print('画像同士を組み合わせてサイコロが3つ映った画像を作成中・・・（数十分かかるので気長に待機してください）')
        three_dices = get_three_dices(dices, cropped_dices)
        with open(output_dir / 'three_dices.pkl', mode='wb') as f:
            pickle.dump(three_dices, f)

    # データセットを作成し保存する
    dices = dices + two_dices + three_dices
    x_train, y_train, y_train_center, y_train_center_target = create_dataset(dices)
    np.save(output_dir / 'X_train', x_train)
    np.save(output_dir / 'y_train', y_train)
    np.save(output_dir / 'y_train_center', y_train_center)
    np.save(output_dir / 'y_train_center_target', y_train_center_target)

    print(f'save {(output_dir / "X_train.npy").as_posix()}')
    print(f'save {(output_dir / "y_train.npy").as_posix()}')
    print(f'save {(output_dir / "y_train_center.npy").as_posix()}')
    print(f'save {(output_dir / "y_train_center_target.npy").as_posix()}')

    print(f'画像同士を組み合わせてサイコロが2つ映った画像を{len(two_dices)}枚生成しました')
    print(f'画像同士を組み合わせてサイコロが3つ映った画像を{len(three_dices)}枚生成しました')
    print(f'{len(dices)}件にアノテーションしました')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('x_train_path', type=str)
    parser.add_argument('y_train_path', type=str)
    parser.add_argument('num_dices_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--center_pred_root_list', nargs="*", type=str, default=None)
    parser.add_argument('--image_size', type=int, default=20)
    parser.add_argument('--two_dice', action='store_true')
    parser.add_argument('--three_dice', action='store_true')
    main(parser.parse_args())
