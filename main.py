import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage import io

from dataset import DatasetIcmbio
from models import build_model
from trainer import EarlyStopping, Trainer
from utils import (
    calculate_cm,
    convert_to_color,
    is_save_epoch,
    iter_original_test_pairs,
    load_datasets,
    metrics_from_confusion_matrix,
    original_test_label_paths,
    palette,
    weights_calculator_loss,
)


def save_segmentation(output_dir, tile_id, prediction, num_classes):
    filename = output_dir / f'inference_tile_{tile_id}.png'
    colors = convert_to_color(prediction, palette) if num_classes == len(palette) else convert_to_color(prediction)
    io.imsave(str(filename), colors, check_contrast=False)


def evaluate_original_resolution(predictions, test_images, original_labels, params, best_epoch):
    output_dir = Path(params['results_folder']) / 'original_results'
    segmentations_dir = output_dir / 'segmentations'
    segmentations_dir.mkdir(parents=True, exist_ok=True)

    classes = params['classes']
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for tile_id, prediction, label in iter_original_test_pairs(
        predictions, test_images, original_labels, params['dim_original']
    ):
        confusion_matrix += calculate_cm(prediction.ravel(), label.ravel(), range(len(classes)))
        save_segmentation(segmentations_dir, tile_id, prediction, len(classes))

    _, information = metrics_from_confusion_matrix(confusion_matrix, classes, output_dir)
    with (output_dir / 'training_results.txt').open('w', encoding='utf-8') as report:
        report.write(f'Evaluation checkpoint: best_model.pth.tar (epoch {best_epoch})\n')
        for line in information:
            report.write(f'{line}\n')


def run_fold(params):
    fold = params['fold']
    results_dir = Path(params['results_folder'])
    print(f'Starting training for {fold}...')
    start_time = time.time()

    consistency = params['semi_supervised'] and params.get('consistency', True)
    contrastive = params['semi_supervised'] and params.get('contrastive', True)
    use_teacher = consistency or contrastive

    train_images, train_labels = load_datasets(params, 'train')
    test_images, test_labels = load_datasets(params, 'test')
    val_images, val_labels = load_datasets(params, 'val')

    if not test_images or len(test_images) != len(test_labels):
        raise ValueError('The test split must contain one label per image.')
    tile_ids = [Path(image).stem for image in test_images]
    if len(set(tile_ids)) != len(tile_ids):
        raise ValueError('The test image list contains duplicate tile identifiers.')

    original_dir = params.get('root_dir_original')
    original_labels_dir = Path(original_dir) / 'labels_modificadas' if original_dir else None
    original_labels = None
    if tuple(params['dim_original']) != tuple(params['dim_entrada']):
        if original_labels_dir is None:
            raise ValueError('Set root_dir_original to evaluate at the original resolution.')
        original_labels = original_test_label_paths(test_images, test_labels, original_labels_dir)

    dataset_options = {'window_size': params['window_size'], 'cache': params['cache']}
    label_options = {
        'icmbio_labels': params['n_classes'] == len(palette),
        'original_labels_dir': original_labels_dir,
    }
    train_dataset = DatasetIcmbio(
        train_images, label_files=train_labels, len=params['len_train'],
        augmentation=params['augmentation'], **dataset_options, **label_options,
    )
    test_dataset = DatasetIcmbio(
        test_images, label_files=test_labels, len=params['len_train'],
        augmentation=False, **dataset_options,
    )
    val_dataset = DatasetIcmbio(
        val_images, label_files=val_labels, len=1400, augmentation=False, **dataset_options, **label_options,
    )
    loaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=params['bs'], shuffle=True),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=params['bs']),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=32),
    }

    if use_teacher:
        if not params['teacher']:
            raise ValueError('Consistency and contrastive training require teacher: true.')
        unlabeled_images, _ = load_datasets(params, 'train_unlabeled')
        unlabeled_dataset = DatasetIcmbio(
            unlabeled_images, len=params['len_unlabeled'],
            augmentation=params['augmentation'], **dataset_options,
        )
        loaders['train_unlabeled'] = torch.utils.data.DataLoader(
            unlabeled_dataset, batch_size=params['bs_unlabeled'], shuffle=True,
        )

    weights_calculator_loss(params, train_labels)
    model_name = params['model']['name']
    model = build_model(model_name, params)
    teacher_model = build_model(model_name, params) if use_teacher else None
    trainer = Trainer(model, params, loader=loaders, net_teacher=teacher_model, semi=params['semi_supervised'])
    early_stopping = EarlyStopping(
        patience=params['early_stopping_patience'], path=results_dir / 'best_model.pth.tar',
    )

    for epoch in range(trainer.last_epoch + 1, params['maximum_epochs'] + 1):
        trainer.train()
        if is_save_epoch(params, epoch, ignore_epoch=params['maximum_epochs']):
            trainer.save(results_dir / f'trained_epoch_{epoch}.pth.tar')

        validation_loss = trainer.epoch_val_losses[-1]
        if not np.isfinite(validation_loss):
            raise ValueError(f'Validation loss at epoch {epoch} is not finite: {validation_loss}.')
        early_stopping(validation_loss, trainer.net, epoch)
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch} (best epoch: {early_stopping.best_epoch}).')
            break

    trainer.save(results_dir / f'finalresult_{model_name}.pth.tar')
    training_time_hours = (time.time() - start_time) / 3600

    trainer.net.load_state_dict(torch.load(early_stopping.path, map_location='cpu', weights_only=True))
    print(f'Testing the model from epoch {early_stopping.best_epoch}...')
    _, predictions, _, information = trainer.test(
        all=True, stride=params['test_stride'], batch_size=params['batch_test'],
        original_label_files=original_labels,
    )

    segmentations_dir = results_dir / 'segmentations'
    segmentations_dir.mkdir(exist_ok=True)
    for tile_id, prediction in zip(tile_ids, predictions, strict=True):
        save_segmentation(segmentations_dir, tile_id, prediction, params['n_classes'])

    with (results_dir / 'training_results.txt').open('w', encoding='utf-8') as report:
        report.write(f'Model: {model_name}\n')
        report.write(f'Consistency: {consistency} - Contrastive: {contrastive}\n')
        report.write(f'Loss function: {params["loss"]["name"]} - Loss weights: {params["loss"]["params"]["weights"]}\n')
        report.write(f'Optimizer: {params["optimizer_params"]["optimizer"]} - Augmentation: {params["augmentation"]}\n')
        if consistency or contrastive:
            report.write(f'Unlabeled input noise: {params["noise"]}\n')
        if consistency:
            report.write(f'Consistency weight: {params["w_consistency"]}\n')
        if contrastive:
            report.write(f'Contrastive weight: {params["w_contrastive"]}\n')
        report.write(f'Training time: {training_time_hours:.2f} hours - Num epochs: {epoch}\n')
        report.write(f'Evaluation checkpoint: best_model.pth.tar (epoch {early_stopping.best_epoch})\n')
        report.write(f'Best validation loss: {early_stopping.val_loss_min}\n')
        for line in information:
            report.write(f'{line}\n')

    with (results_dir / 'params.txt').open('w', encoding='utf-8') as report:
        for name, value in params.items():
            report.write(f'{name}: {value}\n')

    if original_labels is not None:
        evaluate_original_resolution(
            predictions, test_images, original_labels, params, early_stopping.best_epoch,
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate a segmentation model.')
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument('config_path', nargs='?', type=Path, help='Path to the YAML configuration.')
    config.add_argument('--config', type=Path, help='Path to the YAML configuration.')
    args = parser.parse_args()
    return args.config or args.config_path


def main():
    with parse_args().open(encoding='utf-8') as config_file:
        params = yaml.safe_load(config_file)

    if params['maximum_epochs'] < 1:
        raise ValueError('maximum_epochs must be at least 1.')
    if params.get('cpu') is not None:
        params['device'] = 'cpu'

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for fold in params['folds']:
        run_fold(params | {'fold': fold, 'results_folder': str(Path(params['results_path']) / fold)})


if __name__ == '__main__':
    main()
