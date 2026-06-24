from skimage import io
import os, time
import torch

from dataset import DatasetIcmbio
from trainer import Trainer, EarlyStopping
from models import build_model
from utils import clear, convert_to_color, load_datasets, is_save_epoch, weights_calculator_loss, load_predictions_resized, metrics, load_original_test_labels

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

import sys
import yaml

# loading params settings file
if len(sys.argv) < 2:
    print("Uso: python3 main.py <arquivo_params.yaml>")
    sys.exit(1)

yaml_file = sys.argv[1]

with open(yaml_file, "r") as f:
    full_params = yaml.safe_load(f)
    
if __name__=='__main__':

    params = {}

    for fold in full_params['folds']:
        params = full_params.copy()
        
        params.update({
            "results_folder": os.path.join(params['results_path'], f'{fold}'),
            "fold": fold,
        })

        print(f"Starting training for {fold}...")
        start_time = time.time()    

        # Create train, train_unlabeled, val and test sets
        train_images, train_labels = load_datasets(params, 'train')
        train_images_unlabeled, _ = load_datasets(params, 'train_unlabeled') 
        test_images, test_labels = load_datasets(params, 'test')
        val_images, val_labels = load_datasets(params, 'val') 

        train_dataset = DatasetIcmbio(train_images, label_files=train_labels, window_size = params['window_size'], cache = params['cache'], len=params['len_train'], augmentation=params['augmentation'])
        train_unlabeled_dataset = DatasetIcmbio(train_images_unlabeled, window_size = params['window_size'], cache = params['cache'], len=params['len_unlabeled'], 
                                                augmentation=params['augmentation'])
        test_dataset = DatasetIcmbio(test_images, label_files=test_labels, window_size = params['window_size'], cache = params['cache'], len=params['len_train'], augmentation=False)
        val_dataset = DatasetIcmbio(val_images, label_files=val_labels, window_size = params['window_size'], cache = params['cache'], len=1400, augmentation=False)
        
        # # Load dataset classes in pytorch dataloader handler object
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = params['bs'], shuffle=True)
        train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_dataset, batch_size = params['bs_unlabeled'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = params['bs'], shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, shuffle=False)

        loader = {
            "train": train_loader,
            "train_unlabeled": train_unlabeled_loader,
            "test": test_loader,
            "val": val_loader
        }

        # load the weights for the loss function
        weights_calculator_loss(params, train_labels)    

        model = build_model(model_name=params['model']['name'], params=params)
        
        cbkp=None
        
        if params['teacher']:
            teacher_model = build_model(model_name=params['model']['name'], params=params)
            trainer = Trainer(model, params, loader=loader, net_teacher=teacher_model, semi=params['semi_supervised'],cbkp=cbkp)
        else:
            trainer = Trainer(model, params, loader=loader, semi=params['semi_supervised'],cbkp=cbkp)
        
        print('Starting training...')
        
        early_stopping = EarlyStopping(patience=params["early_stopping_patience"], verbose=False, path=os.path.join(params['results_folder'], 'best_model.pth.tar'))
        
        for epoch in range(trainer.last_epoch+1, params['maximum_epochs'] + 1):
            
            trainer.train()

            if is_save_epoch(params, epoch, ignore_epoch=params['maximum_epochs']):
                trainer.save(os.path.join(params['results_folder'], 'trained_epoch_{}.pth.tar'.format(epoch)))

            early_stopping(trainer.epoch_val_losses[-1], trainer.net, epoch)

            if early_stopping.early_stop:
                print("Early stopping activated - epoch: {} - best_epoch: {} ".format(epoch, early_stopping.best_epoch))
                break
        
        print('Saving final model...')
        trainer.save(os.path.join(params['results_folder'], 'finalresult_{}.pth.tar'.format(params['model']['name'])))
        print('Final model saved.')

        end_time = time.time()
        training_time = end_time - start_time
        training_time_hours = training_time / 3600.0
        print("Training finished. Tempo gasto: {:.2f} horas".format(training_time_hours))

        # Test the model.
        print('Testing the model...')
        acc, all_preds, all_gts, informations = trainer.test(all=True, stride=params['test_stride'], batch_size=params['batch_test'])
        
        input_ids, label_ids = test_loader.dataset.get_dataset()
        all_ids = [os.path.split(f)[1].split('.')[0] for f in input_ids]
        
        # saving inference results
        output_dir = os.path.join(params['results_folder'], 'segmentations')
        os.makedirs(output_dir, exist_ok=True)
        for p, id_ in zip(all_preds, all_ids):
            img = convert_to_color(p)
            io.imsave(os.path.join(output_dir, 'inference_tile_{}.png'.format(id_)), img)
            
        with open(f"{params['results_folder']}/training_results.txt", "w") as f:
            f.write("=================== MODEL RESULTS =================== \n \n")
            f.write(f"Model: {params['model']['name']} - Consistency: {params['consistency']} - Loss function: {params['loss']['name']} - Loss weights: {params['loss']['params']['weights']} - Optimizer: {params['optimizer_params']['optimizer']} - Augmentation: {params['augmentation']}\n")
            if params['consistency']:
                f.write(f"Consistency from epoch: {params['consistency_from']} - Adaptive weight: {params['adaptive_weight']} - Noise: {params['noise']} - w_mc: {params['w_mc']} - w_oc: {params['w_oc']}\n")
            f.write("Training time: {:.2f} hours - Num epochs: {} - Best epoch: {} - Best_val_loss: {}\n".format(training_time_hours, epoch, early_stopping.best_epoch, early_stopping.val_loss_min))
            f.write("Evaluation metrics: \n ")
            for info in informations:
                f.write(f"{info}\n")
        
        with open(f"{params['results_folder']}/params.txt", "w", encoding="utf-8") as f:
            for chave, valor in params.items():
                f.write(f"{chave}: {valor}\n")
    
    # transforming metrics for original resolution (if case)
    if (params['dim_original'][0] != params['dim_entrada'][0]) or (params['dim_original'][1] != params['dim_entrada'][1]):
        print(f'Starting process of redimensioned prediction...')

        labels = load_original_test_labels(
                    dir_original_labels=os.path.join(params['root_dir_original'], 'labels_modificadas'),
                    dir_original_labels_txt='/prj/posgrad/lorran/pesquisa/dissertacao/data/test_labels.txt')

        for fd in params['folds']:
            print(f'Fold: {fd}')
            predictions = load_predictions_resized(
                dim_original=params['dim_original'],
                dir_predictions=os.path.join(params['results_path'], fd, 'segmentations'))

            results_folder = os.path.join(params['results_path'], fd, 'original_results')

            accuracy, informations = metrics(
                predictions=np.concatenate([p.ravel() for p in predictions]), 
                gts=np.concatenate([p.ravel() for p in labels]).ravel(),
                label_values=params['classes'],
                params=params,
                all=True,
                filepath=results_folder
            )

            # saving results
            output_dir = os.path.join(results_folder, 'segmentations')
            os.makedirs(output_dir, exist_ok=True)
            for p, id_ in zip(predictions, all_ids):
                img = convert_to_color(p)
                io.imsave(os.path.join(output_dir, 'inference_tile_{}.png'.format(id_)), img)
                
            with open(f"{results_folder}/training_results.txt", "w") as f:
                f.write("=================== MODEL RESULTS =================== \n \n")
                for info in informations:
                    f.write(f"{info}\n")