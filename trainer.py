import warnings
from copy import deepcopy
from pathlib import Path
import torch.nn as nn
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm

from semi import contrastive_method
from utils import (
    calculate_cm,
    convert_from_color,
    count_sliding_window2,
    grouper,
    load_icmbio_label,
    make_optimizer,
    make_scheduler,
    metrics_from_confusion_matrix,
    palette,
    resize_label_mask,
    save_test,
    sliding_window2,
)
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

class Trainer():
    """
    Class to train and evaluate the network.
    """
    def __init__(self, net, params, loader=None, net_teacher=None, semi=False, scheduler = True, cbkp = None,):
        """
        Args:
            net: network to train
            loader: data loader (train, test, unlabeled)
            params: parameters for training
            consistency: if True, use consistency loss
            scheduler: if True, use learning rate scheduler
            cbkp: checkpoint to load
        """
        
        self.net = net
        self.net_teacher = net_teacher
        self.loader = loader
        self.params = params
        self.iter_ = 0
        self.epoch_loss = []
        self.epoch_val_losses = []
        self.epoch_acc = []
        self.accuracies = []
        self.losses = np.zeros(1000000)
        self.mean_losses = np.zeros(100000000)
        self.print_each = params['print_each'] or 100 
        self.use_consistency = semi and params.get('consistency', True)
        self.use_contrastive = semi and params.get('contrastive', True)
        self.semi = self.use_consistency or self.use_contrastive
        
        self.epoch_supervised_loss = []

        # # Weights for class balancing
        self.weight_cls = self.prepare([self.params['weights']])

        if self.params['loss']['name'] == 'CROSS_ENTROPY':
            #self.criterion = nn.CrossEntropyLoss(weight=self.weight_cls[0])
            self.criterion = nn.CrossEntropyLoss(weight=self.weight_cls)
        elif self.params['loss']['name'] == 'FOCAL_LOSS':
            self.criterion = FocalLoss(mode='multiclass', alpha=self.params['loss']['params']['alpha'], gamma=self.params['loss']['params']['gamma'], reduction='mean')
        else:
            raise Exception('Loss function not implemented')
        
        # Define an id to a trained model. Use the number of seconds since 1970
        time_ = str(time.time())
        time_ = time_.replace(".", "")
        self.model_id = time_

        self.student_projection = None
        self.teacher_projection = None
        trainable = self.net
        if self.use_contrastive:
            reference = next(self.net.parameters())
            self.student_projection = contrastive_method.ProjectionHead(self.net.contrastive_channels)
            self.student_projection.to(device=reference.device, dtype=reference.dtype)
            self.teacher_projection = deepcopy(self.student_projection).requires_grad_(False).eval()
            trainable = nn.ModuleList([self.net, self.student_projection])

        self.optimizer = make_optimizer(self.params['optimizer_params'], trainable)
        # Create scheduler
        self.scheduler = make_scheduler(self.params['lrs_params'], self.optimizer) if scheduler else None

        self.last_epoch = 0

        Path(os.path.join(self.params['results_folder'])).mkdir(parents=True, exist_ok=True)

        if self.semi:
            self.teacher_load_student()

            self.semi_loss1 = []
            self.semi_loss2 = []
            self.semi_loss3 = []
            self.semi_loss4 = []
            self.semi_loss5 = []

        if cbkp is not None:
            self.load(cbkp)

    def teacher_load_student(self):
        if self.net_teacher is None:
            raise Exception('Teacher network not set.')

        self.net_teacher.load_state_dict(self.net.state_dict())
        for param in self.net_teacher.parameters():
            param.requires_grad_(False)
        self.net_teacher.eval()

    def save(self, path=None):
        """
        Save the model.

        Args:
            path (str, optional): path to the model checkpoint
        """
        if path is None:
            path = os.path.join(
                self.params['results_folder'],
                f"{self.model_id}_model_final.pth.tar"
            )

        checkpoint = {
            'epoch': self.last_epoch,
            'losses': self.losses,
            'mean_losses': self.mean_losses,
            'epoch_loss': self.epoch_loss,
            'epoch_acc': self.epoch_acc,
            'epoch_supervised_loss': self.epoch_supervised_loss,
            'epoch_val_losses': self.epoch_val_losses,
            'iter_': self.iter_,
            'model_id': self.model_id,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        if self.semi:
            checkpoint['teacher_state_dict'] = self.net_teacher.state_dict()
        if self.use_contrastive:
            checkpoint['student_projection_state_dict'] = self.student_projection.state_dict()
            checkpoint['teacher_projection_state_dict'] = self.teacher_projection.state_dict()

        # Salvar estado do scheduler, se existir
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"Model saved at {path}")

    def load(self, path):
        """
        Load a previously trained model.

        Args:
            path (str): path to the model checkpoint
        """
        assert os.path.exists(path), f"{path} can't be opened"

        checkpoint = torch.load(path, map_location=self.params.get("device", "cpu"))

        # Restaurar atributos principais
        self.last_epoch = checkpoint.get('epoch', 0)
        self.model_id = checkpoint.get('model_id', "unknown_model")
        self.losses = checkpoint.get('losses', [])
        self.mean_losses = checkpoint.get('mean_losses', [])
        self.epoch_loss = checkpoint.get('epoch_loss', [])
        self.epoch_acc = checkpoint.get('epoch_acc', [])
        self.iter_ = checkpoint.get('iter_', 0)

        missing_history = [float('nan')] * len(self.epoch_loss)
        self.epoch_supervised_loss = checkpoint.get('epoch_supervised_loss', missing_history.copy())
        self.epoch_val_losses = checkpoint.get('epoch_val_losses', missing_history)

        # Carregar modelo e otimizador
        print("Loading checkpoint model...")
        self.net.load_state_dict(checkpoint['model_state_dict'])
        if self.semi:
            if 'teacher_state_dict' in checkpoint:
                self.net_teacher.load_state_dict(checkpoint['teacher_state_dict'])
            else:
                self.teacher_load_student()

        optimizer_state = checkpoint['optimizer_state_dict']
        if self.use_contrastive:
            if 'student_projection_state_dict' in checkpoint:
                self.student_projection.load_state_dict(checkpoint['student_projection_state_dict'])
                self.teacher_projection.load_state_dict(checkpoint['teacher_projection_state_dict'])
            else:
                # Os checkpoints antigos só têm os parâmetros da rede no otimizador.
                saved_params = optimizer_state['param_groups'][0]['params']
                current_params = self.optimizer.state_dict()['param_groups'][0]['params']
                saved_params.extend(current_params[len(saved_params):])
                self.teacher_projection.load_state_dict(self.student_projection.state_dict())
        self.optimizer.load_state_dict(optimizer_state)

        # Carregar scheduler se salvo
        if self.scheduler is not None:
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # else:
            #     # retrocompatibilidade: avança epochs
            #     for _ in range(0, self.last_epoch):
            #         self.scheduler.step()
        
    def prepare(self, l, non_blocking=True):
        """
        Prepare input and labels for training sending to the GPU.
        
        Args:
            l: list of tensors ou apenas um tensor
            non_blocking: se True, a cópia para a GPU será assíncrona
        """

        #device = torch.device('cpu' if self.params['cpu'] is not None else 'cuda') 
        device = torch.device('cpu' if self.params['cpu'] is not None else self.params['device']) 

        def _prepare(tensor: torch.Tensor):
            if self.params['precision'] == 'half':
                tensor = tensor.half()
            return tensor.to(device, non_blocking=non_blocking)

        if isinstance(l, list):
            prepped = [_prepare(_l) for _l in l]
            return prepped[0] if len(prepped) == 1 else prepped
        else:
            return _prepare(l)

    @torch.inference_mode()
    def test(self, test_loader=None, stride=None, window_size=None, batch_size=None, all=False,
             original_label_files=None):
        """Run sliding-window inference and evaluate the resulting masks."""
        test_ld = test_loader if test_loader is not None else self.loader['test']
        test_stride = stride if stride is not None else self.params.get('test_stride', self.params.get('stride'))
        test_ws = window_size if window_size is not None else self.params['window_size']
        bs = batch_size if batch_size is not None else self.params['bs'] or 1

        image_files, label_files = test_ld.dataset.get_dataset()
        if original_label_files is not None:
            label_files = original_label_files

        output_dir = Path(self.params['results_folder'])
        inference_dir = output_dir / 'inference'
        inference_dir.mkdir(parents=True, exist_ok=True)
        classes = self.params['classes']
        class_ids = range(len(classes))
        confusion_matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
        all_preds = []
        all_gts = []
        self.net.eval()

        pairs = zip(image_files, label_files, strict=True)
        for image_file, label_file in tqdm(pairs, total=len(image_files), leave=False):
            image = np.asarray(io.imread(image_file)[:, :, :3], dtype='float32') / 255
            if original_label_files is not None:
                label = resize_label_mask(load_icmbio_label(label_file), image.shape[:2][::-1])
            elif len(classes) == len(palette):
                label = load_icmbio_label(label_file)
            else:
                label = convert_from_color(io.imread(label_file)[:, :, :3])
            if label.shape != image.shape[:2]:
                raise ValueError(f'Label {label_file} has shape {label.shape}, expected {image.shape[:2]}.')

            scores = np.zeros(image.shape[:2] + (self.params['n_classes'],))
            windows = sliding_window2(image, step=test_stride, window_size=test_ws)
            total_windows = count_sliding_window2(image, step=test_stride, window_size=test_ws)
            total_batches = (total_windows + bs - 1) // bs
            for coords in tqdm(grouper(bs, windows), total=total_batches, leave=False):
                patches = np.stack([image[x:x+w, y:y+h].transpose(2, 0, 1) for x, y, w, h in coords])
                inputs = torch.from_numpy(patches).to(self.params['device'])
                outputs = self._get_first(self.net(inputs)).cpu().numpy()
                for patch, (x, y, w, h) in zip(outputs, coords):
                    scores[x:x+w, y:y+h] += patch.transpose(1, 2, 0)

            prediction = scores.argmax(axis=-1)
            confusion_matrix += calculate_cm(prediction.ravel(), label.ravel(), class_ids)
            if all:
                all_preds.append(prediction)
                all_gts.append(label)

        accuracy, information = metrics_from_confusion_matrix(
            confusion_matrix, classes, output_dir if all else inference_dir,
        )
        if all:
            save_test(acc=accuracy, all_preds=all_preds, all_gts=all_gts, path=str(output_dir / 'results.npz'))
            return accuracy, all_preds, all_gts, information
        return accuracy

    @staticmethod
    def _get_first (result):
        if isinstance(result, tuple):
            return result[0]
        return result

    @torch.no_grad()
    def _update_teacher(self, alpha=0.99):
        """
        Atualiza os pesos da rede professora (teacher) com EMA usando os pesos da estudante (student).
        A chamada deve ser feita a cada batch de treinamento.
        Args:
            alpha (float): Fator de suavização (quanto maior, mais lenta a atualização).
        """
        for teacher, student in zip(self.net_teacher.parameters(), self.net.parameters()):
            teacher.lerp_(student, 1 - alpha)
        for teacher, student in zip(self.net_teacher.buffers(), self.net.buffers()):
            teacher.copy_(student)
        if self.use_contrastive:
            for teacher, student in zip(self.teacher_projection.parameters(), self.student_projection.parameters()):
                teacher.lerp_(student, 1 - alpha)
    
    def _train(self):
        """"
        Train the network with supervised manner.
        """
        train_running_loss = 0.0
        train_running_correct = 0
        train_running_supervised_loss = 0.0

        self.last_epoch = self.scheduler.last_epoch + 1        

        pbar = tqdm(self.loader['train'],         
                    ncols=150,                    
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]' 
)
        self.net.train()

        # training
        counter = 0
        for batch_id, (inputs, labels) in enumerate(pbar):
            
            # labeled training
            inputs, labels = self.prepare([inputs, labels]) 
            counter+=labels.size(0)
                
            outputs = self.net(inputs) 
            loss = self.criterion(self._get_first(outputs), labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.set_postfix({
                'Epoch': self.last_epoch, 
                'Loss': loss.item()
            })
            
            train_running_supervised_loss += loss.item()
            train_running_loss += loss.item()
            outputs = self._get_first(outputs)
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += preds.eq(labels).sum().item()
            self.losses[self.iter_] = loss.item()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.mean_losses[self.iter_] = np.mean(self.losses[max(0,self.iter_-100):self.iter_])
                
            self.iter_ += 1
            del(inputs, labels, loss)

        # evaluate the model on validation dataset
        if 'val' in self.loader and self.loader['val'] is not None:
            self.net.eval()
            val_running_loss = 0.0
            val_loader = self.loader['val']  

            with torch.no_grad():  
                for val_batch_id, (val_inputs, val_labels) in enumerate(val_loader):
                    val_inputs, val_labels = self.prepare([val_inputs, val_labels])
                    val_outputs = self.net(val_inputs)
                    val_loss = self.criterion(self._get_first(val_outputs), val_labels)
                    val_running_loss += val_loss.item()

            #val_running_loss /= (len(self.loader['val']))  # Média correta por batches
            val_running_loss /= (val_batch_id+1)
            self.epoch_val_losses.append(val_running_loss)
        
        if self.scheduler is not None:
            self.scheduler.step()

        # Loss and accuracy for the complete epoch.
        self.epoch_loss.append(train_running_loss/len(self.loader['train']))
        self.epoch_acc.append(100. * train_running_correct / counter)
        self.epoch_supervised_loss.append(train_running_supervised_loss/len(self.loader['train']))

    def train(self):
        """"
        Train the network.
        """
        if self.semi:
            
            method = self.params['semi_method']

            if method == 'Our':
                contrastive_method.saliency_contrastive(self)
            else:
                print(f'Semi-supervised method {self.params["semi_method"]} not implemented.')
                raise ValueError
        else:
            self._train()

        fig = plt.figure()
        
        #plt.plot(np.linspace(1, len(self.epoch_loss), len(self.epoch_loss)).astype(int), self.epoch_loss, linestyle='--', color='blue', label='Total Loss', marker='o')
        plt.plot(np.linspace(1, len(self.epoch_supervised_loss), len(self.epoch_supervised_loss)).astype(int), self.epoch_supervised_loss, linestyle='--', color='blue', label='Supervised Loss', marker='^')
        plt.plot(np.linspace(1, len(self.epoch_loss), len(self.epoch_loss)).astype(int), self.epoch_val_losses, linestyle='--', color='red', label='Val Loss', marker='v')
        
        plt.grid(
            True,
            which='both',      # Aplica a ambos eixos major e minor
            axis='both',       # Habilita grid para x e y
            linestyle=':',     # Estilo pontilhado
            linewidth=0.5,     # Espessura fina
            alpha=0.5,         # Transparência
            color='gray'       # Cor cinza
            )
        
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Train Losses x Val Loss')
        fig.savefig(os.path.join(self.params['results_folder'], 'train_epoch_loss'), dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)

        self.__save_plots()

    def __save_plots(self):
        # loss plots
        fig = plt.figure()
        plt.plot(np.linspace(1, len(self.epoch_loss), len(self.epoch_loss)).astype(int), self.epoch_loss, 'ro-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Train Epoch/Loss')
        fig.savefig(os.path.join(self.params['results_folder'], 'train_epoch_loss_curve'), dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)
        
        # acc plots
        fig = plt.figure()
        plt.plot(np.linspace(1, len(self.epoch_acc), len(self.epoch_acc)).astype(int), self.epoch_acc, 'bo-')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title('Train Epoch/Acc')
        fig.savefig(os.path.join(self.params['results_folder'], 'train_epoch_acc_curve'), dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0, path='best_model.pt'):
        """
        Args:
            patience (int): Quantas épocas esperar após a última melhora.
            verbose (bool): Se True, imprime mensagens de melhoria.
            delta (float): Mínima diferença para considerar como melhoria.
            path (str): Caminho para salvar o melhor modelo.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Salva o modelo quando há melhora na perda de validação.'''
        if self.verbose:
            print(f'Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.best_epoch = epoch
