import warnings
from pathlib import Path
from torch.autograd import Variable
import torch.nn as nn
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm

from semi import contrastive_method
from utils import make_optimizer, make_scheduler, metrics, save_test, grouper, convert_from_color, convert_to_color, count_sliding_window2, sliding_window2, gaussian_kernel, gaussian_square_map 
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

class ProjectionHead(nn.Module):
    def __init__(self, in_channels=2048, out_dim=128):
        super().__init__()
        
        self.proj = nn.Linear(in_channels, out_dim)

    def forward(self, x):
        """
        x: Tensor de shape (B, C, H, W)
        Retorna: Tensor de shape (B, H, W, out_dim)
        """
        # B, C, H, W = x.shape
        # # Reorganiza para (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        # Aplica a projeção em todos os pixels (flatten e projeta)
        x = self.proj(x)  # shape (B, H, W, out_dim)
        
        return x

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
        self.semi = semi
        
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

        # Create optimizer
        self.optimizer = make_optimizer(self.params['optimizer_params'], self.net)
        # Create scheduler
        self.scheduler = make_scheduler(self.params['lrs_params'], self.optimizer) if scheduler else None

        self.last_epoch = 0

        # Load a previously model if it exists
        if cbkp is not None:
            self.load(cbkp)

        Path(os.path.join(self.params['results_folder'])).mkdir(parents=True, exist_ok=True)
        
        # After loading the model, so we can load the net model to copy student model params
        if self.semi:    
            self.teacher_load_student()

            self.semi_loss1 = []
            self.semi_loss2 = []
            self.semi_loss3 = []
            self.semi_loss4 = []
            self.semi_loss5 = []

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
            'iter_': self.iter_,
            'model_id': self.model_id,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        # Salvar métricas extras se consistency estiver ativado
        if getattr(self, "semi_supervised", False):
            checkpoint.update({
                'epoch_supervised_loss': getattr(self, "epoch_supervised_loss", None),
                'epoch_val_losses': getattr(self, "epoch_val_losses", None),
                'epoch_out_consistency_loss': getattr(self, "epoch_out_consistency_loss", None),
                'epoch_multiscale_consistency_loss': getattr(self, "epoch_multiscale_consistency_loss", None),
            })

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
        self.epoch_loss = checkpoint.get('epoch_loss', None)
        self.epoch_acc = checkpoint.get('epoch_acc', None)
        self.iter_ = checkpoint.get('iter_', 0)

        # Restaurar atributos extras se consistency
        if getattr(self, "consistency", False):
            self.epoch_supervised_loss = checkpoint.get('epoch_supervised_loss', None)
            self.epoch_val_losses = checkpoint.get('epoch_val_losses', None)
            self.epoch_out_consistency_loss = checkpoint.get('epoch_out_consistency_loss', None)
            self.epoch_multiscale_consistency_loss = checkpoint.get('epoch_multiscale_consistency_loss', None)

        # Carregar modelo e otimizador
        print("Loading checkpoint model...")
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

    def test(self, test_loader = None, stride = None, window_size = None, batch_size = None, all=False):
        """"
        Test the network on the test set.
        
        Args:
            test_loader: data loader for the test set
            stride: stride for the sliding window
            window_size: size of the sliding window
            batch_size: batch size for the test set
            all: if True, return all predictions and ground truths
        """
    
        test_ld = test_loader if test_loader is not None else self.loader['test']
        assert test_ld is not None, "Test_loader can't be None"

        test_stride = stride if stride is not None else self.params['stride']
        assert test_stride is not None, "Stride not set"

        test_ws = window_size if window_size is not None else self.params['window_size']
        assert test_ws is not None, "Window size not set"

        bs = batch_size if batch_size is not None else self.params['bs']
        bs = bs if bs is not None else 1

        # Get all data in test data loader
        input_ids, label_ids = test_ld.dataset.get_dataset()

        # Use the network on the test set
        test_images = ((1 / 255) * np.asarray(io.imread(id)[:,:,:3], dtype='float32') for id in input_ids)
        test_labels = (np.asarray(io.imread(id)[:,:,:3], dtype='uint8') for id in label_ids)
        eroded_labels = (convert_from_color(io.imread(id)[:,:,:3]) for id in label_ids)
        
        all_preds = []
        all_gts = []
        
        # Switch the network to inference mode
        self.net.eval()
        
        for img, gt, gt_e, image_path in tqdm(zip(test_images, test_labels, eroded_labels, input_ids), total=len(input_ids), leave=False):

            filepath = os.path.split(image_path)[1].split('.')[0]
            Path(os.path.join(self.params['results_folder'], 'inference')).mkdir(parents=True, exist_ok=True)
            
            pred = np.zeros(img.shape[:2] + (self.params['n_classes'],))

            total = count_sliding_window2(img, step=test_stride, window_size=test_ws) // bs
            for i, coords in enumerate(tqdm(grouper(bs, sliding_window2(img, step=test_stride, window_size=test_ws)), total=total, leave=False)):
                
                # # Display in progress results
                # if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                #     _pred = np.argmax(pred, axis=-1) #Usar argmax do pytorch
                #     fig = plt.figure()
                #     fig.add_subplot(1,3,1)
                #     plt.imshow(np.asarray(255 * img, dtype='uint8'))
                #     fig.add_subplot(1,3,2)
                #     plt.imshow(convert_to_color(_pred))
                #     fig.add_subplot(1,3,3)
                #     plt.imshow(gt)
                #     fig.savefig(f"./tmp/test_progress", dpi=fig.dpi, bbox_inches='tight')
                #     plt.close(fig)
                        
                # Build the tensor (ex: 2048px/256px = batch size patches de 256x256)
                with torch.no_grad():
                    image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
                    image_patches = np.asarray(image_patches)
                    #image_patches = Variable(torch.from_numpy(image_patches).cuda())
                    image_patches = Variable(torch.from_numpy(image_patches).to(self.params['device']))
                
                # Do the inference
                outs = self._get_first(self.net(image_patches))
                outs = outs.data.cpu().numpy()
                # trocar pred pela função do pytorch
                
                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1,2,0)) # Usar transpose do Pytorch
                    pred[x:x+w, y:y+h] += out # ponderar essa soma para mais próximo do centro?
                del(outs)

            pred = np.argmax(pred, axis=-1)

            # # Display the result
            # fig = plt.figure(figsize=(30, 10))
            
            # ax1 = fig.add_subplot(1,3,1)
            # ax1.set_title('RGB')
            # plt.imshow(np.asarray(255 * img, dtype='uint8'))

            # ax2 = fig.add_subplot(1,3,2)
            # ax2.set_title('Pred')
            # plt.imshow(convert_to_color(pred))

            # ax3 = fig.add_subplot(1,3,3)
            # ax3.set_title('GT')
            # plt.imshow(gt)
            
            # fig.savefig(os.path.join(self.params['results_folder'], 'inference', filepath), dpi=fig.dpi, bbox_inches='tight')
            # plt.close(fig)

            all_preds.append(pred)
            all_gts.append(gt_e)
        
        # metrics calculation
        accuracy, informations = metrics(
            predictions=np.concatenate([p.ravel() for p in all_preds]), 
            gts=np.concatenate([p.ravel() for p in all_gts]).ravel(),
            label_values=self.params['classes'],
            params=self.params,
            all=all,
            filepath=filepath
        )

        if all:
            save_test(acc=accuracy, all_preds=all_preds, all_gts=all_gts, 
                        path=os.path.join(self.params['results_folder'], 'results.npz'))
            return accuracy, all_preds, all_gts, informations
        else:
            return accuracy

    @staticmethod
    def _get_first (result):
        if isinstance(result, tuple):
            return result[0]
        return result

    def _update_teacher(self, alpha=0.99): # ta fazendo certo?
        """
        Atualiza os pesos da rede professora (teacher) com EMA usando os pesos da estudante (student).
        A chamada deve ser feita a cada batch de treinamento.
        Args:
            alpha (float): Fator de suavização (quanto maior, mais lenta a atualização).
        """
        with torch.no_grad():  
            for t_param, s_param in zip(self.net_teacher.parameters(), self.net.parameters()):
                t_param.data.mul_(alpha).add_((1 - alpha) * s_param.data)
    
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