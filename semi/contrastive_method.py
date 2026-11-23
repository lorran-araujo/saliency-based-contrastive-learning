import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import warnings
import numpy as np

import semi.sw_augmentations as aug
import semi.negatives as neg
import semi.saliency as saliency

class ProjectionHead(nn.Module):
    def __init__(self, in_channels=2048, out_dim=128):
        super().__init__()
        # Camada linear aplicada a cada vetor de feature (por pixel)
        self.proj = nn.Linear(in_channels, out_dim)

    def forward(self, x):
        """
        x: Tensor de shape (B, C, H, W)
        Retorna: Tensor de shape (B, H, W, out_dim)
        """
        B, C, H, W = x.shape
        # Reorganiza para (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        # Aplica a projeção em todos os pixels (flatten e projeta)
        x = self.proj(x)  # shape (B, H, W, out_dim)
        
        return x
    
def batch_normalization_deactivate(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm3d):
            module.eval()

def info_nce_pixel_loss(z_anchor, z_pos, negatives, temperature=0.07):
    """
    Calcula a InfoNCE Loss para um vetor âncora, um positivo e vários negativos.
    z_anchor: (D,)
    z_pos: (D,)
    negatives: (N, D)
    """
    # Normaliza para comparação de similaridade por produto escalar
    z_anchor = F.normalize(z_anchor.unsqueeze(0), dim=1)  # (1, D)
    z_pos    = F.normalize(z_pos.unsqueeze(0), dim=1)      # (1, D)
    negatives = F.normalize(negatives, dim=1)              # (N, D)

    # Similaridades
    pos_sim = torch.matmul(z_anchor, z_pos.T) / temperature  # (1, 1)
    neg_sim = torch.matmul(z_anchor, negatives.T) / temperature  # (1, N)

    logits = torch.cat([pos_sim, neg_sim], dim=1)  # (1, 1+N)
    labels = torch.zeros(1, dtype=torch.long, device=z_anchor.device)  # positivo é índice 0

    loss = F.cross_entropy(logits, labels)
    return loss

# ==============================================================
# My proposed framework TCR-based
# =============================================================

def contrastive_loss_from_negatives_tensor(embeddings_w_proj, embeddings_s_proj, negatives_tensor, N_neg_sample=200, temperature=0.07):
    """
    Calcula a loss contrastiva usando embeddings e tensores de negativos pré-calculados.

    embeddings_w_proj: (B, H, W, D) - âncoras
    embeddings_s_proj: (B, H, W, D) - positivos
    negatives_tensor: (B, H, W, num_negatives, 3) - coordenadas (b_neg, i_neg, j_neg)
    N_neg_sample: número máximo de negativos por âncora
    """
    B, H, W, D = embeddings_s_proj.shape
    total_loss = 0.0
    count = 0

    for b in range(B):
        for i in range(H):
            for j in range(W):
                negs_coord = negatives_tensor[b, i, j]  # (num_negatives, 3)
                
                # Filtra entradas inválidas (-1)
                negs_coord = negs_coord[negs_coord[:,0] >= 0]
                if len(negs_coord) == 0:
                    continue

                # Limita a N_neg_sample
                if len(negs_coord) > N_neg_sample:
                    idx = torch.randperm(len(negs_coord))[:N_neg_sample]
                    negs_coord = negs_coord[idx]

                H_emb, W_emb = embeddings_s_proj.shape[1], embeddings_s_proj.shape[2]

                mask = (
                    (negs_coord[:,0] >= 0) & (negs_coord[:,0] < H_emb) &
                    (negs_coord[:,1] >= 0) & (negs_coord[:,1] < W_emb)
                )

                negs_coord = negs_coord[mask]

                # Pega os embeddings negativos
                negatives = embeddings_s_proj[
                    negs_coord[:,0], negs_coord[:,1], negs_coord[:,2]
                ]  # (N_neg, D)

                # Pega âncora e positivo correspondente
                z_anchor = embeddings_w_proj[b, i, j]  # (D,)
                z_pos    = embeddings_s_proj[b, i, j]  # (D,)

                # Calcula a loss
                loss_ij = info_nce_pixel_loss(z_anchor, z_pos, negatives, temperature=temperature)
                total_loss += loss_ij
                count += 1

    return total_loss / max(count, 1)

def contrastive_loss_from_sampled_anchor(embeddings_w_proj, embeddings_s_proj, anchors, negatives_pairs, temperature=0.07):
    """
    Calcula a loss contrastiva dos embeddings dos vetores ancoras e seus respectivos tensores de negativos pré-calculados.

    Args:
    embeddings_w_proj: tensor com shape (B, H, W, D)
    embeddings_s_proj: tensor com shape (B, H, W, D)
    anchors: tensor com shape (M, 3), onde cada (b,i,j) são as coordenadas píxeis ancoras do embedding
    negatives_tensor: lista com tensores de shape (N, 3) onde cada um se trata dos respectivos N coordenadas negativas para a lista de anchors
    """
    B, H, W, D = embeddings_s_proj.shape
    total_loss = torch.tensor(0.0, device=embeddings_w_proj.device)
    count = 0

    for k, anch_coord in enumerate(anchors):
        b, i, j = anch_coord.tolist()

        negs_coords = negatives_pairs[k] # (num_negatives, 3)

        # ignorando se pixel ancora nao possuir pares negativos
        if negs_coords.numel() == 0:  # tensor vazio
            continue

        # # filtrar coordenadas invalidas
        # mask_valid = (negs_coords[:,0] >= 0)
        # negs_coords = negs_coords[mask_valid]

        # Pega os embeddings negativos
        negatives = embeddings_s_proj[
            negs_coords[:,0], negs_coords[:,1], negs_coords[:,2]
        ]  # (N_neg, D)

        # Pega âncora e positivo correspondente
        z_anchor = embeddings_w_proj[b, i, j]  # (D,)
        z_pos    = embeddings_s_proj[b, i, j]  # (D,)

        # Calcula a loss
        loss_ij = info_nce_pixel_loss(z_anchor, z_pos, negatives, temperature=temperature)
        total_loss += loss_ij
        count += 1

    return total_loss / max(count, 1)

def saliency_contrastive(obj):
    """Treina a rede com as perdas semissupervisionadas habilitadas."""
    train_running_loss = 0.0
    train_running_correct = 0
    train_running_supervised_loss = 0.0
    semi_loss1 = 0.0
    semi_loss2 = 0.0
    semi_loss3 = 0.0
    semi_loss4 = 0.0
    obj.last_epoch += 1

    train_loader = obj.loader['train']
    unlabeled_loader = obj.loader['train_unlabeled']
    steps = min(len(train_loader), len(unlabeled_loader))
    pbar = tqdm(zip(train_loader, unlabeled_loader),
                ncols=150,
                total=steps,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]' 
                )

    counter = 0
    for (inputs, labels), unlabeled_inputs in pbar:
        obj.net.train()
        inputs, labels = obj.prepare([inputs, labels])
        counter += labels.size(0)
        outputs = obj._get_first(obj.net(inputs))
        supervised_loss = obj.criterion(outputs, labels)
        loss = supervised_loss

        unlabeled_inputs = obj.prepare([unlabeled_inputs])
        noisy_inputs = aug.add_noise(unlabeled_inputs, obj.params['noise'])
        obj.net.eval()

        with torch.no_grad():
            teacher_outputs, teacher_embeddings = obj.net_teacher(unlabeled_inputs)

        consistency_cutmix = consistency_grid = consistency = contrastive = 0.0
        if obj.use_consistency:
            cutmix_inputs, lam, coords = aug.cutmix(noisy_inputs, beta=1.0, return_all=True)
            grid_inputs, perms = aug.grid_shuffle(noisy_inputs, grid_size=(4, 4), return_perm=True)
            cutmix_outputs = obj._get_first(obj.net(cutmix_inputs))
            grid_outputs = obj._get_first(obj.net(grid_inputs))
            cutmix_targets = aug.cutmix(teacher_outputs, lm=lam, coords=coords)
            grid_targets = aug.grid_shuffle(teacher_outputs, grid_size=(4, 4), perm=perms)

            cutmix_loss = F.mse_loss(
                F.normalize(cutmix_outputs, dim=1, eps=1e-8),
                F.normalize(cutmix_targets, dim=1, eps=1e-8),
            )
            grid_loss = F.mse_loss(
                F.normalize(grid_outputs, dim=1, eps=1e-8),
                F.normalize(grid_targets, dim=1, eps=1e-8),
            )
            loss = loss + obj.params['w_consistency'] * (cutmix_loss + grid_loss)
            consistency_cutmix = cutmix_loss.item()
            consistency_grid = grid_loss.item()
            consistency = consistency_cutmix + consistency_grid

        if obj.use_contrastive:
            student_outputs, student_embeddings = obj.net(noisy_inputs)
            student_vectors = obj.student_projection(student_embeddings[-1])
            with torch.no_grad():
                teacher_vectors = obj.teacher_projection(teacher_embeddings[-1])

            batch_size, height, width, _ = student_vectors.shape
            positions = batch_size * height * width
            count = min(obj.params['num_vet_cont'], positions)
            flat_indices = np.random.choice(positions, size=count, replace=False)
            batch, offset = divmod(flat_indices, height * width)
            row, col = divmod(offset, width)
            indices = torch.as_tensor(np.stack((batch, row, col), axis=1), device=noisy_inputs.device)

            saliency_maps = saliency.generate_saliency_maps_of_sampled_index(
                obj.net, noisy_inputs, sorted_indices=indices, device=noisy_inputs.device,
            )
            with torch.no_grad():
                anchors, negative_pairs, _ = neg.negative_sample_by_saliency_of_sampled_index(
                    indices, saliency_maps, obj.params['num_ancoras'], student_outputs,
                    device=noisy_inputs.device,
                    top_k_pixels=min(obj.params['topk'], student_outputs.shape[-2] * student_outputs.shape[-1]),
                    min_confidence=obj.params['min_confidence'],
                )
            contrastive_loss = contrastive_loss_from_sampled_anchor(
                teacher_vectors, student_vectors, anchors, negative_pairs,
            )
            loss = loss + obj.params['w_contrastive'] * contrastive_loss
            contrastive = contrastive_loss.item()

        obj.optimizer.zero_grad()
        loss.backward()
        obj.optimizer.step()
        
        # update the weights of teacher network
        obj._update_teacher()

        # define the postfix for the progress bar
        pbar.set_postfix({
                'Epoch': f'{obj.last_epoch:2d}', 
                'Total_Loss': f'{loss.item():.4f}',
                'S_Loss': f'{supervised_loss.item():.4f}',
                'CO_Loss1': f'{consistency_cutmix:.3f}',
                'CO_Loss3': f'{consistency_grid:.3f}',
                'Consistency': f'{consistency:.3f}',
                'Contrastive': f'{contrastive:.3f}'
            })
        
        train_running_supervised_loss += supervised_loss.item()
        train_running_loss += loss.item()
        semi_loss1 += contrastive
        semi_loss2 += consistency
        semi_loss3 += consistency_cutmix
        semi_loss4 += consistency_grid
        preds = outputs.argmax(dim=1)
        train_running_correct += preds.eq(labels).sum().item()
        obj.losses[obj.iter_] = loss.item()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            obj.mean_losses[obj.iter_] = np.mean(obj.losses[max(0,obj.iter_-100):obj.iter_])
            
        obj.iter_ += 1
    
    # evaluate the model on validation dataset
    if 'val' in obj.loader and obj.loader['val'] is not None:
        obj.net.eval()
        val_running_loss = 0.0
        val_loader = obj.loader['val']  

        with torch.no_grad():  
            for val_batch_id, (val_inputs, val_labels) in enumerate(val_loader):
                val_inputs, val_labels = obj.prepare([val_inputs, val_labels])
                val_outputs = obj.net(val_inputs)
                val_loss = obj.criterion(obj._get_first(val_outputs), val_labels)
                val_running_loss += val_loss.item()

        #val_running_loss /= (len(obj.loader['val']))  # Média correta por batches
        val_running_loss /= (val_batch_id+1)
        obj.epoch_val_losses.append(val_running_loss)

    if obj.scheduler is not None:
        obj.scheduler.step()

    # saving consistency's loss
    obj.semi_loss1.append(semi_loss1 / steps)
    obj.semi_loss2.append(semi_loss2 / steps)
    obj.semi_loss3.append(semi_loss3 / steps)
    obj.semi_loss4.append(semi_loss4 / steps)
    obj.semi_loss5.append(0.0)

    # Loss and accuracy for the complete epoch.
    obj.epoch_supervised_loss.append(train_running_supervised_loss / steps)
    obj.epoch_loss.append(train_running_loss / steps)
    obj.epoch_acc.append(100. * train_running_correct / counter)

def only_C(obj):
    """"
    Train the network with the proposed contrastive training.
    """
    train_running_loss = 0.0
    train_running_correct = 0
    train_running_supervised_loss = 0.0
    out_consistency_running_loss = 0.0
    multiscale_consistency_running_loss = 0.0

    obj.last_epoch = obj.scheduler.last_epoch + 1        

    loaders = zip(obj.loader['train'], obj.loader['train_unlabeled']) # tomar cuidado para ter o mesmo número de batches nos dois loaders

    pbar = tqdm(loaders,         
                ncols=150,       
                total=min(len(obj.loader['train']), len(obj.loader['train_unlabeled'])),             
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]' 
                )
    
    #obj.net.train()
    
    counter = 0
    for (inputs, labels), unlabeled_inputs in pbar:
        
        # labeled training
        inputs, labels = obj.prepare([inputs, labels]) 
        counter+=labels.size(0)
            
        outputs = obj.net(inputs) 
        supervised_loss = obj.criterion(obj._get_first(outputs), labels)

        # unlabeled training
        unlabeled_inputs = obj.prepare([unlabeled_inputs])

        obj.net.eval() 
    
        # # foward pass in teacher network
        with torch.no_grad():
            unlabeled_outputs_t, unlabeled_embeddings_t = obj.net_teacher(unlabeled_inputs) # teacher com imagens não rotuladas sem ruido

        # # weak noise
        unlabeled_inputs_gaussian_noise = aug.add_noise(unlabeled_inputs, obj.params['noise'])

        # # foward pass in student network
        unlabeled_outputs_s, unlabeled_embeddings_s = obj.net(unlabeled_inputs_gaussian_noise) # output only with gaussian to student
        #obj.net.train()

        # # # contrastive loss

        if obj.last_epoch >= obj.params['contrastive_from']:
            
            # # select inference of only output layer of embedding
            embeddings_t = unlabeled_embeddings_t[-1] # shape (B, F, H, W) - teacher embedding
            embeddings_s = unlabeled_embeddings_s[-1] # shape (B, F, H, W) - student embedding
            
            # Fazendo embedding para menos filtros
            proj_head_teacher = ProjectionHead(in_channels=embeddings_t.shape[1], out_dim=128).to(obj.params['device']) 
            proj_head_student = ProjectionHead(in_channels=embeddings_s.shape[1], out_dim=128).to(obj.params['device'])

            embeddings_t_proj = proj_head_teacher(embeddings_t)   # (B, H, W, C)
            embeddings_s_proj = proj_head_student(embeddings_s) # (B, H, W, C)
            
            # # sorting N random indices
            N = obj.params['num_vet_cont']
            B, H, W, _ = embeddings_s_proj.shape
            flat_idx = np.random.choice(B*H*W, size=N, replace=False)
            
            b, hw = divmod(flat_idx, H * W)
            h, w = divmod(hw, W)
            sorted_indices = list(zip(b, h, w))
            sorted_indices = torch.tensor(sorted_indices, dtype=torch.long, device=obj.params['device'])

            # # calculating saliency maps
            saliency_maps = saliency.generate_saliency_maps_of_sampled_index(obj.net, unlabeled_inputs_gaussian_noise, sorted_indices=sorted_indices,device=obj.params['device'])
            
            M_ancoras = obj.params['num_ancoras']    
            anchors, negative_pairs,_ = neg.negative_sample_by_saliency_of_sampled_index(sorted_indices, saliency_maps, M_ancoras, unlabeled_outputs_s, device=obj.params['device'], top_k_pixels=obj.params['topk'], min_confidence=obj.params['min_confidence'])
        
            loss_contrastive = contrastive_loss_from_sampled_anchor(embeddings_t_proj, embeddings_s_proj, anchors, negative_pairs)
            
        else:
            loss_contrastive = torch.tensor(0)
        
        # FINAL LOSS
        loss = supervised_loss + 0.1*loss_contrastive

        obj.optimizer.zero_grad()
        loss.backward()
        obj.optimizer.step()
        
        # update the weights of teacher network
        obj._update_teacher()

        # define the postfix for the progress bar
        pbar.set_postfix({
                'Epoch': f'{obj.last_epoch:2d}', 
                'Total_Loss': f'{loss.item():.4f}',
                'S_Loss': f'{supervised_loss.item():.4f}',
                #'CO_Loss1': f'{loss_consistency_out1.item():.3f}',
                #'CO_Loss2': f'{loss_consistency_out2.item():.2f}',
                #'CO_Loss3': f'{loss_consistency_out3.item():.3f}',
                #'Consistency': f'{loss_consistency_out.item():.3f}',
                'Contrastive': f'{loss_contrastive.item():.3f}'
            })
        
        train_running_supervised_loss += supervised_loss.item()
        train_running_loss += loss.item()
        outputs = obj._get_first(outputs)
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += preds.eq(labels).sum().item()
        obj.losses[obj.iter_] = loss.item()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            obj.mean_losses[obj.iter_] = np.mean(obj.losses[max(0,obj.iter_-100):obj.iter_])
            
        obj.iter_ += 1
        del(inputs, labels, unlabeled_inputs, supervised_loss, loss)#loss_consistency_out, loss)#, loss_consistency_multiscale)
    
    # evaluate the model on validation dataset
    if 'val' in obj.loader and obj.loader['val'] is not None:
        obj.net.eval()
        val_running_loss = 0.0
        val_loader = obj.loader['val']  

        with torch.no_grad():  
            for val_batch_id, (val_inputs, val_labels) in enumerate(val_loader):
                val_inputs, val_labels = obj.prepare([val_inputs, val_labels])
                val_outputs = obj.net(val_inputs)
                val_loss = obj.criterion(obj._get_first(val_outputs), val_labels)
                val_running_loss += val_loss.item()

        #val_running_loss /= (len(obj.loader['val']))  # Média correta por batches
        val_running_loss /= (val_batch_id+1)
        obj.epoch_val_losses.append(val_running_loss)

    if obj.scheduler is not None:
        obj.scheduler.step()

    # saving consistency's loss
    obj.epoch_out_consistency_loss.append(out_consistency_running_loss / len(obj.loader['train_unlabeled']))
    obj.epoch_multiscale_consistency_loss.append(multiscale_consistency_running_loss / len(obj.loader['train_unlabeled']))

    # Loss and accuracy for the complete epoch.
    obj.epoch_supervised_loss.append(train_running_supervised_loss/len(obj.loader['train']))        
    obj.epoch_loss.append(train_running_loss/len(obj.loader['train']))
    obj.epoch_acc.append(100. * train_running_correct / counter)

def TCR_PC2Seg(obj):
    """"
    Train the network.
    """
    train_running_loss = 0.0
    train_running_correct = 0
    train_running_supervised_loss = 0.0
    semi_loss1 = 0
    semi_loss2 = 0
    semi_loss3 = 0
    semi_loss4 = 0
    semi_loss5 = 0

    obj.last_epoch = obj.scheduler.last_epoch + 1        

    loaders = zip(obj.loader['train'], obj.loader['train_unlabeled']) # tomar cuidado para ter o mesmo número de batches nos dois loaders

    pbar = tqdm(loaders,         
                ncols=150,       
                total=min(len(obj.loader['train']), len(obj.loader['train_unlabeled'])),             
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]' 
                )

    #obj.net.train()

    # training
    counter = 0
    for (inputs, labels), unlabeled_inputs in pbar:
        
        # labeled training
        inputs, labels = obj.prepare([inputs, labels]) 
        counter+=labels.size(0)
            
        outputs = obj.net(inputs) 
        supervised_loss = obj.criterion(obj._get_first(outputs), labels)

        # unlabeled training
        unlabeled_inputs = obj.prepare([unlabeled_inputs])

        # consistency TCR train
        obj.net.eval()
        
        with torch.no_grad():
            unlabeled_outputs_t, unlabeled_embeddings_t = obj.net_teacher(unlabeled_inputs)

        unlabeled_inputs = aug.add_noise(unlabeled_inputs, obj.params['noise'])
        unlabeled_outputs_s, unlabeled_embeddings_s = obj.net(unlabeled_inputs)

        noisy_images1, lam, coords = aug.cutmix(unlabeled_inputs, beta=1.0, return_all=True)
        #noisy_images2, params = aug.random_affine_transform(unlabeled_inputs, return_params=True)
        noisy_images3, perms = aug.grid_shuffle(unlabeled_inputs, grid_size=(4,4), return_perm=True)
                
        unlabeled_outputs1_s = obj._get_first(obj.net(noisy_images1))
        #unlabeled_outputs2_s = obj._get_first(obj.net(noisy_images2))
        unlabeled_outputs3_s = obj._get_first(obj.net(noisy_images3))

        unlabeled_outputs1_t = aug.cutmix(unlabeled_outputs_t, lm=lam, coords=coords)
        #unlabeled_outputs2_t = aug.random_affine_transform(unlabeled_outputs_t, params=params)
        unlabeled_outputs3_t = aug.grid_shuffle(unlabeled_outputs_t, grid_size=(4,4), perm=perms)
        
        loss_consistency_out1 = F.mse_loss(F.normalize(unlabeled_outputs1_s, dim=1, eps=1e-8), F.normalize(unlabeled_outputs1_t, dim=1, eps=1e-8))
        #loss_consistency_out2 = F.mse_loss(F.normalize(unlabeled_outputs2_s, dim=1, eps=1e-8), F.normalize(unlabeled_outputs2_t, dim=1, eps=1e-8))
        loss_consistency_out3 = F.mse_loss(F.normalize(unlabeled_outputs3_s, dim=1, eps=1e-8), F.normalize(unlabeled_outputs3_t, dim=1, eps=1e-8))

        loss_consistencyTCR = loss_consistency_out1 + loss_consistency_out3

        # CONTRASTIVE LOSS
        
        # # select inference of only output layer of embedding
        embeddings_t = unlabeled_embeddings_t[-1] # shape (B, F, H, W) - teacher embedding
        embeddings_s = unlabeled_embeddings_s[-1] # shape (B, F, H, W) - student embedding
        
        # Fazendo embedding para menos filtros
        proj_head_teacher = ProjectionHead(in_channels=embeddings_t.shape[1], out_dim=128).to(obj.params['device']) 
        proj_head_student = ProjectionHead(in_channels=embeddings_s.shape[1], out_dim=128).to(obj.params['device'])

        embeddings_w_proj = proj_head_teacher(embeddings_t)   # (B, H, W, C)
        embeddings_s_proj = proj_head_student(embeddings_s) # (B, H, W, C)

        total_loss = 0.0
        count = 0
        
        B, H, W , _ = embeddings_w_proj.shape        
        N = obj.params['num_ancoras']  # quantidade de índices desejados
        flat_indices = np.random.choice(B * H * W, size=N, replace=False)

        b_in = flat_indices // (H * W)
        h_in = (flat_indices % (H * W)) // W
        w_in = flat_indices % W

        for b, i, j in zip(b_in, h_in, w_in):
            z_anchor = embeddings_w_proj[b, i, j]   # (128,)
            z_pos    = embeddings_s_proj[b, i, j] # (128,)
            #negatives = neg.sampling_negative_pairs_uniform(embeddings_s_proj, N=obj.params['num_vet_cont'], i_idx=(b,i,j))   # (N, 128)
            negatives = neg.sampling_negative_pseudolabels_from_others_images(embeddings_s_proj, unlabeled_outputs_t, N=obj.params['num_vet_cont'], i_idx=(b,i,j))

            loss_ij = info_nce_pixel_loss(z_anchor, z_pos, negatives)
            total_loss += loss_ij
            count += 1
        loss_contrastive = total_loss / count
        
        # Loss final
        loss = supervised_loss + 0.1*loss_contrastive + loss_consistencyTCR

        obj.optimizer.zero_grad()
        loss.backward()
        obj.optimizer.step()

        # update the weights of teacher network
        obj._update_teacher()

        # define the postfix for the progress bar
        pbar.set_postfix({
                'Epoch': f'{obj.last_epoch:2d}', 
                'Total_Loss': f'{loss.item():.4f}',
                'S_Loss': f'{supervised_loss.item():.4f}',
                'Constrastive': f'{loss_contrastive.item():.4f}',
                #'Consistency': f'{loss_consistency.item():.4f}',
                'TCR': f'{loss_consistencyTCR.item():.4f}',
            })
        
        train_running_supervised_loss += supervised_loss.item()
        train_running_loss += loss.item()
        semi_loss1 += loss_contrastive.item()
        semi_loss2 += loss_consistencyTCR.item()
        semi_loss3 = loss_consistency_out1.item()
        semi_loss4 += loss_consistency_out3.item()
        outputs = obj._get_first(outputs)
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += preds.eq(labels).sum().item()
        obj.losses[obj.iter_] = loss.item()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            obj.mean_losses[obj.iter_] = np.mean(obj.losses[max(0,obj.iter_-100):obj.iter_])
            
        obj.iter_ += 1
        del(inputs, labels, unlabeled_inputs, supervised_loss, loss_contrastive, loss, loss_ij) #loss_consistency
    
    # evaluate the model on validation dataset
    if 'val' in obj.loader and obj.loader['val'] is not None:
        obj.net.eval()
        val_running_loss = 0.0
        val_loader = obj.loader['val']  

        with torch.no_grad():  
            for val_batch_id, (val_inputs, val_labels) in enumerate(val_loader):
                val_inputs, val_labels = obj.prepare([val_inputs, val_labels])
                val_outputs = obj.net(val_inputs)
                val_loss = obj.criterion(obj._get_first(val_outputs), val_labels)
                val_running_loss += val_loss.item()

        #val_running_loss /= (len(obj.loader['val']))  # Média correta por batches
        val_running_loss /= (val_batch_id+1)
        obj.epoch_val_losses.append(val_running_loss)

    if obj.scheduler is not None:
        obj.scheduler.step()

    # saving consistency's loss
    obj.semi_loss1.append(semi_loss1 / len(obj.loader['train_unlabeled']))
    obj.semi_loss2.append(semi_loss2 / len(obj.loader['train_unlabeled']))
    obj.semi_loss3.append(semi_loss3 / len(obj.loader['train_unlabeled']))
    obj.semi_loss4.append(semi_loss4 / len(obj.loader['train_unlabeled']))
    obj.semi_loss5.append(semi_loss5 / len(obj.loader['train_unlabeled']))

    # Loss and accuracy for the complete epoch.
    obj.epoch_supervised_loss.append(train_running_supervised_loss/len(obj.loader['train']))        
    obj.epoch_loss.append(train_running_loss/len(obj.loader['train']))
    obj.epoch_acc.append(100. * train_running_correct / counter)
