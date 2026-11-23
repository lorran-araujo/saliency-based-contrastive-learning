import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
from pathlib import Path
from PIL import Image

import sys
sys.path.append('./transformation')
from models import ConvolutionalAutoencoder

def original_test_label_paths(image_files, label_files, original_labels_dir):
    """Resolve original masks in test image order using the split manifests."""
    if not image_files or len(image_files) != len(label_files):
        raise ValueError("The test split must contain one label per image.")

    image_ids = [Path(path).stem for path in image_files]
    label_names = [Path(path).name for path in label_files]
    labels_by_id = {Path(name).stem: name for name in label_names}
    if len(set(image_ids)) != len(image_ids) or len(labels_by_id) != len(label_names):
        raise ValueError("The test split contains duplicate image or label identifiers.")

    # Use tile identifiers when the masks and images share a naming convention.
    # Distinct naming conventions use the explicit pairs in the split manifests.
    if set(image_ids) == labels_by_id.keys():
        label_names = [labels_by_id[tile_id] for tile_id in image_ids]
    else:
        for tile_id, label_name in zip(image_ids, label_names):
            label_id = Path(label_name).stem
            if label_id != tile_id and (tile_id in labels_by_id or label_id in image_ids):
                raise ValueError(f"Test manifests disagree on the pairing for tile {tile_id}.")

    label_paths = [Path(original_labels_dir) / name for name in label_names]
    for path in label_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Original test label not found: {path}")
    return label_paths


def iter_original_test_pairs(predictions, image_files, original_label_files, dim_original):
    """Yield each tile ID, resized prediction and original-resolution label."""
    if len(predictions) != len(image_files) or len(predictions) != len(original_label_files):
        raise ValueError("Original-resolution evaluation requires a prediction and label for each test image.")

    original_size = tuple(dim_original)
    for prediction, image_path, label_path in zip(predictions, image_files, original_label_files):
        label = load_icmbio_label(label_path)
        label_size = (label.shape[1], label.shape[0])
        if label_size != original_size:
            raise ValueError(f"Original label {label_path} has size {label_size}, expected {original_size}.")

        yield Path(image_path).stem, resize_label_mask(prediction, original_size), label

def gaussian_square_map(size, sigma=0.5):
    """
    Gera um mapa de ponderação quadrado (L-inf) baseado em função gaussiana.

    Args:
        size (int): tamanho do lado do mapa (size x size).
        sigma (float): controla a suavidade da transição.

    Returns:
        np.ndarray: mapa 2D normalizado em [0,1].
    """
    # Coordenadas normalizadas no intervalo [-1, 1]
    coords = np.linspace(-1, 1, size)
    xv, yv = np.meshgrid(coords, coords, indexing="xy")

    # Distância L-inf (Chebyshev): max(|x|, |y|)
    d_inf = np.maximum(np.abs(xv), np.abs(yv))

    # Mapa gaussiano invertido (cresce até a borda)
    w = np.exp(-(d_inf**2) / (2 * sigma**2))

    # Normaliza para [0,1]
    #w = (w - w.min()) / (w.max() - w.min())
    return w

def compute_iou(preds, labels, num_classes):
    """
    preds: tensor (N, H, W) com rótulos preditos
    labels: tensor (N, H, W) com rótulos verdadeiros
    num_classes: número total de classes
    """
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        label_inds = (labels == cls)

        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()

        if union == 0:
            ious.append(float("nan"))  # ignora se a classe não aparece
        else:
            ious.append(intersection / union)
    return ious

def gaussian_kernel(w, h, sigma=0.5):
    """Cria um kernel gaussiano normalizado para ponderação."""
    # Cria coordenadas normalizadas em [-1,1]
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    d = np.sqrt(xx**2 + yy**2)
    gauss = np.exp(- (d**2) / (2 * sigma**2))
    gauss /= gauss.max()  # normaliza para [0,1]
    return gauss


def is_save_epoch(params, epoch, ignore_epoch=0):
    return params['save_epoch'] is not None and epoch % params['save_epoch'] == 0 and epoch != ignore_epoch
    
class LossFN:
    CROSS_ENTROPY = 'cross_entropy'
    FOCAL_LOSS = 'focal_loss'

class ModelChooser:
    SEGNET_MODIFICADA = 'segnet_modificada'
    UNET = 'unet'
    LANET = 'lanet'

def weights_calculator_loss(params, train_labels):
    try:
        if params['loss']['name'] == "CROSS_ENTROPY":
            if params['loss']['params']['weights'] == 'equal':
                params['weights'] = torch.ones(params['n_classes'])
            elif params['loss']['params']['weights'] == 'calculate':
                if os.path.exists('./loss_weights.npy'):
                    loss_weights = load_loss_weights('./loss_weights.npy')
                    params['weights'] = torch.from_numpy(loss_weights['weights']).float()
                else:
                    import extra.weights_calculator as wc
                    loss_weights, _ = wc.WeightsCalculator(train_labels, params['classes'], dev=False).calculate_and_save()
                    params['weights'] = torch.from_numpy(loss_weights).float()
        elif params['loss']['name'] == "FOCAL_LOSS":
            params['weights'] = torch.ones(params['n_classes'])

        # Imprimindo os pesos das classes para a loss
        print(f"Pesos das classes: {params['weights']}.")
    except Exception as e:
        print(e)
        raise e

def saliency_noise_mean(model, input_batch, device='cuda', noise_intensity=0, backward=False, return_map=False):
    """
    Aplica ruído baseado em saliency a um batch de imagens.
    
    Args:
        input_batch (torch.Tensor): Tensor de imagens com shape [B, C, H, W]
        model (nn.Module): Modelo usado para calcular a saliency
        device (str): Dispositivo onde os cálculos serão feitos
        noise_intensity (float): Intensidade do ruído a ser aplicado
        
    Returns:
        torch.Tensor: Batch de imagens perturbadas com shape [B, C, H, W]
    """
    input_batch = input_batch.clone().detach().to(device).requires_grad_(True)
    #input_batch = input_batch.to(device).requires_grad_()
    
    outputs = model(input_batch)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    saliency_maps = []
    for i in range(input_batch.size(0)):
        output = outputs[i]  # [num_classes, H, W]
            
        # Encontra a classe predita para cada pixel
        target_class_map = output.argmax(dim=0)  # [H, W]
            
        # Pega os logits correspondentes às classes preditas
        H, W = target_class_map.shape
        rows = torch.arange(H, device=device).unsqueeze(1)
        cols = torch.arange(W, device=device)
        logits_per_pixel = output[target_class_map, rows, cols]  # [H, W]
            
        # Calcula o score como média dos logits máximos
        score = logits_per_pixel.mean()
        
        if input_batch.grad is not None:
            input_batch.grad.zero_()
        
        if backward:    
            score.backward(retain_graph=True if i < input_batch.size(0)-1 else False)
                
            grad = input_batch.grad[i].abs()  # [C, H, W]
            saliency = grad.max(dim=0)[0]  # [H, W]
        else:
            grads = torch.autograd.grad(
            outputs=score,
            inputs=input_batch,
            retain_graph=True if i < input_batch.size(0)-1 else False,
            create_graph=False   # True se você quiser derivar dos grads depois
            )[0]  # grads tem shape [B, C, H, W]

            if grads is not None:
                grad = grads[i].abs()  # [C, H, W]
                saliency = grad.max(dim=0)[0]  # [H, W]
            else:
                saliency = torch.zeros_like(target_class_map, dtype=input_batch.dtype)
            
        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        saliency_maps.append(saliency_norm)
        
    saliency_stack = torch.stack(saliency_maps)  # [B, H, W]
    
    noise = torch.randn_like(saliency_stack) * noise_intensity
    perturbation = saliency_stack * noise  # [B, H, W]
    
    perturbation_rgb = perturbation.unsqueeze(1).repeat(1, input_batch.size(1), 1, 1)
    perturbed_batch = input_batch.detach() + perturbation_rgb.detach()

    perturbed_batch = torch.clamp(perturbed_batch, 0, 1)
    
    input_batch = input_batch.detach().requires_grad_(False)

    if return_map:
        return perturbed_batch, saliency_stack
    return perturbed_batch

def entropy_noise(model, input_batch, device='cuda', noise_intensity=1.0, entropy_threshold=0.5, return_map=False):
    """
    Aplica ruído nos pixels mais incertos com base na entropia do modelo, sem usar gradientes.

    Args:
        model (nn.Module): Modelo de segmentação.
        input_batch (torch.Tensor): Batch de imagens [B, C, H, W].
        device (str): Device para cálculo.
        noise_intensity (float): Intensidade do ruído.
        entropy_threshold (float): Limiar de entropia normalizada (0-1) para selecionar pixels incertos.
        reduction (str): 'mean' ou 'max' (mantido para compatibilidade).
        return_map (bool): Se True, retorna também mapas de entropia e máscaras.

    Returns:
        perturbed_batch (torch.Tensor): Imagens perturbadas [B, C, H, W].
        saliency_stack (torch.Tensor): Mapas de entropia normalizada [B, H, W].
        uncertainty_masks (torch.Tensor): Máscaras binárias de pixels incertos [B, H, W].
    """
    
    B, C, H, W = input_batch.shape
    input_batch = input_batch.clone().detach().to(device)

    with torch.no_grad():
        outputs = model(input_batch)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probs = torch.softmax(outputs, dim=1)  # [B, C, H, W]
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=1)  # [B, H, W]
        normalized_entropy = entropy / torch.log(torch.tensor(outputs.shape[1], device=device))
        uncertainty_masks = (normalized_entropy > entropy_threshold).float()  # [B, H, W]

        # Aplicação de ruído apenas nos pixels incertos
        noise = torch.randn_like(normalized_entropy) * noise_intensity
        perturbation = noise * uncertainty_masks  # ruído apenas onde há incerteza

        # Expandir para os canais (RGB, etc.)
        perturbation_rgb = perturbation.unsqueeze(1).expand(-1, C, -1, -1)
        perturbed_batch = (input_batch + perturbation_rgb).clamp(0, 1)

        masked_entropy = normalized_entropy * uncertainty_masks

    if return_map:
        return perturbed_batch, masked_entropy#, normalized_entropy#, uncertainty_masks.bool()
    return perturbed_batch


def saliency_noise_entropy(model, input_batch, device='cuda', noise_intensity=1, entropy_threshold=0.5,
                           reduction='mean', backward=False, return_map=False):
    """
    Aplica ruído baseado em saliency, usando entropia cruzada para identificar pixels incertos.
    
    Args:
        model (nn.Module): Modelo de segmentação.
        input_batch (torch.Tensor): Batch de imagens [B, C, H, W].
        device (str): Device para cálculo.
        noise_intensity (float): Intensidade do ruído.
        entropy_threshold (float): Limiar de entropia normalizada (0-1) para selecionar pixels incertos.
        reduction (str): 'mean' ou 'max' para reduzir gradientes espaciais.
        backward (bool): Se True, usa .backward(); se False, usa autograd.grad para evitar alterar .grad.
        
    Returns:
        perturbed_batch (torch.Tensor): Imagens perturbadas [B, C, H, W].
        saliency_stack (torch.Tensor): Mapas de saliency [B, H, W].
        uncertainty_masks (torch.Tensor): Máscaras binárias de pixels incertos [B, H, W].
    """
    
    B, C, H, W = input_batch.shape
    input_batch = input_batch.clone().detach().to(device).requires_grad_(True)

    outputs = model(input_batch)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    # Entropia cruzada sem rótulo verdadeiro
    probs = torch.softmax(outputs, dim=1)  # [B, C, H, W]
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=1)  # [B, H, W]
    normalized_entropy = entropy / torch.log(torch.tensor(outputs.shape[1], device=device))
    uncertainty_masks = (normalized_entropy > entropy_threshold)  # [B, H, W]
    
    saliency_maps = torch.zeros(B, H, W, device=device)

    for b in range(B):
        if backward and input_batch.grad is not None:
            input_batch.grad.zero_()

        uncertain_pixels = torch.nonzero(uncertainty_masks[b], as_tuple=True)
        if len(uncertain_pixels[0]) == 0:
            continue

        selected_outputs = outputs[b, :, uncertain_pixels[0], uncertain_pixels[1]]
        score = selected_outputs.max(dim=1)[0].mean()

        if backward:
            score.backward(retain_graph=(b < B - 1))
            grad = input_batch.grad[b]
        else:
            grads = torch.autograd.grad(
                outputs=score,
                inputs=input_batch,
                retain_graph=True,
                create_graph=False
            )[0]
            if grads is None:
                grad = torch.zeros_like(input_batch[b])
            else:
                grad = grads[b]

        grad = grad.abs()
        if reduction == 'mean':
            saliency_maps[b] = grad.mean(dim=0)
        elif reduction == 'max':
            saliency_maps[b] = grad.max(dim=0)[0]
        else:
            raise ValueError("Reduction must be 'mean' or 'max'.")

    # Aplica ruído proporcional à saliência
    with torch.no_grad():
        saliency_stack = (saliency_maps - saliency_maps.min()) / (saliency_maps.max() - saliency_maps.min() + 1e-8)
        noise = torch.randn_like(saliency_stack) * noise_intensity
        perturbation = saliency_stack * noise
        perturbation_rgb = perturbation.unsqueeze(1).expand(-1, C, -1, -1)
        perturbed_batch = (input_batch.detach() + perturbation_rgb).clamp(0, 1)
    if return_map:
        return perturbed_batch, saliency_stack, uncertainty_masks
    return perturbed_batch

def autoencoder_noise(images):
    model = ConvolutionalAutoencoder()
    model.load_state_dict(torch.load('./transformation/autoencoder.pth'))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    images = images.to(device)
    
    outputs = model(images)

    return outputs

def add_noise(images, noise_level=0.5):
    """
    Adiciona ruído gaussiano às imagens.

    Args:
        images (torch.Tensor): Tensor de imagens no formato (batch_size, channels, height, width). Espera-se que as imagens tenham escala 0 a 1.
        noise_level (float): Nível de ruído (desvio padrão do ruído gaussiano).

    Returns:
        torch.Tensor: Imagens com ruído adicionado.
    """
    noise = torch.randn_like(images) * noise_level
    noisy_images = images + noise
    
    noisy_images = torch.clamp(noisy_images, min=0.0, max=1.0)
    
    return noisy_images

def fourier_noise(batch, noise_strength=1):
    """
    Adiciona ruído no domínio da frequência usando a Transformada de Fourier.

    Args:
        batch (torch.Tensor): Tensor de imagens no formato (batch_size, channels, height, width). Espera-se que as imagens tenham escala 0 a 1.
        noise_strength (float): Força do ruído a ser adicionado.
    Returns:
        torch.Tensor: Imagens com ruído adicionado.
    """
    batch_noised = []

    for img_tensor in batch:
        channels = []
        for c in img_tensor:
            fft = torch.fft.fft2(c)
            fft_shift = torch.fft.fftshift(fft)

            magnitude = torch.abs(fft_shift)
            phase = torch.angle(fft_shift)

            random_phase = phase + torch.rand_like(phase) * noise_strength - (noise_strength / 2)
            fft_noised = magnitude * torch.exp(1j * random_phase)

            fft_ishift = torch.fft.ifftshift(fft_noised)
            recon = torch.fft.ifft2(fft_ishift).real
            recon = recon.clamp(0, 1)
            channels.append(recon)

        img_noised = torch.stack(channels)
        batch_noised.append(img_noised)

    return torch.stack(batch_noised)  # [B, C, H, W]

def saliency_noise1(model, input_batch, target_class, device = 'cuda', noise_std=10):
    """
    Aplica uma perturbação guiada por saliency em um batch de imagens.

    Args:
        model: modelo PyTorch com suporte a gradiente
        input_batch: tensor [B, C, H, W] com requires_grad=True
        target_class: classe alvo (int)
        noise_std: desvio padrão do ruído multiplicativo
    
    Returns:
        perturbed_batch: tensor [B, C, H, W] com as imagens perturbadas
        saliency_maps: lista de mapas de saliency normalizados [H, W]
    """

    model.to(device)
    input_batch = input_batch.to(device)
    
    model.eval()
    input_batch.requires_grad_(True)
    
    outputs = model(input_batch)  # [B, num_classes, H, W]
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    B, _, H, W = outputs.shape

    saliency_maps = []
    perturbed_batch = []

    for i in range(B):
        output = outputs[i:i+1]  # [1, C, H, W]
        class_score_map = output[:, target_class, :, :]  # [1, H, W]
        score = class_score_map.mean()
        
        model.zero_grad()
        if input_batch.grad is not None:
            input_batch.grad.zero_()
        score.backward(retain_graph=True)
        
        saliency = input_batch.grad[i].abs().max(dim=0)[0]  # [H, W]
        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        saliency_maps.append(saliency_norm)
        
        noise = torch.randn_like(saliency_norm) * noise_std
        noise = 1
        perturbation = saliency_norm * noise  # [H, W]
        perturbation_rgb = perturbation.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]

        img = input_batch[i].detach()
        img_perturbed = img + perturbation_rgb
        img_perturbed = torch.clamp(img_perturbed, 0, 1)

        perturbed_batch.append(img_perturbed)

    perturbed_batch = torch.stack(perturbed_batch)  # [B, C, H, W]
    model.train()
    return perturbed_batch, saliency_maps

def saliency_noise2(model, input_batch, device='cuda', noise_intensity=0.3):
    """
    Aplica ruído baseado em saliency a um batch de imagens.
    
    Args:
        input_batch (torch.Tensor): Tensor de imagens com shape [B, C, H, W]
        model (nn.Module): Modelo usado para calcular a saliency
        device (str): Dispositivo onde os cálculos serão feitos
        noise_intensity (float): Intensidade do ruído a ser aplicado
        
    Returns:
        torch.Tensor: Batch de imagens perturbadas com shape [B, C, H, W]
    """
    model.eval() #confirmar

    input_batch = input_batch.requires_grad_()
    
    outputs = model(input_batch)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    saliency_maps = []
    for i in range(input_batch.size(0)):
        # Seleciona a saída para a i-ésima imagem
        output = outputs[i]  # [num_classes, H, W]
            
        # Encontra a classe predita para cada pixel
        target_class_map = output.argmax(dim=0)  # [H, W]
            
        # Pega os logits correspondentes às classes preditas
        H, W = target_class_map.shape
        rows = torch.arange(H, device=device).unsqueeze(1)
        cols = torch.arange(W, device=device)
        logits_per_pixel = output[target_class_map, rows, cols]  # [H, W]
            
        # Calcula o score como média dos logits máximos
        score = logits_per_pixel.mean()
            
        model.zero_grad() # confirmar
        if input_batch.grad is not None:
            input_batch.grad.zero_()
            
        score.backward(retain_graph=True if i < input_batch.size(0)-1 else False)
            
        grad = input_batch.grad[i].abs()  # [C, H, W]
        saliency = grad.max(dim=0)[0]  # [H, W]
            
        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        saliency_maps.append(saliency_norm)
        
    saliency_stack = torch.stack(saliency_maps)  # [B, H, W]
    
    noise = torch.randn_like(saliency_stack) * noise_intensity # olhar melhor
    perturbation = saliency_stack * noise  # [B, H, W]

    perturbation_rgb = perturbation.unsqueeze(1).repeat(1, input_batch.size(1), 1, 1)
    
    perturbed_batch = input_batch.detach() + perturbation_rgb
    perturbed_batch = torch.clamp(perturbed_batch, 0, 1)

    model.train() # confirmar
    
    return perturbed_batch, saliency_stack


def visualize_predictions(data_test, segmentation_test, prediction, prediction_consistency, num_images=5):
    plt.figure(figsize=(10, 15))
    
    for i in range(num_images):
        
        index = np.random.randint(0, len(data_test))
        
        # Imagem original
        plt.subplot(num_images, 4, i * 4 + 1)
        plt.imshow(data_test[index])
        plt.title("Original Image")
        plt.axis("off")
        
        # Segmentação verdadeira
        plt.subplot(num_images, 4, i * 4 + 2)
        plt.imshow(segmentation_test[index], cmap='gray')
        plt.title("True Segmentation")
        plt.axis("off")
        
        # Segmentação prevista modelo normal
        plt.subplot(num_images, 4, i * 4 + 3)
        plt.imshow(prediction[index], cmap='gray')
        plt.title("Predicted Segmentation")
        plt.axis("off")

        # Segmentação prevista modelo consistency
        plt.subplot(num_images, 4, i * 4 + 4)
        plt.imshow(prediction_consistency[index], cmap='gray')
        plt.title("Consistency")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def load_datasets(params, subject):
    """
    Load datasets from .txt files and return images and labels.
    Args:
        params (dict): Parameters containing the root directory and other configurations.
        subject (str): Subject type ('train', 'val', 'test', 'train_unlabeled').
    Returns:
        images (list): List of image file paths.
        labels (list): List of label file paths.
    """

    images_dir = os.path.join(params['root_dir'], 'images')
    labels_dir = os.path.join(params['root_dir'], 'labels')

    if subject == 'train':

        if params['cross_validation']:
            images = []
            labels = []
            
            base_path = os.path.join(params['root_dir_txt'], 'splits', params['split'], params['kfold_type'])
            
            for fold in params['folds']:
                if fold != params['fold']:  
                    
                    img_path = os.path.join(base_path, str(fold), "images.txt")
                    imgs = pd.read_table(img_path, header=None)[0].tolist()  
                    images.extend([os.path.join(images_dir, img.strip()) for img in imgs if img.strip()])
                    
                    lbl_path = os.path.join(base_path, str(fold), "labels.txt")
                    lbls = pd.read_table(lbl_path, header=None)[0].tolist()
                    labels.extend([os.path.join(labels_dir, lbl.strip()) for lbl in lbls if lbl.strip()])

        else:
            images = pd.read_table(os.path.join(params['root_dir_txt'], 'splits', params['split'], "train_images.txt"), header=None).values
            images = [os.path.join(images_dir, f[0]) for f in images]
            labels = pd.read_table(os.path.join(params['root_dir_txt'], 'splits', params['split'], "train_labels.txt"), header=None).values
            labels = [os.path.join(labels_dir, f[0]) for f in labels]

    elif subject == 'val':
        if params['cross_validation']:
            images = pd.read_table(os.path.join(params['root_dir_txt'], 'splits', params['split'], params['kfold_type'], params['fold'], "images.txt"), header=None).values
            images = [os.path.join(images_dir, f[0]) for f in images]
            labels = pd.read_table(os.path.join(params['root_dir_txt'], 'splits', params['split'], params['kfold_type'], params['fold'], "labels.txt"), header=None).values
            labels = [os.path.join(labels_dir, f[0]) for f in labels]
        else:
            split_dir = os.path.join(params['root_dir_txt'], 'splits', params['split'])
            image_list = os.path.join(split_dir, 'val_images.txt')
            label_list = os.path.join(split_dir, 'val_labels.txt')
            for manifest in (image_list, label_list):
                if not os.path.isfile(manifest):
                    raise FileNotFoundError(
                        f"Independent validation requires {manifest} when cross_validation is false."
                    )

            with open(image_list, encoding='utf-8') as entries:
                images = [os.path.join(images_dir, line.strip()) for line in entries if line.strip()]
            with open(label_list, encoding='utf-8') as entries:
                labels = [os.path.join(labels_dir, line.strip()) for line in entries if line.strip()]
            if not images or len(images) != len(labels):
                raise ValueError("Validation lists must contain the same nonzero number of entries.")

            validation_paths = {os.path.realpath(path) for path in images + labels}
            other_subjects = ['train', 'test']
            if params.get('semi_supervised') and (params.get('consistency', True) or params.get('contrastive', True)):
                other_subjects.append('train_unlabeled')
            for other_subject in other_subjects:
                other_images, other_labels = load_datasets(params, other_subject)
                other_paths = {os.path.realpath(path) for path in other_images + (other_labels or [])}
                overlap = validation_paths & other_paths
                if overlap:
                    raise ValueError(
                        f"Validation overlaps with {other_subject}: {min(overlap)}"
                    )

    elif subject == 'train_unlabeled':
        images = pd.read_table(os.path.join(params['root_dir_txt'], "splits", params['split'], "train_unlabeled_images.txt"), header=None).values
        images = [os.path.join(images_dir, f[0]) for f in images]
        # labels = pd.read_table(os.path.join(params['root_dir_txt'], "splits", "original/", params['dataset'], params['split'], "train_unlabeled_labels.txt"), header=None).values
        # labels = [os.path.join(labels_dir, f[0]) for f in labels]
        return images, None  # No labels for unlabeled training data

    elif subject == 'test':
        images = pd.read_table(os.path.join(params['root_dir_txt'], "test_images.txt"), header=None).values
        images = [os.path.join(images_dir, f[0]) for f in images]
        labels = pd.read_table(os.path.join(params['root_dir_txt'], "test_labels.txt"), header=None).values
        labels = [os.path.join(labels_dir, f[0]) for f in labels]
    
    else:
        raise ValueError("Invalid subject. Choose 'train', 'val', 'test' or 'train_unlabeled'.")

    return images, labels

def count_sliding_window2(top, step=10, window_size=(20, 20)):
    """Count the number of windows in an image"""
    count = 0
    max_x = top.shape[0] - window_size[0]
    max_y = top.shape[1] - window_size[1]

    for x in range(0, max_x + 1, step):
        for y in range(0, max_y + 1, step):
            count += 1
    return count

def sliding_window2(top, step=10, window_size=(20, 20)):
    """Slide a window across the image with a stride of `step`"""
    max_x = top.shape[0] - window_size[0]
    max_y = top.shape[1] - window_size[1]

    for x in range(0, max_x + 1, step):
        for y in range(0, max_y + 1, step):
            yield x, y, window_size[0], window_size[1]


# Utils
import random, os, copy
import numpy as np
import torch
import itertools
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from mlxtend.plotting import plot_confusion_matrix
from os import system, name

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

#### TRAINER UTILS ####
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import torch.optim as optim
# import segmentation_models_pytorch as smp

# palette = {
#     0 : (255, 0, 0),        # Desenvolvimento (vermelho)
#     1 : (38, 115, 0),       # Floresta Mata (verde escuro)
#     2 : (0, 255, 197),      # Piscina (ciano)
#     3 : (0, 0, 0),          # Sombra (preto)
#     4 : (133, 199, 126),    # Floresta Regeneração (verde claro)
#     5 : (255, 255, 0),      # Agricultura (amarelo)
#     6 : (255, 85, 0),       # Formação Rochosa (laranja)
#     7 : (115, 76, 0),       # Solo Exposto (marrom)
#     8 : (84, 117, 168),     # Água (azul escuro)
# }


palette = {
    0 : (255, 0, 0),        # Desenvolvimento (vermelho)
    1 : (38, 115, 0),       # Floresta Mata (verde escuro)
    # 2 : (0, 255, 197),      # Piscina (ciano)
    2 : (0, 0, 0),          # Sombra (preto)
    3 : (133, 199, 126),    # Floresta Regeneração (verde claro)
    4 : (255, 255, 0),      # Agricultura (amarelo)
    5 : (255, 85, 0),       # Formação Rochosa (laranja)
    6 : (115, 76, 0),       # Solo Exposto (marrom)
    7 : (84, 117, 168),     # Água (azul escuro)
}

ICMBIO_INVERT_PALETTE = {color: class_id for class_id, color in palette.items()}


palette2 = {
    0: (255,255,255), # "impervious_surface",
    1: (0,0,255),    # "building",
    2: (0,255,255),   # "low_vegetation",
    3: (0,255,0),     # "tree",
    4: (255,255,0),   # "car",
    5: (255,0,0),     # "clutter"
}

invert_palette = {v: k for k, v in palette2.items()}

def convert_to_color(arr_2d, palette=palette2):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def load_icmbio_label(path):
    """Read the class IDs from a mask encoded with the ICMBio palette."""
    with Image.open(path) as image:
        rgb_mask = image.convert("RGB")
        colors = rgb_mask.getcolors(maxcolors=len(ICMBIO_INVERT_PALETTE))
        if colors is None or any(color not in ICMBIO_INVERT_PALETTE for _, color in colors):
            raise ValueError(f"Label {path} contains colors outside the ICMBio palette.")
        return convert_from_color(np.asarray(rgb_mask), palette=ICMBIO_INVERT_PALETTE)


def resize_label_mask(mask, size):
    """Resize class IDs with nearest-neighbor interpolation to (width, height)."""
    image = Image.fromarray(np.asarray(mask, dtype=np.int32))
    return np.asarray(image.resize(tuple(size), Image.Resampling.NEAREST))


def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    # W, H = img.shape[:-1] # CHANGED
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

def CrossEntropy2d(input, target, weight=None, reduction='mean'):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, reduction=reduction)
    elif dim == 4:
        output = input.view(input.size(0),input.size(1), -1)
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target,weight, reduction=reduction)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def sliding_window(top, step=10, window_size=(20,20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
            
def count_sliding_window(top, step=10, window_size=(20,20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
        
"""  Make optimizer routine"""
def make_optimizer(args, net):
    trainable = filter(lambda x: x.requires_grad, net.parameters()) # Only the parameters that requires gradient are passed to the optimizer

    if args['optimizer'] == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args['momentum']}

    elif args['optimizer'] == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args['beta1'], args['beta2']),
            'eps': args['epsilon']}
        
    elif args['optimizer'] == 'ADAMW':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args['beta1'], args['beta2']),
            'eps': args['epsilon']}
        
    elif args['optimizer'] == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args['epsilon']}

    kwargs['lr'] = args['lr']
    kwargs['weight_decay'] = args['weight_decay']
    
    return optimizer_function(trainable, **kwargs)


""" Make scheduler routine """
def make_scheduler(args, optimizer):
    if args['type'] == 'multi':
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=args['milestones'],
            gamma=args['gamma']
        )
    else:
        scheduler = lrs.StepLR(
            optimizer,
            step_size=args['lr_decay'],
            gamma=args['gamma']
        )
    return scheduler


def calculate_cm(predictions, labels, label_values = None, normalize = None):
    return confusion_matrix(labels, predictions, labels=label_values, normalize=normalize)


""" Global acurracy metric calculation """
def global_accuracy(predictions, labels):
    # Calculate confusion matrix
    cm = calculate_cm(predictions, labels)
    # Sum all values in main diagonal
    main_diagonal = sum([cm[i][i] for i in range(len(cm))])
    # return TP+TN / TP+TN+FN x 100%
    return 100 * float(main_diagonal) / np.sum(cm)

def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    # dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image.cpu().numpy().transpose(1,2,0))
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

def metrics(predictions, gts, label_values, params, all=False, filepath=None):
    cm = calculate_cm(predictions, gts, label_values=range(len(label_values)))
    output_dir = params['results_folder'] if all else filepath
    return metrics_from_confusion_matrix(cm, label_values, output_dir)


def metrics_from_confusion_matrix(cm, label_values, output_dir):
    txt = []
    fig, _ = plot_confusion_matrix(conf_mat=cm,
                                   colorbar=True,
                                   show_absolute=False,
                                   show_normed=True,
                                   class_names=label_values)
    fig.savefig(os.path.join(output_dir, 'cm'), dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)
    
    print("Confusion matrix :")
    print(cm)
    txt.append("Confusion Matrix")
    txt.append(cm)
    
    print("---")
    txt.append("----------------------------")
    
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    txt.append("{} pixels processed".format(total))
    txt.append("Total accuracy : {}%".format(accuracy))
    
    print("---")
    txt.append("----------------------------")
    
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in cnlass i for test set
            pass
    print("F1Score :")
    txt.append('F1Score :')
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))
        txt.append("{}: {}".format(label_values[l_id], score))

    print("---")
    txt.append("----------------------------")
        
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))
    txt.append("Kappa: " + str(kappa))
    txt.append("----------------------------")

    # compute mIoU 
    cm = torch.tensor(cm)
    TP = torch.diag(cm)
    FP = torch.sum(cm, dim=0) - TP
    FN = torch.sum(cm, dim=1) - TP
    IoU = TP / (TP + FP + FN + 1e-7)
    mIoU = IoU[~torch.isnan(IoU)].mean()
    print(f"mIoU: {mIoU}")
    txt.append(f"mIoU: {mIoU}")

    return accuracy, txt

# define our clear function
def clear():
 
    # for windows
    if name == 'nt':
        _ = system('cls')
 
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')
        
def plot_confusion_matrix_local(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    # plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if save:
        fig.savefig(f"./tmp/cm", dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean, std

def save_test(acc, all_preds, all_gts, path=None):
    try:
        print('************** save test results **************')
        if path is None:
            path = './segnet256_test_result.npz'
            
        np.savez_compressed(path, {
            'acc': acc,
            'all_preds': all_preds,
            'all_gts': all_gts,
        })
    except:
        print('[AVISO] Erro ao salvar os resultados do teste!')
        pass
    
def load_test(path=None):
    assert os.path.exists(path), "{} cant be opened".format(path)
    
    data = np.load(path, allow_pickle=True)
    return data.item()

def save_loss_weights(data, path=None):
    if path is None:
        path = './loss_weights.npy'
    np.save(path, data)
        
def load_loss_weights(path=None):
    if path is None:
        path = './loss_weights.npy'
    x = np.load(path, allow_pickle=True)
    return x.item()

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# import torchvision.models as models
# import torch.nn as nn
# def build_model_efficientnet_b0(num_classes, pretrained=True, fine_tune=True):
#     if pretrained:
#         print('[INFO]: Loading pre-trained weights')
#     else:
#         print('[INFO]: Not loading pre-trained weights')
#     model = models.efficientnet_b0(pretrained=pretrained)
#     if fine_tune:
#         print('[INFO]: Fine-tuning all layers...')
#         for params in model.parameters():
#             params.requires_grad = True
#     elif not fine_tune:
#         print('[INFO]: Freezing hidden layers...')
#         for params in model.parameters():
#             params.requires_grad = False
#     # Change the final classification head.
#     model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
#     return model
