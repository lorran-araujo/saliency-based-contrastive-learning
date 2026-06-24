import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF
import torch
import torchvision.transforms as T

import sys
sys.path.append('./transformation')
from models import ConvolutionalAutoencoder

def saliency_noise_mean_improved(model, input_batch, device='cuda', noise_intensity=0.1, backward=False, return_map=False, alpha=0.5):
    input_batch = input_batch.clone().detach().to(device).requires_grad_(True)
    
    outputs = model(input_batch)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    saliency_maps = []
    for i in range(input_batch.size(0)):
        output = outputs[i]
        target_class_map = output.argmax(dim=0)
        
        H, W = target_class_map.shape
        rows = torch.arange(H, device=device).unsqueeze(1)
        cols = torch.arange(W, device=device)
        logits_per_pixel = output[target_class_map, rows, cols]
        score = logits_per_pixel.mean()
        
        if input_batch.grad is not None:
            input_batch.grad.zero_()
        
        if backward:    
            score.backward(retain_graph=True if i < input_batch.size(0)-1 else False)
            grad = input_batch.grad[i].abs()
            saliency = grad.max(dim=0)[0]
        else:
            grads = torch.autograd.grad(
                outputs=score,
                inputs=input_batch,
                retain_graph=True if i < input_batch.size(0)-1 else False,
                create_graph=False
            )[0]
            
            if grads is not None:
                grad = grads[i].abs()
                saliency = grad.max(dim=0)[0]
            else:
                saliency = torch.zeros_like(target_class_map, dtype=input_batch.dtype)
        
        # Normalização robusta
        q_low = saliency.quantile(0.1)
        q_high = saliency.quantile(0.9)
        saliency_norm = (saliency - q_low) / (q_high - q_low + 1e-10)
        saliency_norm = torch.clamp(saliency_norm, 0, 1)
        saliency_maps.append(saliency_norm)
    
    saliency_stack = torch.stack(saliency_maps)
    
    # Ruído balanceado
    noise = (torch.rand_like(saliency_stack) * 2 - 1) * noise_intensity
    perturbation = saliency_stack * noise
    
    perturbation_rgb = perturbation.unsqueeze(1).repeat(1, input_batch.size(1), 1, 1)
    
    # Mistura adaptativa com a imagem original
    perturbed_batch = input_batch.detach() * (1 - alpha) + (input_batch.detach() + perturbation_rgb.detach()) * alpha
    perturbed_batch = torch.clamp(perturbed_batch, 0, 1)
    
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


def saliency_noise_entropy(model, input_batch, device='cuda', noise_intensity=1, entropy_threshold=0.5, reduction='mean', backward=False, return_map=False):
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

def cutmix(imgs, beta=1.0, lm=None, coords=None, return_all=False):
    B, C, H, W = imgs.size()
    idx = torch.arange(B)
    idx_next = (idx + 1) % B
    imgs_aug = imgs.clone()

    lams = []
    all_coords = []

    for i in range(B):
        if lm is None:
            lam = np.random.beta(beta, 1.0)
        else:
            lam = lm[i]

        if coords is None:
            cut_w = int(W * np.sqrt(1 - lam))
            cut_h = int(H * np.sqrt(1 - lam))
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            x1 = np.clip(cx - cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y2 = np.clip(cy + cut_h // 2, 0, H)
        else:
            x1, y1, x2, y2 = coords[i]

        imgs_aug[i, :, y1:y2, x1:x2] = imgs[idx_next[i], :, y1:y2, x1:x2]
        lams.append(lam)
        all_coords.append((x1, y1, x2, y2))

    if return_all:
        return imgs_aug, lams, all_coords
    return imgs_aug

def get_gaussian_kernel(kernel_size=5, sigma=1.0):
    """
    Gera um kernel gaussiano 2D separável.
    """
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    ax = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = ax[:, None] @ ax[None, :]
    kernel = kernel / kernel.sum()
    return kernel

import math

def gaussian_blur(imgs, sigma=10.0, kernel_size=32):
    """
    Aplica blur gaussiano a um batch de imagens.
    
    Parâmetros:
    - imgs: tensor (B, C, H, W), valores em [0, 1]
    - kernel_size: tamanho do kernel gaussiano (ímpar)
    - sigma: desvio padrão da Gaussiana (quanto maior, mais borrado)
    
    Retorna:
    - imgs_blur: tensor com imagens desfocadas
    """
    B, C, H, W = imgs.shape
    kernel = get_gaussian_kernel(kernel_size, sigma)
    kernel = kernel.to(imgs.device)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)

    # aplicar convolução por canal
    imgs_blur = F.conv2d(imgs, kernel, padding=kernel_size // 2, groups=C)
    return imgs_blur

def box_blur(imgs, kernel_size=5):
    """
    Aplica box blur (filtro da média) a um batch de imagens.

    Parâmetros:
    - imgs: tensor (B, C, H, W)
    - kernel_size: tamanho do kernel (ímpar), ex: 3, 5, 7

    Retorna:
    - imgs_blur: tensor com box blur aplicado
    """
    B, C, H, W = imgs.shape
    kernel = torch.ones((kernel_size, kernel_size), device=imgs.device)
    kernel = kernel / kernel.sum()
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)

    imgs_blur = F.conv2d(imgs, kernel, padding=kernel_size // 2, groups=C)
    return imgs_blur


def cutout_images(images):
    cutout = transforms.RandomErasing(p=1,
                                    ratio=(1, 1),
                                    scale=(0.01, 0.05),
                                    value=127)
    
    return torch.stack([cutout(img) for img in images])

def SA(imgs):
    methods=['cutmix_images', 'cutout_images']
    random_method = random.choice(methods)
    if random_method=='cutmix_images':
        mixed_images = cutmix(imgs, beta=1.0)
        return mixed_images
    else:
        return cutout_images(imgs)
    

def WA(images, p=0.5, rotation_range=(-15, 15), scale_range=(0.9, 1.1), translation_range=(-0.05, 0.05)):
    """
    Aplicação de transformações geométricas com melhor controle de artefatos.
    """
    device = images.device
    B, C, H, W = images.shape
    theta = torch.zeros((B, 2, 3), device=device)

    for i in range(B):
        # Inicialização da matriz como identidade
        angle = scale_x = scale_y = tx = ty = 0.0
        flip = 1.0

        if np.random.rand() < p:
            angle = np.random.uniform(*rotation_range)
        if np.random.rand() < p:
            scale_x = np.random.uniform(*scale_range)
            scale_y = np.random.uniform(*scale_range)
        else:
            scale_x = scale_y = 1.0
        if np.random.rand() < p:
            tx = np.random.uniform(*translation_range)
            ty = np.random.uniform(*translation_range)
        if np.random.rand() < p:
            flip = -1.0

        angle = np.deg2rad(angle)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Matriz 2x3 para affine_grid (com coordenadas normalizadas [-1, 1])
        theta[i, 0, 0] = flip * scale_x * cos_a
        theta[i, 0, 1] = -scale_y * sin_a
        theta[i, 1, 0] = scale_x * sin_a
        theta[i, 1, 1] = scale_y * cos_a
        theta[i, 0, 2] = tx * 2  # porque o grid é normalizado [-1, 1]
        theta[i, 1, 2] = ty * 2

    grid = F.affine_grid(theta, images.size(), align_corners=False)
    transformed = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return transformed

def random_affine_transform(imgs, degrees=(-180, 180), translate=(0.2, 0.2), scale_ranges=(0.5, 1.5), return_params=False, params=None):
    """
    Aplica transformação afim aleatória a um batch de imagens.

    Parâmetros:
    - imgs: Tensor de forma (B, C, H, W)
    - degrees: intervalo de rotação em graus
    - translate: fração da largura e altura para translação (tx, ty)
    - scale_ranges: intervalo de escala (zoom)
    - return_params: se True, retorna os parâmetros usados
    - params: lista de dicionários com transformações específicas a aplicar (útil para labels)

    Retorna:
    - imgs_tf: imagens transformadas
    - all_params (opcional): lista de parâmetros de cada transformação
    """
    B, C, H, W = imgs.shape
    imgs_tf = []

    all_params = []

    for i in range(B):
        if params is None:
            angle = random.uniform(*degrees)
            translate_x = random.uniform(-translate[0], translate[0]) * W
            translate_y = random.uniform(-translate[1], translate[1]) * H
            scale = random.uniform(*scale_ranges)
            shear = [0.0, 0.0]  # sem cisalhamento
            current_params = {
                'angle': angle,
                'translate': (translate_x, translate_y),
                'scale': scale,
                'shear': shear
            }
        else:
            current_params = params[i]

        img = TF.affine(
            imgs[i], 
            angle=current_params['angle'],
            translate=current_params['translate'],
            scale=current_params['scale'],
            shear=current_params['shear']
        )
        imgs_tf.append(img)
        all_params.append(current_params)

    imgs_tf = torch.stack(imgs_tf)

    if return_params:
        return imgs_tf, all_params
    return imgs_tf

def grid_shuffle(imgs, grid_size=(3, 3), return_perm=False, perm=None):
    """
    Aplica grid shuffle a um batch de imagens, embaralhando os blocos de MxN,
    preservando o tamanho original da imagem, com padding se necessário.

    Parâmetros:
    - imgs: Tensor de forma (B, C, H, W)
    - grid_size: tupla (M, N) que define quantas divisões na vertical e horizontal
    - return_perm: se True, retorna a permutação usada
    - perm: permutação específica a ser aplicada (por exemplo, nas labels)

    Retorna:
    - imgs_shuffle: Tensor embaralhado (B, C, H, W)
    - perms (opcional): lista de permutações aplicadas por imagem
    """
    B, C, H, W = imgs.shape
    M, N = grid_size
    
    # Calcula o tamanho dos patches
    patch_h = H // M
    patch_w = W // N

    imgs_shuffle = []
    perms = []

    for b in range(B):
        patches = []

        # Divide a imagem em MxN blocos
        for i in range(M):
            for j in range(N):
                y1 = i * patch_h
                y2 = (i + 1) * patch_h if i < M - 1 else H
                x1 = j * patch_w
                x2 = (j + 1) * patch_w if j < N - 1 else W
                patch = imgs[b, :, y1:y2, x1:x2]
                patches.append(patch)

        # Aplica a permutação
        if perm is None:
            idx = list(range(M * N))
            random.shuffle(idx)
        else:
            idx = perm[b]
        patches_shuffled = [patches[i] for i in idx]
        perms.append(idx)

        # Reconstrói a imagem
        # Primeiro determina o tamanho máximo de altura e largura entre os patches
        max_h = max(p.shape[1] for p in patches_shuffled)
        max_w = max(p.shape[2] for p in patches_shuffled)
        
        # Aplica padding a todos os patches para terem o mesmo tamanho
        padded_patches = []
        for p in patches_shuffled:
            pad_h = max_h - p.shape[1]
            pad_w = max_w - p.shape[2]
            # Pad: (left, right, top, bottom)
            p = F.pad(p, (0, pad_w, 0, pad_h), mode='reflect')
            padded_patches.append(p)
        
        # Agrupa os patches em linhas (M linhas de N patches cada)
        rows = []
        for i in range(M):
            row = torch.cat(padded_patches[i*N : (i+1)*N], dim=2)  # concat horizontal
            rows.append(row)
        
        # Concatena as linhas verticalmente
        img_shuffled = torch.cat(rows, dim=1)
        
        # Corta para o tamanho original (caso tenha sido adicionado padding)
        img_shuffled = img_shuffled[:, :H, :W]
        imgs_shuffle.append(img_shuffled)

    imgs_shuffle = torch.stack(imgs_shuffle)

    if return_perm:
        return imgs_shuffle, perms
    return imgs_shuffle

def random_crop_flip_resize(
    imgs,
    p_flip_h=0.3,
    p_flip_v=0.3,
    p_crop=0.3,
    crop_size=(224, 224),
    p_resize=0.3,
    resize_size=(224, 224)
):
    """
    Aplica flip, crop e resize aleatórios a um batch de imagens (B, C, H, W).

    Parâmetros:
    - p_flip_h: probabilidade de flip horizontal
    - p_flip_v: probabilidade de flip vertical
    - p_crop: probabilidade de crop aleatório
    - crop_size: tupla (h, w) para crop
    - p_resize: probabilidade de resize
    - resize_size: tupla (h, w) para resize

    Retorna:
    - imgs_tf: tensor (B, C, H', W') com imagens transformadas
    """
    B, C, H, W = imgs.shape
    imgs_tf = []

    crop = False
    if random.random() < p_crop and crop_size[0] <= imgs[0].shape[1] and crop_size[1] <= imgs[0].shape[2]:
        top = random.randint(0, imgs[0].shape[1] - crop_size[0])
        left = random.randint(0, imgs[0].shape[2] - crop_size[1])
        crop = True

    res = False
    if random.random() < p_resize:
        res = True
    
    for i in range(B):
        img = imgs[i]

        # Flip horizontal
        if random.random() < p_flip_h:
            img = TF.hflip(img)

        # Flip vertical
        if random.random() < p_flip_v:
            img = TF.vflip(img)

        # Crop aleatório
        if crop:
            img = TF.crop(img, top, left, crop_size[0], crop_size[1])

        # Resize
        if res:
            img = TF.resize(img, resize_size, antialias=True)

        imgs_tf.append(img)

    imgs_tf = torch.stack(imgs_tf)
    return imgs_tf

def strong_augmentations(batch):
    """
    Aplica augmentações fortes (brilho, contraste e matiz) em um batch de imagens.

    Parâmetros:
    - batch (Tensor): tensor de imagens com shape (B, C, H, W) e valores em [0, 1]

    Retorna:
    - Tensor com as imagens transformadas, mesmo shape do input (B, C, H, W)
    """

    B, C, H, W = batch.shape
    assert C == 3, "As imagens devem ter 3 canais (RGB) para alteração de cor"

    # Define as transformações
    color_jitter = T.ColorJitter(
        brightness=0.5,   # +/- 50% de brilho
        contrast=0.5,     # +/- 50% de contraste
        hue=0.2           # +/- 20% de matiz
    )

    # Aplica a transformação individualmente em cada imagem do batch
    augmented = torch.stack([color_jitter(img) for img in batch])

    return augmented
