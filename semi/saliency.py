import torch

def generate_saliency_maps_for_batch(model, input_images, device='cpu'):
    """
    Gera mapas de saliência para todas as imagens do batch sem afetar os gradientes do modelo.

    Parâmetros:
    - model: O modelo PyTorch que gera o embedding.
    - input_images: Tensor de entrada (B, C, H, W).
    - device: 'cpu' ou 'cuda'.

    Retorna:
    - saliency_maps: Tensor de saliência (B, H_emb, W_emb, H_in, W_in).
    """
    model.eval()

    input_clone = input_images.clone().to(device)
    input_clone.requires_grad_(True)

    _, embedding = model(input_clone)
    embedding = embedding[-1]  # pega o último embedding
    B, C_emb, H_emb, W_emb = embedding.shape
    _, C_in, H_in, W_in = input_clone.shape

    saliency_maps = torch.zeros(B, H_emb, W_emb, H_in, W_in, device=device)

    for h_emb in range(H_emb):
        for w_emb in range(W_emb):

            target_vectors = embedding[:, :, h_emb, w_emb]  # (B, C_emb)

            target_scalars = target_vectors.sum(dim=1) # (B,)

            grads = torch.autograd.grad(
                outputs=target_scalars,    
                inputs=input_clone,        # (B, C, H_in, W_in)
                grad_outputs=torch.ones_like(target_scalars),  
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]  # (B, C, H_in, W_in)
        
            saliency_map, _ = torch.max(grads.abs(), dim=1)  # (B, H_in, W_in)

            saliency_maps[:, h_emb, w_emb] = saliency_map.detach()

    return saliency_maps # (B, H_emb, W_emb, H_in, W_in)

def generate_saliency_map_from_embedding(model, inputs, device):
    inputs = inputs.to(device).requires_grad_(True)
    embeddings = model(inputs)  # shape: [B, C, H, W]
    embeddings = embeddings[1][-1]
    print(f'Embeddings shape is: {embeddings.shape}')

    # critério: norma L2 dos canais em cada posição
    scores = torch.norm(embeddings, dim=1)  # shape: [B, H, W]

    # Agregar para um escalar por batch (senão autograd reclama)
    batch_scores = scores.mean(dim=(1, 2))  # shape: [B]

    saliency_maps = []
    for i in range(inputs.size(0)):
        grads = torch.autograd.grad(
            outputs=batch_scores[i],
            inputs=inputs,
            retain_graph=True
        )[0][i]  # grad da amostra i
        saliency_maps.append(grads.abs().max(dim=0)[0])  # exemplo: max abs nos canais

    saliency_maps = torch.stack(saliency_maps)  # [B, H, W]

    print(f'saliency_maps shape is {saliency_maps.shape}.')
    return saliency_maps

def generate_saliency_maps_for_batch_teste(model, input_images, device='cpu', block_size=32):
    """
    Gera mapas de saliência independentes para cada posição (h_emb, w_emb),
    processando em blocos para maior eficiência.

    Retorna:
    - saliency_maps: (B, H_emb, W_emb, H_in, W_in)
    """
    model.eval()

    input_clone = input_images.clone().to(device)
    input_clone.requires_grad_(True)

    _, embedding = model(input_clone)
    embedding = embedding[-1]  # (B, C_emb, H_emb, W_emb)

    B, C_emb, H_emb, W_emb = embedding.shape
    _, C_in, H_in, W_in = input_clone.shape
    N = H_emb * W_emb

    # (B, C_emb, N)
    embedding_flat = embedding.view(B, C_emb, -1)
    # soma sobre canais -> (B, N)
    target_scalars = embedding_flat.sum(dim=1)

    saliency_blocks = []

    for start in range(0, N, block_size):
        end = min(start + block_size, N)

        # grad_outputs: (B, N), mas só ativa o bloco [start:end]
        grad_outputs = torch.zeros((B, N), device=device)
        grad_outputs[:, start:end] = 1.0

        grads = torch.autograd.grad(
            outputs=target_scalars,   # (B, N)
            inputs=input_clone,
            grad_outputs=grad_outputs,  # (B, N)
            retain_graph=True,
            create_graph=False
        )[0]  # (B, C_in, H_in, W_in)

        # max abs sobre canais -> (B, H_in, W_in)
        sal_block, _ = grads.abs().max(dim=1)

        # guarda o bloco expandido (B, end-start, H_in, W_in)
        saliency_blocks.append(
            sal_block.unsqueeze(1).expand(-1, end-start, -1, -1)
        )

    # junta tudo -> (B, N, H_in, W_in)
    saliency_maps = torch.cat(saliency_blocks, dim=1)

    # reshape -> (B, H_emb, W_emb, H_in, W_in)
    saliency_maps = saliency_maps.view(B, H_emb, W_emb, H_in, W_in)

    return saliency_maps

def generate_saliency_maps_of_sampled_index(model, input_images, sorted_indices, device='cpu'):
    """
    Gera mapas de saliência apenas para os índices sorteados.

    Parâmetros:
    - model: O modelo PyTorch que gera o embedding.
    - input_images: Tensor de entrada (B, C, H_in, W_in).
    - sorted_indices: tensor de shape (N, 3) que contém N coordenadas (b,i,j).
    - device: 'cpu' ou 'cuda'.

    Retorna:
    - saliency_maps: dict { (b,h_emb,w_emb) : mapa_de_salienca (H_in, W_in) }
    """
    model.eval()

    input_clone = input_images.clone().to(device)
    input_clone.requires_grad_(True)

    _, embedding = model(input_clone)
    embedding = embedding[-1]   # (B, C_emb, H_emb, W_emb)
    B, C_emb, H_emb, W_emb = embedding.shape
    _, C_in, H_in, W_in = input_clone.shape

    saliency_maps = []

    for (b, h_emb, w_emb) in sorted_indices:
        # pega o vetor do embedding nesse ponto específico
        target_vector = embedding[b, :, h_emb, w_emb]   # (C_emb,)
        target_scalar = target_vector.sum()             # escalar

        grads = torch.autograd.grad(
            outputs=target_scalar,
            inputs=input_clone,
            grad_outputs=torch.ones_like(target_scalar),
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]   # (B, C_in, H_in, W_in)

        if grads is None:
            continue

        # pega só o gradiente da imagem correspondente ao batch b
        grad_b = grads[b]   # (C_in, H_in, W_in)

        saliency_map, _ = torch.max(grad_b.abs(), dim=0)  # (H_in, W_in)

        saliency_maps.append(saliency_map.detach())

    return saliency_maps
