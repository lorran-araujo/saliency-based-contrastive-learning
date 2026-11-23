import torch
import torch.nn.functional as F

# Sampling methods for negative pairs

# ======================================================================================================================================
#  Methods of article: https://openaccess.thecvf.com/content/ICCV2021/html/Zhong_Pixel_Contrastive-Consistent_Semi-Supervised_Semantic_Segmentation_ICCV_2021_paper.html
# ======================================================================================================================================

def sampling_negative_pairs_uniform(embedding, N, i_idx):
    """
    Amostra N pixels negativos do embedding uniformemente.
    
    Parâmetros:
    - embedding: Tensor de shape (B, H, W, C)
    - N: número de pixels negativos a amostrar
    - i_idx: tupla (b, h, w) do pixel positivo (opcional)
    - exclude_i: se True, exclui o pixel positivo da amostragem

    Retorna:
    - Tensor de shape (N, C) com os embeddings dos pixels negativos
    """
    embedding_r = embedding.permute(0, 3, 1, 2)
    B, C, H, W = embedding_r.shape
    M = B * H * W

    # Cria todos os índices possíveis
    all_indices = torch.arange(M)
    
    # Excluindo índice i_idx
    i_flat = i_idx[0] * (H * W) + i_idx[1] * W + i_idx[2]
    all_indices = all_indices[all_indices != i_flat]

    # Amostra N índices únicos uniformemente
    sampled_indices = torch.randperm(len(all_indices))[:N]
    flat_neg_indices = all_indices[sampled_indices]

    # Converte índices achatados para (b, h, w)
    b_idx = flat_neg_indices // (H * W)
    hw_idx = flat_neg_indices % (H * W)
    h_idx = hw_idx // W
    w_idx = hw_idx % W

    # Coleta os vetores do embedding_r
    neg_pixels = embedding_r[b_idx, :, h_idx, w_idx]  # shape (N, C)

    return neg_pixels

def sample_negatives_from_other_images(embedding, N, i_idx):
    """
    Amostra N pixels negativos uniformemente entre as outras imagens do mini-batch
    (exclui todos os pixels da imagem correspondente a i_idx[0]).

    Parâmetros:
    - embedding: Tensor (B, H, W, C)
    - N: número de pixels negativos
    - i_idx: tupla (b, h, w) do pixel positivo

    Retorna:
    - Tensor (N, C) com os embeddings dos pixels negativos
    """
    embedding_r = embedding.permute(0, 3, 1, 2)
    B, C, H, W = embedding_r.shape
    b_i = i_idx[0]

    # Cria todos os índices possíveis (batch, h, w)
    b_all = torch.arange(B).repeat_interleave(H * W)
    h_all = torch.arange(H).repeat(W).repeat(B)
    w_all = torch.arange(W).repeat(B * H)

    # Remove os índices da imagem correspondente a b_i
    mask = b_all != b_i
    b_valid = b_all[mask]
    h_valid = h_all[mask]
    w_valid = w_all[mask]

    total_valid = b_valid.shape[0]
    if N > total_valid:
        raise ValueError(f"N={N} é maior que o número de pixels disponíveis fora da imagem {b_i} ({total_valid}).")

    # Amostra N pixels aleatórios
    rand_indices = torch.randperm(total_valid)[:N]
    b_idx = b_valid[rand_indices]
    h_idx = h_valid[rand_indices]
    w_idx = w_valid[rand_indices]

    # Coleta os pixels negativos
    negatives = embedding_r[b_idx, :, h_idx, w_idx]  # (N, C)

    return negatives

def sampling_negative_pseudolabels(embedding, pseudo_labels, N, i_idx, exclude_i=True):
    """
    Amostra N pixels negativos do embedding com ponderação baseada nos pseudo-rótulos.
    
    Parâmetros:
    - embedding: Tensor de shape (B, H, W, C)
    - pseudo_labels: Tensor de shape (B, num_classes, H_p, W_p)
    - N: número de pixels negativos a amostrar
    - i_idx: tupla (b, h, w) indicando o pixel âncora positivo
    - exclude_i: se True, exclui o pixel positivo da amostragem

    Retorna:
    - Tensor de shape (N, C) com os embeddings dos pixels negativos
    """

    embedding_r = embedding.permute(0, 3, 1, 2)
    B, C, H, W = embedding_r.shape
    _, num_classes, H_p, W_p = pseudo_labels.shape

    # Redimensiona pseudo_labels para ter o mesmo tamanho espacial dos embeddings
    if (H_p != H) or (W_p != W):
        pseudo_labels = F.interpolate(pseudo_labels, size=(H, W), mode='bilinear', align_corners=False) #como é feito? quais classes ficam no final?

    # Achata os mapas para forma (B*H*W, ...)
    M = B * H * W
    embedding_r_flat = embedding_r.permute(0, 2, 3, 1).reshape(M, C)              # (M, C)
    labels_flat = pseudo_labels.permute(0, 2, 3, 1).reshape(M, num_classes)   # (M, num_classes)

    # Índice achatado do pixel i
    if i_idx is None:
        raise ValueError("i_idx não pode ser None.")

    b, h, w = i_idx
    i_flat = b * H * W + h * W + w

    # Vetor de pseudo-label de i
    yi = labels_flat[i_flat]  # shape: (num_classes,)

    # Produto escalar com todos os outros yj
    similarity = torch.matmul(labels_flat, yi)  # (M,)
    prob_diff = 1.0 - similarity  # Quanto maior, mais diferente

    # Remove o próprio i se necessário
    if exclude_i:
        prob_diff[i_flat] = 0.0

    # Normaliza para formar uma distribuição de probabilidade
    probs = prob_diff / (prob_diff.sum() + 1e-8)

    # Amostra N índices com reposição usando as probabilidades ponderadas
    sampled_indices = torch.multinomial(probs, N, replacement=False)  # shape: (N,)

    # Pega os embedding_rs correspondentes
    neg_pixels = embedding_r_flat[sampled_indices]  # shape: (N, C)

    return neg_pixels

def sampling_negative_pseudolabels_from_others_images(embedding, pseudo_labels, N, i_idx):
    """
    Amostra N pixels negativos do embedding, excluindo todos os pixels da mesma imagem (mesmo índice de batch).
    A ponderação é feita com base nos pseudo-rótulos: P(negativo) ∝ 1 - y_i^T y_j.
    
    Parâmetros:
    - embedding: Tensor de shape (B, H, W, C)
    - pseudo_labels: Tensor de shape (B, num_classes, H_p, W_p)
    - N: número de pixels negativos a amostrar
    - i_idx: tupla (b, h, w) indicando o pixel âncora positivo

    Retorna:
    - Tensor de shape (N, C) com os embeddings dos pixels negativos
    """

    embedding = embedding.permute(0, 3, 1, 2)
    B, C, H, W = embedding.shape
    _, num_classes, H_p, W_p = pseudo_labels.shape

    # Redimensiona pseudo_labels para o mesmo tamanho dos embeddings
    if (H_p != H) or (W_p != W):
        pseudo_labels = F.interpolate(pseudo_labels, size=(H, W), mode='bilinear', align_corners=False) # como esta sendo feito? quando uma classe é previlegiada?

    # Achata os tensores
    M = B * H * W
    embedding_flat = embedding.permute(0, 2, 3, 1).reshape(M, C)              # (M, C)
    labels_flat = pseudo_labels.permute(0, 2, 3, 1).reshape(M, num_classes)   # (M, num_classes)
    
    # Índice achatado de i
    if i_idx is None:
        raise ValueError("i_idx não pode ser None.")
    b, h, w = i_idx
    i_flat = b * H * W + h * W + w

    # Vetor de pseudo-label do pixel i
    yi = labels_flat[i_flat]  # shape: (num_classes,)

    # Produto escalar com todos os y_j
    labels_flat = torch.softmax(labels_flat, dim=-1)
    yi = torch.softmax(yi, dim=-1)
    similarity = torch.matmul(labels_flat, yi)  # (M,)
    prob_diff = 1.0 - similarity  # (M,)

    # Cria máscara para excluir todos os pixels da imagem de índice b
    batch_indices = torch.arange(M) // (H * W)  # shape (M,), valores de 0 a B-1
    mask = batch_indices != b  # True para manter, False para remover

    # Aplica máscara
    valid_probs = prob_diff[mask]
    valid_embeddings = embedding_flat[mask]

    # tratando casos de erro
    sum_probs = valid_probs.sum()
    
    if sum_probs <= 0 or torch.isnan(sum_probs):
        probs = torch.ones_like(valid_probs) / len(valid_probs)
        if sum_probs <= 0:
            print('Usando distribuição uniforme pois sum_probs = 0.')
        else:
            print('Usando distribuição uniforme sum_probs é NaN.')
    else:
        # Normaliza as probabilidades
        probs = valid_probs / (sum_probs + 1e-8)

    # Amostra N índices com base nas probabilidades ponderadas
    sampled_indices = torch.multinomial(probs, N, replacement=False)  # (N,)

    # Coleta os embeddings correspondentes
    neg_pixels = valid_embeddings[sampled_indices]  # (N, C)

    return neg_pixels

# ======================================================================================================================================
# My methods
# ======================================================================================================================================

def negative_sample_by_saliency(
    embedding, saliency_map, pred_logits, device='cuda',
    top_k_pixels=50, min_confidence=0.8, num_negatives=20):
    """
    Para cada vetor (b, i, j) do embedding, calcula as N amostras negativas. 
    
    A função pega um vetor do embedding e, através do mapa de saliência já calculado anteriormente, verifica quais píxeis
    da entrada foram mais importantes para a contrução daquele vetor do embedding. Selecionado os top_k pixeis pixeis, será olhada
    a possivel classe deles atraves de pred_logits. Se a predição desses topk pixeis mais importantes do mapa de saliencia do vetor de embedding
    possuir uma classe dominante, então diremos que esse vetor de embedding é dessa clase. Dessa
    forma, cada vetor do embedding de cada batch terá uma classe, e assim podemos sortear pixeis negativos para eles de todo o batch, 
    de forma que não sejam da mesma classe.

    Args:
        embedding: Tensor (B, C_emb, H_emb, W_emb), de embeddings da rede
        saliency_map: Tensor (B, H_emb, W_emb, H, W), com mapa de saliencia para cada vetor do embedding
        pred_logits: Tensor (B, C_out, H, W), com a predição da rede para aquele batch
        top_k_pixels: scalar, com o número de pixeis do mapa de saliencia a ser considerado
        min_confidence: percentage (0 a 1), com a porcentagem de dominancia para afirmar que um pixel representa uma determinada classe
        num_negatives: scalar, número de pares negativos a ser sorteado
    
    Returns:
        negative_list (B, H_emb, W_emb, num_negatives, 3), lista de num_negativos pares, onde 3 se refere a (b, h, w) (coordenadas negativa sorteada)
    """
    # calculando os topk indices para cada pixel do embedding

    Bs, H_emb, W_emb, H, W = saliency_map.shape
    saliency_flattened = saliency_map.view(Bs, H_emb, W_emb, -1) # shape (B, H_emb, W_emb, H*W)

    topk_values, topk_index = torch.topk(saliency_flattened, k=top_k_pixels, dim=-1)

    h_indices = topk_index // W
    w_indices = topk_index % W

    index_2d = torch.stack([h_indices, w_indices], dim=-1) # shape (B, H_emb, W_emb, k, 2)

    # usar predlogits desses indices para mapear cada vetor do embedding para uma classe
    B, C, _, _ = pred_logits.shape

    batch_idx = torch.arange(B).view(B, 1, 1, 1, 1).expand(B, H_emb, W_emb, top_k_pixels, 1)
    class_idx = torch.arange(C).view(1, 1, 1, 1, C).expand(B, H_emb, W_emb, top_k_pixels, C)
    
    h_indices = index_2d[..., 0]  # shape (B, H_emb, W_emb, k)
    w_indices = index_2d[..., 1]  # shape (B, H_emb, W_emb, k)

    selected_preds = pred_logits[batch_idx[...,0], :, h_indices, w_indices] # shape (B, H_emb, W_emb, k, C)
    predicted_classes = selected_preds.argmax(dim=-1) # shape (B, H_emb, W_emb, k)
    majority_classes, _ = predicted_classes.mode(dim=-1)  # shape (B, H_emb, W_emb)
    
    # usando vetores de classes para poder calcular pixeis negativos

    negative_list = torch.zeros(B, H_emb, W_emb, num_negatives, 3, dtype=torch.long, device=device)

    all_coords = torch.stack(torch.meshgrid(
        torch.arange(B, device=device),
        torch.arange(H_emb, device=device),
        torch.arange(W_emb, device=device),
        indexing='ij'
    ), dim=-1).view(-1, 3)  # shape (B*H_emb*W_emb, 3)
    
    all_classes = majority_classes.view(-1)  # shape (B*H_emb*W_emb)

    for b in range(B):
        for i in range(H_emb):
            for j in range(W_emb):
                current_class = majority_classes[b, i, j]
                current_idx = b * H_emb * W_emb + i * W_emb + j
                
                different_mask = (all_classes != current_class)
                neg_coords = all_coords[different_mask]
                
                self_mask = (neg_coords != torch.tensor([b, i, j], device=device)).all(dim=1)
                neg_coords = neg_coords[self_mask]
                
                # Selecionar negativos
                if len(neg_coords) == 0:
                    # Usar qualquer pixel (exceto o próprio)
                    other_mask = torch.ones(len(all_coords), dtype=torch.bool, device=device)
                    other_mask[current_idx] = False
                    neg_coords = all_coords[other_mask]
                
                if len(neg_coords) < num_negatives:
                    # Repetir ou usar todos disponíveis
                    if len(neg_coords) > 0:
                        repeat_times = (num_negatives + len(neg_coords) - 1) // len(neg_coords)
                        selected_negatives = neg_coords.repeat(repeat_times, 1)[:num_negatives]
                    else:
                        # Caso extremo: só tem 1 pixel no dataset
                        selected_negatives = all_coords[0:1].repeat(num_negatives, 1)
                else:
                    # Amostrar aleatoriamente de classes diferentes (conferir se entra aqui, é o que quero)
                    rand_indices = torch.randperm(len(neg_coords), device=device)[:num_negatives]
                    selected_negatives = neg_coords[rand_indices]
                
                negative_list[b, i, j] = selected_negatives

    return negative_list

def negative_sample_by_saliency_of_sampled_index(sorted_index, saliency_map, N_ancoras, pred_logits, device='cuda', top_k_pixels=50, min_confidence=0.1):
    """
    Para cada indice (b,i,j), consulta o respectivo mapa de saliência referente a esse vetor em saliency_map. A partir disso, seleciona os top_k_pixeis
    mais importantes para esse indice no mapa de saliencia, e confere a label desses pixeis em pred_logits. Se os pixeis forem em porcentagem min_confidence
    de uma mesma classe, então denominamos que esse indice (b,i,j) é dessa classe, construindo então para cada indice (b,i,j) uma variavel contendo suas classes.
    Após isso, sortear N píxeis ancoras de sorted index. Para cada um deles, vamos pegar todos os outros pixeis não ancoras que forem de classes diferentes
    para usar como amostras negativas. Logo, para cada um dos N indice ancora, teremos os respectivos pares negativos.

    Args:
    sorted_index: tensor de shape (N, 3) que contem N indices (b,i,j).
    saliency_map: lista de tensores com os respectivos mapas de saliencia para cada um dos sorted_index.
        N_ancoras: número máximo de âncoras sorteadas entre os vetores com classe definida.
    pred_logits: tensor [B, num_classes, H, W] com as probabilidades/logits preditos.
    device: 'cuda' ou 'cpu'.
    top_k_pixels: quantos pixeis mais salientes considerar.
    min_confidence: fração mínima para definir a classe dominante.
    
    Retorna:
        anchors: tensor [A, 3] com índices das âncoras, com A <= N_ancoras.
        neg_samples: lista de tensores, cada tensor contém índices [M_neg, 3] correspondentes aos pares negativos de cada âncora.
        anchor_classes: tensor [A] com a classe definida de cada âncora.
    """

    B, num_classes, H, W = pred_logits.shape
    probs = torch.softmax(pred_logits, dim=1)

    index_classes = []

    for k, idx in enumerate(sorted_index):
        b, y, x = idx
        sal_map = saliency_map[k]  # [H, W]

        # top-k pixeis mais relevantes
        flat_sal = sal_map.view(-1)
        topk_vals, topk_inds = torch.topk(flat_sal, top_k_pixels)

        topk_y = topk_inds // W
        topk_x = topk_inds % W

        probs_topk = probs[b, :, topk_y, topk_x]  # [num_classes, top_k]
        topk_classes = torch.argmax(probs_topk, dim=0)  # [top_k]

        unique_classes = torch.unique(topk_classes)

        counts = torch.bincount(topk_classes)   # conta quantas vezes cada classe aparece
        dominant_class = torch.argmax(counts)

        dominant_fraction = counts[dominant_class].float() / topk_classes.numel()

        if dominant_fraction >= min_confidence:
            index_classes.append(dominant_class)
        else:
            index_classes.append(-1)

    index_classes = torch.tensor(index_classes, dtype=torch.long, device=device)
    has_class = index_classes != -1
    candidates = has_class.nonzero(as_tuple=True)[0]
    perm = torch.randperm(candidates.numel(), device=device)
    anchor_indices = candidates[perm[:N_ancoras]]
    anchors = sorted_index[anchor_indices]
    anchor_classes = index_classes[anchor_indices]

    neg_samples = []
    for anc_class in anchor_classes:
        mask = has_class & (index_classes != anc_class)
        negs = sorted_index[mask] 
        neg_samples.append(negs)

    return anchors, neg_samples, anchor_classes

def negative_sample_pondered_by_saliency_of_sampled_index(sorted_index, saliency_map, N_ancoras, pred_logits, device='cuda', top_k_pixels=50, min_confidence=0.1):
    """
    Para cada indice (b,i,j), consulta o respectivo mapa de saliência referente a esse vetor em saliency_map. A partir disso, seleciona os top_k_pixeis
    mais importantes para esse indice no mapa de saliencia, e confere a label desses pixeis em pred_logits. Se os pixeis forem em porcentagem min_confidence
    de uma mesma classe, então denominamos que esse indice (b,i,j) é dessa classe, construindo então para cada indice (b,i,j) uma variavel contendo suas classes.
    Após isso, sortear N píxeis ancoras de sorted index. Para cada um deles, vamos pegar todos os outros pixeis não ancoras que forem de classes diferentes
    para usar como amostras negativas. Logo, para cada um dos N indice ancora, teremos os respectivos pares negativos.

    Args:
    sorted_index: tensor de shape (N, 3) que contem N indices (b,i,j).
    saliency_map: lista de tensores com os respectivos mapas de saliencia para cada um dos sorted_index.
    N_ancoras: número de âncoras a serem sorteadas de sorted_index.
    pred_logits: tensor [B, num_classes, H, W] com as probabilidades/logits preditos.
    device: 'cuda' ou 'cpu'.
    top_k_pixels: quantos pixeis mais salientes considerar.
    min_confidence: fração mínima para definir a classe dominante.
    
    Retorna:
        anchors: tensor [N_ancoras, 3] com índices das âncoras.
        neg_samples: lista de tensores, cada tensor contém índices [M_neg, 3] correspondentes aos pares negativos de cada âncora.
        anchor_classes: tensor [N_ancoras] com a classe definida de cada âncora.
    """

    B, num_classes, H, W = pred_logits.shape
    probs = torch.softmax(pred_logits, dim=1)

    # pondering by class imbalanced strategy
    pesos_dict = {0: 0.0758, 1:0.0253, 2:0.1352,3: 0.0293, 4: 0.0981,5: 0.1604, 6:0.3437, 7: 0.1323}
    pesos = torch.tensor([pesos_dict[c] for c in sorted(pesos_dict.keys())], dtype=torch.float32) 
    probs = probs * pesos.view(1, -1, 1, 1)

    index_classes = []

    print('\n')
    for k, idx in enumerate(sorted_index):
        b, y, x = idx
        sal_map = saliency_map[k]  # [H, W]

        # top-k pixeis mais relevantes
        flat_sal = sal_map.view(-1)
        topk_vals, topk_inds = torch.topk(flat_sal, top_k_pixels)

        topk_y = topk_inds // W
        topk_x = topk_inds % W

        # optional: pondering by saliency map
        saliencia = saliencia.view(1, 1, *saliencia.shape[-2:])
        probs_ponderado = probs * saliencia

        probs_topk = probs_ponderado[b, :, topk_y, topk_x]  # [num_classes, top_k]
        topk_classes = torch.argmax(probs_topk, dim=0)  # [top_k]

        unique_classes = torch.unique(topk_classes)
        #print(f'Classes diferentes nos top-k pixels: {unique_classes}')

        counts = torch.bincount(topk_classes)   # conta quantas vezes cada classe aparece
        dominant_class = torch.argmax(counts)
        #print(f'a classe dominante foi: {dominant_class}')

        dominant_fraction = counts[dominant_class].float() / topk_classes.numel()

        if dominant_fraction >= min_confidence:
            index_classes.append(dominant_class)
        else:
            index_classes.append(-1)

    index_classes = torch.tensor(index_classes, device=device)

    # sortear âncoras
    perm = torch.randperm(len(sorted_index), device=device)
    anchors = sorted_index[perm[:N_ancoras]]
    anchor_classes = index_classes[perm[:N_ancoras]]

    neg_samples = []
    for anc_class, anc_idx in zip(anchor_classes, anchors):
        mask = (index_classes != anc_class) & (index_classes != -1)
        negs = sorted_index[mask] 
        neg_samples.append(negs)

    return anchors, neg_samples, anchor_classes
