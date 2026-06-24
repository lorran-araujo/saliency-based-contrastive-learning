import torch
import torch.nn as nn

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: Tensor de predições com shape (batch, C, H, W) - antes do softmax
        targets: Tensor de rótulos com shape (batch, H, W) - valores inteiros de 0 a C-1
        """
        preds = torch.softmax(preds, dim=1)  # Converte logits em probabilidades
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.float()

        dice_per_class = []
        for c in range(self.num_classes):
            pred_flat = preds[:, c, :, :].contiguous().view(preds.shape[0], -1)
            target_flat = targets_one_hot[:, c, :, :].contiguous().view(targets.shape[0], -1)

            intersection = (pred_flat * target_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice)

        dice_mean = torch.mean(torch.stack(dice_per_class, dim=0), dim=0)  # Média das classes
        return 1 - dice_mean.mean()  # Média da Dice Loss no batch
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # Pequeno valor para evitar divisão por zero

    def forward(self, preds, targets):
        """
        preds: Tensor de predições (valores entre 0 e 1) de dimensão (batch, 1, H, W)
        targets: Tensor com rótulos binários (0 ou 1) de dimensão (batch, 1, H, W)
        """
        preds = preds.view(preds.shape[0], -1)  # Flatten para calcular a soma corretamente
        targets = targets.view(targets.shape[0], -1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()  # Média da Dice Loss no batch
