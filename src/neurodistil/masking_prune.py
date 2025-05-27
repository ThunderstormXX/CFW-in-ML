import copy
import torch
import torch.nn as nn


from ..model import MLP


def get_active_neurons(module, threshold=0):
    # Возвращает булев вектор, какие выходные нейроны активны
    mask = module.mask.abs()
    mean_mask = mask.mean(dim=1)  # среднее по входам
    return mean_mask > threshold

def prune_linear_layer(module, out_indices, in_indices):
    # Создаёт новый nn.Linear с весами обрезанными по индексам out_indices (выходы) и in_indices (входы)
    new_layer = nn.Linear(len(in_indices), len(out_indices), bias=module.linear.bias is not None)
    new_layer.weight.data = module.linear.weight.data[out_indices][:, in_indices].clone()
    if module.linear.bias is not None:
        new_layer.bias.data = module.linear.bias.data[out_indices].clone()
    return new_layer

def convert_masked_to_pruned_model(masked_model, input_dim=28*28, output_dim=10):
    pruned_layers = []
    prev_active_indices = None
    modules = [m for m in masked_model.modules() if isinstance(m, MaskedLinear)]
    
    for i, module in enumerate(modules):
        if i == len(modules) - 1:
            # Для последнего слоя не пруним выходные нейроны - оставляем все
            out_indices = torch.arange(module.linear.weight.shape[0])
        else:
            active_neurons = get_active_neurons(module)
            out_indices = torch.where(active_neurons)[0]

        in_indices = (
            torch.arange(module.linear.weight.shape[1])
            if prev_active_indices is None
            else prev_active_indices
        )
        
        pruned_layer = prune_linear_layer(module, out_indices, in_indices)
        pruned_layers.append(pruned_layer)
        prev_active_indices = out_indices
    
    # hidden_dims для новой модели
    hidden_dims = [layer.out_features for layer in pruned_layers[:-1]]
    pruned_output_dim = pruned_layers[-1].out_features
    
    # Создаём новую MLP с нужными размерами слоёв
    pruned_model = MLP(input_dim=input_dim, output_dim=pruned_output_dim, hidden_dims=hidden_dims)

    # Копируем веса
    pruned_model_layers = [module for module in pruned_model.model if isinstance(module, nn.Linear)]
    for new_layer, old_layer in zip(pruned_model_layers, pruned_layers):
        new_layer.weight.data = old_layer.weight.data.clone()
        if old_layer.bias is not None:
            new_layer.bias.data = old_layer.bias.data.clone()
        else:
            new_layer.bias = None

    return pruned_model


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.mask = nn.Parameter(torch.ones(out_features, in_features), requires_grad=False)

    def forward(self, x):
        masked_weight = self.linear.weight * self.mask
        return nn.functional.linear(x, masked_weight, self.linear.bias)


def apply_masked_linear(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            masked = MaskedLinear(module.in_features, module.out_features, module.bias is not None)
            masked.linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                masked.linear.bias.data.copy_(module.bias.data)
            setattr(model, name, masked)
        else:
            apply_masked_linear(module)



def train_masks(model, train_loader, device, n_epochs=1, lr=1e-1):
    mask_params = [m.mask for m in model.modules() if isinstance(m, MaskedLinear)]
    for p in mask_params:
        p.requires_grad = True

    optimizer = torch.optim.Adam(mask_params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    for epoch in range(n_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()




def prune_model(model, train_loader, device='cuda', input_dim=784, output_dim=10, prune_ratio=0.1, n_epochs=1):
    # 1. Копируем и оборачиваем модель
    masked_model = copy.deepcopy(model)
    for p in masked_model.parameters():
        p.requires_grad = False
    apply_masked_linear(masked_model)

    # 2. Обучаем маски
    train_masks(masked_model, train_loader, device=device, n_epochs=n_epochs)

    # 3. Применяем прунинг
    with torch.no_grad():
        for module in masked_model.modules():
            if isinstance(module, MaskedLinear):
                mean_per_neuron = module.mask.abs().mean(dim=1)
                n_prune = int(prune_ratio * mean_per_neuron.numel())
                if n_prune == 0:
                    continue
                prune_indices = torch.topk(mean_per_neuron, k=n_prune, largest=False).indices
                keep_mask = torch.ones_like(mean_per_neuron)
                keep_mask[prune_indices] = 0.0
                keep_mask = keep_mask.view(-1, 1)
                module.mask.data *= keep_mask

    # 4. Собираем финальную модель
    pruned_model = convert_masked_to_pruned_model(masked_model, input_dim=input_dim, output_dim=output_dim)
    return pruned_model
