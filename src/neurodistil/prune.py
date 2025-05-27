import torch
import torch.nn as nn
import math
from copy import deepcopy
from ..model import MLP

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_mlp_with_pruned_layers(model: MLP, sparsity_level: float = 0.5):
    """
    Создаёт новую MLP, удаляя наименее важные нейроны (по L1-норме) без дообучения.
    """
    input_dim = model.model[0].in_features
    output_dim = model.model[-1].out_features
    original_total_params = count_parameters(model)
    target_total_params = int(original_total_params * (1 - sparsity_level))

    print(f"Original parameters: {original_total_params}")
    print(f"Target parameters: {target_total_params}")

    layers = list(model.model)
    linear_layers = [l for l in layers if isinstance(l, nn.Linear)]
    relu_flags = [isinstance(layers[i+1], nn.ReLU) if i + 1 < len(layers) else False
                  for i, l in enumerate(layers) if isinstance(l, nn.Linear)]
    dropout_list = [0.0] * (len(linear_layers) - 1)

    # Оценка важности нейронов по L1-норме весов
    pruned_dims = []
    for i, layer in enumerate(linear_layers[:-1]):  # не трогаем последний слой
        weight = layer.weight.data  # [out_features, in_features]
        importance = weight.abs().sum(dim=1)  # важность каждого нейрона
        orig_dim = weight.shape[0]

        if len(pruned_dims) == 0:
            prev_dims = [input_dim]
        else:
            prev_dims = pruned_dims

        # Эвристика: ищем размер, который в сумме даст не больше target_total_params
        # (с грубой прикидкой)
        total_params_so_far = sum(
            prev_dims[i] * d + d for i, d in enumerate(pruned_dims)
        )
        remaining_budget = target_total_params - total_params_so_far
        max_neurons = min(orig_dim, max(1, remaining_budget // (weight.shape[1] + 1)))

        topk = importance.topk(max_neurons).indices
        pruned_dims.append(len(topk))

    print(f"Pruned hidden dims: {pruned_dims}")

    # Создаём новую модель
    new_model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=pruned_dims,
        dropout_list=dropout_list,
        use_relu_list=relu_flags[:len(pruned_dims)],
    )

    # Копируем параметры (веса и смещения) для выбранных нейронов
    with torch.no_grad():
        src_idx = 0
        dst_idx = 0
        prev_topk = None

        for i, layer in enumerate(linear_layers[:-1]):
            dst_layer = [l for l in new_model.model if isinstance(l, nn.Linear)][i]
            weight = layer.weight.data
            bias = layer.bias.data
            importance = weight.abs().sum(dim=1)
            topk = importance.topk(pruned_dims[i]).indices

            # in_features
            if prev_topk is not None:
                dst_layer.weight.copy_(weight[topk][:, prev_topk])
            else:
                dst_layer.weight.copy_(weight[topk])

            dst_layer.bias.copy_(bias[topk])
            prev_topk = topk

        # Последний слой полностью копируем (без уменьшения)
        last_src = linear_layers[-1]
        last_dst = [l for l in new_model.model if isinstance(l, nn.Linear)][-1]
        if prev_topk is not None:
            last_dst.weight.copy_(last_src.weight[:, prev_topk])
        else:
            last_dst.weight.copy_(last_src.weight)
        last_dst.bias.copy_(last_src.bias)

    return new_model
