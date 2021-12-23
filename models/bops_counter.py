import math

import brevitas.nn as qnn
import numpy as np
import torch


def count_nonzero_weights(model):
    nonzero = total = 0
    layer_count_alive = {}
    layer_count_total = {}
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        layer_count_alive.update({name: nz_count})
        layer_count_total.update({name: total_params})
        nonzero += nz_count
        total += total_params
    return nonzero, total, layer_count_alive, layer_count_total


def calc_bops(model, input_data_precision=32):
    last_bit_width = input_data_precision
    alive, total, l_alive, l_total = count_nonzero_weights(model)
    b_w = model.weight_precision if hasattr(model, "weight_precision") else 32
    total_bops = 0
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            or isinstance(module, qnn.QuantLinear)
            or isinstance(module, torch.nn.Conv2d)
            or isinstance(module, qnn.QuantConv2d)
        ):
            b_a = last_bit_width
            # Dont think this is a property I can access sadly
            # Going with precision as given set in model
            # b_w = module.quant_weight_bit_width
            if isinstance(module, qnn.QuantConv2d):
                n = module.in_channels
                m = module.out_channels
            else:
                n = module.in_features
                m = module.out_features
            if isinstance(module, torch.nn.Conv2d) or isinstance(
                module, qnn.QuantConv2d
            ):
                k = np.prod(module.kernel_size)
            else:
                k = 1
            try:
                total = l_total[name + ".weight"] + l_total[name + ".bias"]
                alive = l_alive[name + ".weight"] + l_alive[name + ".bias"]
                p = 1 - ((total - alive) / total)  # fraction of layer remaining
            except KeyError:
                p = 0
            # assuming b_a is the output bitwidth of the last layer
            module_bops = (
                m * n * k * k * (p * b_a * b_w + b_a + b_w + math.log2(n * k * k))
            )
            last_bit_width = b_w
            total_bops += module_bops

    return total_bops
