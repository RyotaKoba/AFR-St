
def rm_modules(model):
    num_layers = len(model.model.layers)

    rm_modules = [(model.model.layers[n].mlp.gate_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].mlp.up_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].mlp.down_proj,'weight') for n in range(num_layers)]

    return tuple(rm_modules)

def get_vision_rm_modules(vision_model):
    rm_modules = []

    for layer_idx, layer in enumerate(vision_model.vision_model.encoder.layers):
        rm_modules.append((layer.mlp.fc1, 'weight'))
        rm_modules.append((layer.mlp.fc2, 'weight'))

    return rm_modules
