
def rm_modules(model):
    num_layers = len(model.model.layers)
    
    rm_modules = [(model.model.layers[n].mlp.gate_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].mlp.up_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].mlp.down_proj,'weight') for n in range(num_layers)]
    
    return tuple(rm_modules)

def all_rm_modules(model):
    num_layers = len(model.model.layers)
    
    rm_modules = [(model.model.layers[n].self_attn.q_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].self_attn.k_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].self_attn.v_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].self_attn.o_proj,'weight') for n in range(num_layers)]
    
    rm_modules = rm_modules + [(model.model.layers[n].mlp.gate_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].mlp.up_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].mlp.down_proj,'weight') for n in range(num_layers)]
    
    return tuple(rm_modules)

def get_vision_rm_modules(vision_model):
    """
    CLIPVisionModelのMLP層（fc1, fc2）を枝刈り対象として取得

    Args:
        vision_model: CLIPVisionModel (vision_tower.vision_tower)

    Returns:
        list: [(module, 'weight'), ...] のリスト
    """
    rm_modules = []

    # CLIPVisionModelの各エンコーダ層を処理
    for layer_idx, layer in enumerate(vision_model.vision_model.encoder.layers):
        # MLP層のみを対象（fc1とfc2）
        rm_modules.append((layer.mlp.fc1, 'weight'))
        rm_modules.append((layer.mlp.fc2, 'weight'))

    print(f"[Vision RM Modules] Total: {len(rm_modules)}")
    print(f"[Vision RM Modules] Layers: {len(vision_model.vision_model.encoder.layers)}")

    return rm_modules
