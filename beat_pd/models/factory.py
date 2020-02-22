from importlib import import_module

def get_model_instance(model_config: dict):
    kwargs = model_config.get("kwargs", {})
    args = model_config.get("args", [])

    package = import_module(model_config['package'])
    model = getattr(package, model_config['type'])(*args, **kwargs)
    return model

def get_optimizer(model, optim_config: dict):
    kwargs = optim_config.get("kwargs", {})
    args = optim_config.get("args", [])


    package = import_module(optim_config.get('package', 'torch.optim'))
    optim = getattr(package, optim_config['type'])(model.parameters(), *args, **kwargs)

    return optim