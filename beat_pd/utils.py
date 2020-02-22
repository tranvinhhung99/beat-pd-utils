import yaml

def parse_args(config_path, config={}):
    config.update(
        yaml.load(open(config_path))
    )
    return config