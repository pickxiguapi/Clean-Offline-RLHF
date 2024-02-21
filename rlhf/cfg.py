import re
from omegaconf import OmegaConf


def parse_cfg(cfg_path: str) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    base = OmegaConf.load(cfg_path / 'default.yaml')
    cli = OmegaConf.from_cli()
    for k,v in cli.items():
        if v == None:
            cli[k] = True
    base.merge_with(cli)

    # Algebraic expressions
    for k,v in base.items():
        if isinstance(v, str):
            match = re.match(r'(\d+)([+\-*/])(\d+)', v)
            if match:
                base[k] = eval(match.group(1) + match.group(2) + match.group(3))
                if isinstance(base[k], float) and base[k].is_integer():
                    base[k] = int(base[k])

    base.exp_name = str(base.get('exp_name', 'default'))
    return base
