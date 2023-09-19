import math

import hydra
from omegaconf import ListConfig, OmegaConf

OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

OmegaConf.register_new_resolver("int", lambda x: int(x))

OmegaConf.register_new_resolver("float", lambda x: float(x))

OmegaConf.register_new_resolver("tuple", lambda x: tuple(x))

OmegaConf.register_new_resolver("eval", lambda x: eval(x))

OmegaConf.register_new_resolver(
    "world_size",
    lambda x, y: int(x) * y if not isinstance(x, ListConfig) else len(x) * y,
)

# Registering a resolver that can return a callable method
# This also works on classes, but you can also register get_class which is almost identical.
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
