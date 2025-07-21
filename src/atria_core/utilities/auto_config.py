import functools
import inspect

from hydra_zen import builds
from omegaconf import OmegaConf


def auto_config(attr_name="config", exclude: set[str] | None = None):
    def decorator(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            if type(self) is cls:
                sig = inspect.signature(original_init)
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()
                config = {k: v for k, v in bound.arguments.items() if k != "self"}
                if "kwargs" in config:
                    config.update(config.pop("kwargs", {}))
                config = OmegaConf.create(
                    builds(cls, populate_full_signature=True, **config)
                )
                if exclude is not None:
                    config = OmegaConf.to_container(config)
                    config = {k: v for k, v in config.items() if k not in exclude}
                setattr(self, attr_name, config)
            original_init(self, *args, **kwargs)

        cls.__init__ = wrapped_init
        return cls

    return decorator
