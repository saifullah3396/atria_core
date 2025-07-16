import functools
import inspect

from hydra_zen import builds
from omegaconf import OmegaConf


def auto_config(attr_name="config"):
    def decorator(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            if type(self) is cls:
                sig = inspect.signature(original_init)
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()
                config = {k: v for k, v in bound.arguments.items() if k != "self"}
                setattr(
                    self,
                    attr_name,
                    OmegaConf.create(
                        builds(cls, populate_full_signature=True, **config)
                    ),
                )
            original_init(self, *args, **kwargs)

        cls.__init__ = wrapped_init
        return cls

    return decorator
