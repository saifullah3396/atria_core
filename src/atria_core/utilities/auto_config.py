import functools


class AutoConfig:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        original_init = cls.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            import inspect

            from hydra_zen import builds
            from omegaconf import OmegaConf

            original_init(self, *args, **kwargs)

            # Generate config after initialization
            sig = inspect.signature(original_init)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()

            config = {k: v for k, v in bound.arguments.items() if k != "self"}
            if "kwargs" in config:
                config.update(config.pop("kwargs", {}))

            config = OmegaConf.create(
                builds(cls, populate_full_signature=True, **config)
            )
            config = OmegaConf.to_container(config)
            self._config = config

        cls.__init__ = wrapped_init

    @property
    def config(self):
        """
        Returns the configuration of the dataset as a dictionary.
        """
        return self._config
