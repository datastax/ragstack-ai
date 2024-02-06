from abc import ABCMeta


class Singleton(ABCMeta):
    """
    Metaclass for creating singletons.

    Note this is not a thread-safe implementation.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            instance = cls._instances[cls]
            # Allow reinitialization if the class has the attribute __allow_reinitialization
            if (
                hasattr(cls, "__allow_reinitialization")
                and cls.__allow_reinitialization
            ):
                instance.__init__(*args, **kwargs)
        return instance
