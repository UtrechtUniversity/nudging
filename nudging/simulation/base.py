from abc import ABC
import inspect


def sig_to_param(signature):
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class BasePipe(ABC):
    pass

    @property
    def default_param(self):
        """Get the default parameters of the model.

        Returns
        -------
        dict:
            Dictionary with parameter: default value
        """
        cur_class = self.__class__
        default_parameters = sig_to_param(inspect.signature(self.__init__))
        while cur_class != BasePipe:
            signature = inspect.signature(super(cur_class, self).__init__)
            new_parameters = sig_to_param(signature)
            default_parameters.update(new_parameters)
            cur_class = cur_class.__bases__[0]
        return default_parameters
