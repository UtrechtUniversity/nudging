from . import CorrMatrix, CreateFM, Linearizer
from . import GenNudgeOutcome, AddNoise, CreateMatrixData
from . import ConvertAge, ConvertGender, Categorical


class MatrixPipeline():
    """Class for a full simulation pipeline

    Individual elements can be disabled, but not all elements are
    optional:
    Mandatory elements:
        CorrMatrix
        CreateFM
        GenNudgeOutcome
        CreateMatrixData
    Optional elements:
        Linearizer
        AddNoise
        ConvertAge,
        ConvertGender,
        Categorical
    """
    def __init__(self, **kwargs):
        self._pipe_classes = [
            CorrMatrix,
            CreateFM,
            Linearizer,
            GenNudgeOutcome,
            AddNoise,
            CreateMatrixData,
            ConvertAge,
            ConvertGender,
            Categorical,
        ]
        self._pipe_kwargs = [{} for _ in range(len(self._pipe_classes))]
        for key, value in kwargs.items():
            key_found = False
            for i, pipe_class in enumerate(self._pipe_classes):
                params = pipe_class().default_param
                if key in params:
                    self._pipe_kwargs[i].update({key: value})
                    key_found = True
                    break
                if value is not None and value is not False:
                    continue
                if self._pipe_classes[i].__name__ == key:
                    del self._pipe_classes[i]
                    del self._pipe_kwargs[i]
                    key_found = True
                    break
            if not key_found:
                raise ValueError(f"Cannot find use for key `{key}` with value "
                                 "'{value}'.")

        self._pipe = [self._pipe_classes[i](**self._pipe_kwargs[i])
                      for i in range(len(self._pipe_classes))]

    def generate_one(self):
        x = (None, {})
        for executor in self._pipe:
            x = executor.execute(x)
        return x

    def generate(self, n):
        return [self.generate_one() for _ in range(n)]


def generate_datasets(n, **kwargs):
    pipe = MatrixPipeline(**kwargs)
    return pipe.generate(n)
