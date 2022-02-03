from .base import BasePipe
from nudging.dataset import MatrixData


class CreateMatrixData(BasePipe):
    def execute(self, X_nudge_outcome):
        return MatrixData.from_data(X_nudge_outcome[:3],
                                    truth=X_nudge_outcome[3])
