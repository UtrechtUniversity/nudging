import numpy as np
from nudging.simulation.utils import create_corr_matrix,\
    Bounds
from matplotlib import pyplot as plt
from abc import ABC
import inspect
from nudging.dataset.matrix import MatrixData
from scipy.ndimage import gaussian_filter1d


def compute_avg_corr(corr_matrix):
    tril_matrix = np.tril(corr_matrix, k=-1)
    return np.sum(np.abs(tril_matrix))/np.sum(tril_matrix != 0)



def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[window_len//2: -window_len//2+1]


class MatrixGen():
    def __init__(self, n_features_uncorrelated, n_features_correlated):
        rng = np.random.default_rng(1293874)
        self.n_features_correlated = n_features_correlated
        self.n_features_uncorrelated = n_features_uncorrelated
        self.all_eigen = np.append(np.linspace(0, 1, 100), np.linspace(1.01, 50, 900))
        self.avg_corr = []
        self.corr_corr = []
        self.corr_uncorr = []
        for eigen_power in self.all_eigen:
            corr_matrix = create_corr_matrix(
                n_features_correlated=n_features_correlated,
                n_features_uncorrelated=n_features_uncorrelated,
                eigen_power=eigen_power, rng=rng)
#             corr_matrix = corr_matrix[n_features_uncorrelated:, n_features_uncorrelated:]
            self.avg_corr.append(compute_avg_corr(corr_matrix))
            self.corr_corr.append(compute_avg_corr(
                corr_matrix[n_features_uncorrelated:, n_features_uncorrelated:]))
            self.corr_uncorr.append(np.mean(np.abs(corr_matrix[n_features_uncorrelated:, :n_features_uncorrelated])))
        self.avg_corr = np.array(self.avg_corr)

    @property
    def n_features(self):
        return self.n_features_uncorrelated + self.n_features_correlated + 2

    def get_matrix(self, correlation, nf_uncorrelated, nf_correlated):
        assert correlation >= 0 and correlation < 1
        i_eigen = self.get_eigen_idx(correlation, nf_uncorrelated, nf_correlated)
        all_avg_cor = []
        all_pre = []
        n_features = nf_correlated+nf_correlated+2
        avg_corr_matrix = np.zeros((self.n_features, self.n_features))
        for _ in range(1000):
            corr_matrix = create_corr_matrix(
                self.n_features_uncorrelated, self.n_features_correlated,
                self.all_eigen[i_eigen])
            avg_corr_matrix += np.fabs(corr_matrix)
            cols = np.concatenate((
                np.arange(nf_uncorrelated),
                np.arange(nf_correlated)+self.n_features_uncorrelated,
                [-2, -1]))
            all_pre.append(compute_avg_corr(corr_matrix))
            corr_matrix = corr_matrix[cols, :]
            corr_matrix = corr_matrix[:, cols]
            avg_cor = compute_avg_corr(corr_matrix)
            all_avg_cor.append(avg_cor)
            if avg_cor < correlation or avg_cor > correlation + 0.03:
                continue
            f = (correlation/avg_cor)
            corr_matrix = corr_matrix * f + (1-f)*np.identity(corr_matrix.shape[0])
            return corr_matrix
        self.get_eigen_idx(correlation, nf_uncorrelated, nf_correlated)
        plt.imshow(avg_corr_matrix)
        plt.show()
        print(i_eigen, np.min(all_avg_cor), np.max(all_avg_cor))
        raise ValueError(f"Could not find correlation matrix with average correlation of {correlation}")

    def get_eigen_idx(self, correlation, nf_uncorrelated, nf_correlated):
        if correlation == 0.0:
            return 0
        nf = nf_uncorrelated + nf_correlated
        pred_corr = np.array([
            (self.corr_corr[i]*nf_correlated + self.corr_uncorr[i]*nf_uncorrelated)/nf
            for i in range(len(self.corr_corr))])
        within_bounds = np.logical_and(pred_corr >= correlation, pred_corr < correlation + 0.03)
        return np.argmax(smooth(within_bounds.astype(int), 51)[1:])+1


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

#     def __getattr__(self, key):
#         if key[-1] == "_" and key[:-1] in self.default_param:
#             return super().__getattr__(key[:-1])
#         val = super().__getattr__(key)
#         if key in self.default_param:
#             int_val = val[1]
#             if int_val:
#                 return np.random.randint(*val[0])
#             r = np.random.rand()
#             return r*(val[0][1]-val[0][0]) + val[0][0]
#         return val


class CorrMatrix(BasePipe):
    def __init__(self, n_features_uncorrelated=np.array([2, 10], dtype=int),
                 n_features_correlated=np.array([0, 10], dtype=int),
                 avg_correlation=np.array([0.05, 0.3])):
        self.n_features_uncorrelated = Bounds(n_features_uncorrelated, int_val=True)
        self.n_features_correlated = Bounds(n_features_correlated, int_val=True)
        self.avg_correlation = Bounds(avg_correlation)
        self.matrix_gen = MatrixGen(self.n_features_uncorrelated.max(),
                                    self.n_features_correlated.max())

    def execute(self, data):
        truth = data[1]
        avg_corr = self.avg_correlation.rand()
        n_features_uncorrelated = self.n_features_uncorrelated.rand()
        n_features_correlated = self.n_features_correlated.rand()
        corr_matrix = self.matrix_gen.get_matrix(
            avg_corr, n_features_uncorrelated, n_features_correlated)

        truth.update({
            "avg_correlation": avg_corr,
            "n_features_uncorrelated": n_features_uncorrelated,
            "n_features_correlated": n_features_correlated,
        })
        return corr_matrix, truth


class CreateFM(BasePipe):
    def __init__(self,
                 n_samples=np.array([500, 5000], dtype=int),
                 nudge_avg=np.array([0.05, 0.5]),
                 control_unique=np.array([0, 1.0]),
                 control_precision=np.array([0.2, 1.0])):
        self.n_samples = Bounds(n_samples, int_val=True)
        self.nudge_avg = Bounds(nudge_avg)
        self.control_unique = Bounds(control_unique)
        self.control_precision = Bounds(control_precision)

    def execute(self, data):
        corr_matrix, truth = data
        n_samples = self.n_samples.rand()
        nudge_avg = self.nudge_avg.rand()
        control_unique = self.control_unique.rand()
        control_precision = self.control_precision.rand()

        n_features = corr_matrix.shape[0]-2
        L = np.linalg.cholesky(corr_matrix)
        X = np.dot(L, np.random.randn(n_features+2, n_samples)).T
        control_intrinsic = X[:, -1]
        nudge_intrinsic = X[:, -2]
        X = X[:, :-2]

        true_outcome_control = (control_intrinsic*control_unique
                                + (1-control_unique)*nudge_intrinsic)
        true_outcome_control *= control_precision
        true_outcome_nudge = nudge_intrinsic + nudge_avg

        truth.update({
            "n_samples": n_samples,
            "nudge_avg": nudge_avg,
            "control_unique": control_unique,
            "control_precision": control_precision,
        })
        return (X, true_outcome_control, true_outcome_nudge, truth)


def _transform_outcome(outcome, a, powers=np.array([1, 0.5, 0.1])):
    ret_outcome = np.zeros_like(outcome)
    a *= powers
    for i in range(len(a)):
        ret_outcome += (a[i]-powers[i]/2)*outcome**(i+1)
    return ret_outcome


class Linearizer(BasePipe):
    def __init__(self, linear=None):
        self.linear = linear

    def execute(self, X_true_outcome):
        X, true_outcome_control, true_outcome_nudge, truth = X_true_outcome
        if self.linear is None:
            linear = np.bool_(np.random.randint(2))
        else:
            linear = self.linear
        truth["linear"] = linear

        if linear:
            return X_true_outcome
        n_features = X.shape[1]

        a = np.random.rand(3)
        true_outcome_control = _transform_outcome(true_outcome_control, a)
        true_outcome_nudge = _transform_outcome(true_outcome_nudge, a)
        for i_col in range(n_features):
            a = np.random.rand(3)
            X[:, i_col] = _transform_outcome(X[:, i_col], a)

        return (X, true_outcome_control, true_outcome_nudge, truth)


class GenNudgeOutcome(BasePipe):
    def __init__(self, balance=np.array([0.2, 0.8])):
        self.balance = Bounds(balance)

    def execute(self, X_true_outcome):
        balance = self.balance.rand()
        X, true_outcome_control, true_outcome_nudge, truth = X_true_outcome
        n_samples = truth["n_samples"]
        n_treat = round(n_samples*balance)
        n_treat = min(max(2, n_treat), n_samples-2)
        nudge = np.zeros(n_samples, dtype=int)
        nudge[np.random.choice(n_samples, n_treat, replace=False)] = 1
        cate = true_outcome_nudge - true_outcome_control
        outcome = (true_outcome_control*(1-nudge)
                   + true_outcome_nudge*nudge)
        truth["balance"] = balance
        truth["cate"] = cate
        return (X, nudge, outcome, truth)


class AddNoise(BasePipe):
    def __init__(self, noise_frac=np.array([0, 0.99])):
        self.noise_frac = Bounds(noise_frac)

    def execute(self, X_nudge_outcome):
        noise_frac = self.noise_frac.rand()
        X, nudge, outcome, truth = X_nudge_outcome
        n_samples = X.shape[0]
        outcome += (noise_frac/(1-noise_frac))*np.random.randn(n_samples)
        truth["noise_frac"] = noise_frac
        return (X, nudge, outcome, truth)


class CreateMatrixData(BasePipe):
    def execute(self, X_nudge_outcome):
        return MatrixData.from_data(X_nudge_outcome[:3], truth=X_nudge_outcome[3])


def find_free_col(X, n_features_uncorrelated=None):
    feature_names = list(X.standard_df)
    if n_features_uncorrelated is None:
        np.random.shuffle(feature_names)
    for i, name in enumerate(feature_names):
        try:
            int(name)
            return name
        except ValueError:
            pass
        if n_features_uncorrelated is not None and i >= n_features_uncorrelated:
            break
    raise ValueError("Cannot find free column in conversion process.")


def rescale(var, min_value, max_value):
    """ Rescale data to given range
    Args:
        var (series): data to be rescaled
        min_value (int): min value of new range
        max_value (int): max value of new range
    Returns:
        series: rescaled data
    """
    result = (((var - var.min())/(var.max() - var.min() + 1e-12)) *
              (max_value - min_value) + min_value)
    return result.astype(int)


class ConvertAge(BasePipe):
    def execute(self, X):
        truth = X.truth
        col = find_free_col(X, n_features_uncorrelated=truth["n_features_uncorrelated"])
        X.standard_df[col] = rescale(X.standard_df[col].values, 18, 80)
        X.standard_df.rename(columns={col: "age"}, inplace=True)
        if "n_rescale" in truth:
            truth["n_rescale"] += 1
        else:
            truth["n_rescale"] = 1
        return X


class ConvertGender(BasePipe):
    def execute(self, X):
        truth = X.truth
        col = find_free_col(X, n_features_uncorrelated=truth["n_features_uncorrelated"])
        X.standard_df[col] = rescale(X.standard_df[col].values, 0, 2)
        X.standard_df.rename(columns={col: "gender"}, inplace=True)
        if "n_rescale" in truth:
            truth["n_rescale"] += 1
        else:
            truth["n_rescale"] = 1
        return X


class Categorical(BasePipe):
    def __init__(self, n_rescale=np.array([0, 4], dtype=int),
                 n_layers=np.array([0, 5], dtype=int)):
        self.n_rescale = Bounds(n_rescale, int_val=True)
        self.n_layers = Bounds(n_layers, int_val=True)

    def execute(self, X):
        truth = X.truth
        n_rescale = self.n_rescale.rand()
        if "n_rescale" not in truth:
            truth["n_rescale"] = 0
        for _ in range(n_rescale):
            try:
                col = find_free_col(X)
            except ValueError:
                break
            X.standard_df[col] = rescale(X.standard_df[col].values, 0, self.n_layers.rand())
            X.standard_df.rename(columns={col: f"layer_{col}"}, inplace=True)
            truth["n_rescale"] += 1
        return X


class MatrixPipeline():
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
                if value:
                    continue
                if self._pipe_classes[i].__name__ == key:
                    del self._pipe_classes[i]
                    del self._pipe_kwargs[i]
                    key_found = True
                    break
            if not key_found:
                raise ValueError(f"Cannot find use for key `{key}` with value '{value}'.")

        self._pipe = [self._pipe_classes[i](**self._pipe_kwargs[i]) for i in range(len(self._pipe_classes))]

    def generate_one(self):
        x = (None, {})
        for executor in self._pipe:
            x = executor.execute(x)
        return x

    def generate(self, n):
        return [self.generate_one() for _ in range(n)]


