import numpy as np
from matplotlib import pyplot as plt

from .base import BasePipe
from .utils import create_corr_matrix, Bounds


def compute_avg_corr(corr_matrix):
    """Compute the average absolute correlations of the matrix

    Values that are exactly 0 are ignored.
    """
    tril_matrix = np.tril(corr_matrix, k=-1)
    return np.sum(np.abs(tril_matrix))/np.sum(tril_matrix != 0)


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    From: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the size of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve, scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this:
    return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming',"
                         "'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[window_len//2: -window_len//2+1]


class MatrixGen():
    """Class to generate correlation matrices

    The main difficulty is transforming eigen value powers
    into average correlations in a way that it doesn't stretch things
    too much or takes too long to search for a matrix with the right
    properties.
    """
    def __init__(self, n_features_uncorrelated, n_features_correlated):
        rng = np.random.default_rng(1293874)
        self.n_features_correlated = n_features_correlated
        self.n_features_uncorrelated = n_features_uncorrelated
        self.all_eigen = np.append(np.linspace(0, 1, 100),
                                   np.linspace(1.01, 50, 900))
        self.avg_corr = []
        self.corr_corr = []
        self.corr_uncorr = []
        for eigen_power in self.all_eigen:
            corr_matrix = create_corr_matrix(
                n_features_correlated=n_features_correlated,
                n_features_uncorrelated=n_features_uncorrelated,
                eigen_power=eigen_power, rng=rng)
            self.avg_corr.append(compute_avg_corr(corr_matrix))
            self.corr_corr.append(compute_avg_corr(
                corr_matrix[n_features_uncorrelated:, n_features_uncorrelated:]))
            self.corr_uncorr.append(np.mean(np.abs(corr_matrix[
                n_features_uncorrelated:, :n_features_uncorrelated])))
        self.avg_corr = np.array(self.avg_corr)

    @property
    def n_features(self):
        return self.n_features_uncorrelated + self.n_features_correlated + 2

    def get_matrix(self, correlation, nf_uncorrelated, nf_correlated):
        assert correlation >= 0 and correlation < 1
        i_eigen = self.get_eigen_idx(correlation, nf_uncorrelated,
                                     nf_correlated)
        all_avg_cor = []
        all_pre = []
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
        raise ValueError(f"Could not find correlation matrix with average "
                         "correlation of {correlation}")

    def get_eigen_idx(self, correlation, nf_uncorrelated, nf_correlated):
        if correlation == 0.0:
            return 0
        nf = nf_uncorrelated + nf_correlated
        pred_corr = np.array([
            (self.corr_corr[i]*nf_correlated + self.corr_uncorr[i]*nf_uncorrelated)/nf
            for i in range(len(self.corr_corr))])
        within_bounds = np.logical_and(pred_corr >= correlation,
                                       pred_corr < correlation + 0.03)
        return np.argmax(smooth(within_bounds.astype(int), 51)[1:])+1


class CorrMatrix(BasePipe):
    def __init__(self, n_features_uncorrelated=np.array([2, 10], dtype=int),
                 n_features_correlated=np.array([0, 10], dtype=int),
                 avg_correlation=np.array([0.05, 0.3])):
        self.n_features_uncorrelated = Bounds(n_features_uncorrelated,
                                              int_val=True)
        self.n_features_correlated = Bounds(n_features_correlated,
                                            int_val=True)
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
