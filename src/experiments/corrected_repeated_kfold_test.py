from collections import namedtuple
import numpy as np
import scipy.stats

CorrectedRepeatedKFoldCVTestResult = namedtuple("CorrectedRepeatedKFoldCVTestResult",
                                                ["statistic", "pvalue"])


def corrected_repeated_kfold_cv_test(a, b, k=5, r=10):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    x = a - b

    n2n1 = 1 / (k - 1)
    sigma2 = np.var(x, ddof=1)

    t = x.mean() / np.sqrt((1 / len(x) + n2n1) * sigma2)
    return CorrectedRepeatedKFoldCVTestResult(t, scipy.stats.t.sf(np.abs(t),
                                                                  k * r - 1))
