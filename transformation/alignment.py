import numpy as np
import transformation
import scipy.ndimage.filters
import scipy.optimize
import scipy.linalg

def align_transformations(match_what, match_to):
    def objective(x):
        t = transformation.from_array(x)
        return np.sum(match_what.change_basis(t).disparity(match_to))
    constraint = { "type" : "eq", "fun" : (lambda x: (np.linalg.norm(x[:4]) - 1) ** 2) }
    x0 = transformation.Transformation().as_array()
    result = scipy.optimize.minimize(objective, x0, constraints=constraint)
    return transformation.from_array(result.x)

def align_timestamps(match_all, match_to):
    aligned_ixs = []
    i = 1
    for timestamp in match_all:
        while i < len(match_to) and match_to[i] < timestamp:
            i += 1
        aligned_ixs.append(i - 1)
    return aligned_ixs
