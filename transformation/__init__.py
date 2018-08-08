import numpy as np
import quaternion

class Transformation:
    def __init__(self, q=None, t=None):
        if q is None:
            q = quaternion.one
        if t is None:
            t = np.zeros(q.shape + (3,))
        assert(t.shape == (q.shape + (3,)))
        self.q = standardize(q)
        self.t = t
        self.shape = self.q.shape

    def __repr__(self):
        return "({}, {})".format(self.q, self.t)

    def __mul__(self, other):
        if type(other) is Transformation:
            return Transformation(self * other.q, self * other.t)
        elif other.dtype is quaternion.one.dtype:
            return standardize(self.q * other)
        else:
            return rotate_vector(self.q, other) + self.t

    def __getitem__(self, index):
        return Transformation(q=self.q[index], t=self.t[index])

    def __setitem__(self, slice, value):
        self.q[index] = value.q
        self.t[index] = value.t

    def inverse(self):
        q_inv = 1 / self.q
        t_inv = -rotate_vector(q_inv, self.t)
        return Transformation(q=q_inv, t=t_inv)

    def change_basis(self, T):
        return T * self * T.inverse()

    def angle(self):
        return np.linalg.norm(quaternion.as_rotation_vector(self.q), axis=-1)

    def distance(self):
        return np.linalg.norm(self.t, axis=-1)

    def disparity(self, other):
        displacement = self * other.inverse()
        return displacement.angle() + displacement.distance()

    def axis(self):
        theta_v = quaternion.as_rotation_vector(self)
        theta = np.linalg.norm(quaternion.as_rotation_vector(self), axis=-1)
        return theta_v / theta.reshape(theta.shape + (1,))

    def diff(self, delay=1, axis=0):
        index_first = [slice(None)] * len(self.shape)
        index_first[axis] = slice(0, 1)
        index_delayed = [slice(None)] * len(self.shape)
        index_delayed[axis] = slice(0, -1)
        delayed = concatenate((self[index_first], self[index_delayed]))
        return delayed.inverse() * self

    def as_pose_matrix(self):
        R = quaternion.as_rotation_matrix(self.q)
        t = self.t.reshape(self.shape + (3, 1))
        return np.concatenate((R, t), axis=-1)

    def as_circumpolar(self):
        xyz = quaternion.as_rotation_vector(self.q)
        rpq = np.zeros(xyz.shape)
        xy = xyz[...,0]**2 + xyz[...,1]**2
        rpq[...,0] = np.sqrt(xy + xyz[...,2]**2)
        rpq[...,1] = np.arctan2(np.sqrt(xy), xyz[...,2])
        rpq[...,2] = np.arctan2(xyz[...,1], xyz[...,0])
        return rpq, self.t

    def as_array(self):
        return np.concatenate((quaternion.as_float_array(self.q), self.t), axis=-1)

def from_array(array):
    return Transformation(q=quaternion.from_float_array(array[...,:4]), t=array[...,4:])

def from_pose_matrix(matrix):
    q = quaternion.from_pose_matrix(matrix[...,:3])
    t = matrix[...,3]
    return Transformation(q=q, t=t)

def concatenate(Ts, axis=0):
    qs = np.concatenate([T.q for T in Ts], axis=axis)
    ts = np.concatenate([T.t for T in Ts], axis=axis)
    return Transformation(q=qs, t=ts)

def standardize(q):
    return q * np.sign(quaternion.as_float_array(q)[...,0])

def rotate_vector(qs, xyz):
    return quaternion.as_float_array(qs * quaternion_from_xyz(xyz) * (1 / qs))[...,1:]

def quaternion_from_xyz(xyz):
    ws = np.zeros(xyz.shape[:-1] + (1,))
    return quaternion.from_float_array(np.concatenate((ws, xyz), axis=-1))

def quat_diff(qs):
    qs_delayed = np.concatenate((qs[:1], qs[:-1]))
    return (1 / qs_delayed) * qs

def from_list(Ts):
    return Transformation(q=np.array([T.q for T in Ts]), t=np.array([T.t for T in Ts]))

def slerp(Ts, alphas):
    q = np.product([T.q ** alpha for T, alpha in zip(Ts, alphas)])
    t = np.sum([T.t * alpha for T, alpha in zip(Ts, alphas)], axis=0)
    return Transformation(q=q, t=t)
