import numpy as np
import pdb


class Tensor:
  def __init__(self, value, requires_grad = False, grad_fn = None, lhs = None, rhs = None):
    self.requires_grad = requires_grad
    self.value = value
    self.grad = None
    self.grad_fn = grad_fn
    self.lhs = lhs
    self.rhs = rhs

  def __matmul__(self, other):
    requires_grad = self.requires_grad or other.requires_grad
    value = self.value @ other.value
    lhs = self
    rhs = other

    return Tensor(value, grad_fn = 'dot', requires_grad = requires_grad, lhs = lhs, rhs = rhs)

  def __add__(self, other):
    requires_grad = self.requires_grad or (isinstance(other, Tensor) and other.requires_grad)
    other_value = other.value if isinstance(other, Tensor) else other
    value = self.value + other_value
    lhs = self
    rhs = other

    return Tensor(value, grad_fn = 'add', requires_grad = requires_grad, lhs = lhs, rhs = rhs)

  def __mul__(self, other):
    requires_grad = self.requires_grad or (isinstance(other, Tensor) and other.requires_grad)
    other_value = other.value if isinstance(other, Tensor) else other
    value = self.value * other_value
    lhs = self
    rhs = other

    return Tensor(value, grad_fn = 'mul', requires_grad = requires_grad, lhs = lhs, rhs = rhs)

  def mean(self):
    requires_grad = self.requires_grad
    value = np.array([[self.value.mean()]])
    lhs = self

    return Tensor(value, grad_fn = 'mean', requires_grad = requires_grad, lhs = lhs)

  def backwards(self, jacobian_vector = np.array([[1.]])):
    if self.requires_grad == False:
      return

    self.grad = jacobian_vector

    jacobian_vector = jacobian_vector.reshape((jacobian_vector.shape[0] * jacobian_vector.shape[1], 1))

    if self.grad_fn == 'dot':
      self.lhs.backwards((self.dot_jacobian_wrt_lhs() @ jacobian_vector).reshape(self.lhs.value.shape))
      self.rhs.backwards((self.dot_jacobian_wrt_rhs() @ jacobian_vector).reshape(self.rhs.value.shape))

    if self.grad_fn == 'add':
      self.lhs.backwards((self.add_jacobian_wrt_mat(self.lhs.value) @ jacobian_vector).reshape(self.lhs.value.shape))
      if isinstance(self.rhs, Tensor):
        self.rhs.backwards((self.add_jacobian_wrt_mat(self.rhs.value) @ jacobian_vector).reshape(self.rhs.value.shape))

    if self.grad_fn == 'mul':
      self.lhs.backwards((self.mul_jacobian_wrt_mat(self.lhs.value, self.rhs) @ jacobian_vector).reshape(self.lhs.value.shape))
      if isinstance(self.rhs, Tensor):
        self.rhs.backwards((self.mul_jacobian_wrt_mat (self.rhs.value, self.lhs) @ jacobian_vector).reshape(self.rhs.value.shape))

    if self.grad_fn == 'mean':
      print((self.mean_jacobian() @ jacobian_vector).reshape(self.lhs.value.shape))
      self.lhs.backwards((self.mean_jacobian() @ jacobian_vector).reshape(self.lhs.value.shape))

  def dot_jacobian_wrt_rhs(self):
    # Z = X * Y
    # We want to compute dZ / dY
    # Y is splat horizontally, and Z vertically

    # Y
    lin_rhs = self.rhs.value.shape[0]
    col_rhs = self.rhs.value.shape[1]

    # Z
    lin_value = self.value.shape[0]
    col_value = self.value.shape[1]


    lin = lin_rhs * col_rhs
    col = lin_value * col_value

    jacobian = np.zeros((lin, col))
    for i in range(lin):
      for j in range(col):
        rhs_i = i // col_rhs
        rhs_j = i % col_rhs

        value_i = j // col_value
        value_j = j % col_value

        if rhs_j != value_j:
          jacobian[i, j] = 0
        else:
          jacobian[i, j] = self.lhs.value[value_i, rhs_i]

    return jacobian

  def dot_jacobian_wrt_lhs(self):
    # Z = X * Y
    # We want to compute dZ / dX
    # X is splat horizontally, and Z vertically

    # X
    lin_lhs = self.lhs.value.shape[0]
    col_lhs = self.lhs.value.shape[1]

    # Z
    lin_value = self.value.shape[0]
    col_value = self.value.shape[1]


    lin = lin_lhs * col_lhs
    col = lin_value * col_value

    jacobian = np.zeros((lin, col))
    for i in range(lin):
      for j in range(col):
        lhs_i = i // col_lhs
        lhs_j = i % col_lhs

        value_i = j // col_value
        value_j = j % col_value

        if lhs_i != value_i:
          jacobian[i, j] = 0
        else:
          jacobian[i, j] = self.rhs.value[lhs_j, value_j]

    return jacobian

  def add_jacobian_wrt_mat(self, mat):
    # Z = X + Y
    # We want to compute dZ / dY
    # Y is splat horizontally, and Z vertically

    # Y
    lin_mat = mat.shape[0]
    col_mat = mat.shape[1]

    # Z
    lin_value = self.value.shape[0]
    col_value = self.value.shape[1]


    lin = lin_mat * col_mat
    col = lin_value * col_value

    jacobian = np.zeros((lin, col))
    for i in range(lin):
      for j in range(col):
        mat_i = i // col_mat
        mat_j = i % col_mat

        value_i = j // col_value
        value_j = j % col_value

        if mat_i == value_i and mat_j == value_j:
          jacobian[i, j] = 1
        else:
          jacobian[i, j] = 0

    return jacobian

  def mul_jacobian_wrt_mat(self, mat, tensor_or_value):
    # Z = X * Y
    # We want to compute dZ / dY
    # Y is splat horizontally, and Z vertically


    lin_mat = mat.shape[0]
    col_mat = mat.shape[1]

    lin = lin_mat * col_mat
    col = lin

    jacobian = np.zeros((lin, col))
    for i in range(lin):
      mat_i = i // col_mat
      mat_j = i % col_mat

      for j in range(col):
        if i == j:
          jacobian[i, j] = tensor_or_value.value[mat_i, mat_j] if isinstance(tensor_or_value, Tensor) else tensor_or_value
        else:
          jacobian[i, j] = 0

    return jacobian

  def mean_jacobian(self):
    # Z = X * Y
    # We want to compute dZ / dY
    # Y is splat horizontally, and Z vertically

    # Y
    lin_lhs = self.lhs.value.shape[0]
    col_lhs = self.lhs.value.shape[1]

    # Z
    lin_value = self.value.shape[0]
    col_value = self.value.shape[1]


    lin = lin_lhs * col_lhs
    col = lin_value * col_value

    jacobian = np.zeros((lin, col))
    for i in range(lin):
      for j in range(col):
        rhs_i = i // col_lhs
        rhs_j = i % col_lhs

        value_i = j // col_value
        value_j = j % col_value

        jacobian[i, j] = 1. / lin_lhs


    return jacobian








