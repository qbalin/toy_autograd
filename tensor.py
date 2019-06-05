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

    print(jacobian_vector)
    if self.grad is None:
      self.grad = np.zeros(self.value.shape)

    self.grad += jacobian_vector.reshape(self.value.shape)

    if self.grad_fn == 'dot':
      self.lhs.backwards(self.calculate_jacobian(self.lhs.value, self.value, self.rhs.value, self.dot_wrt_lhs_derivative_fn) @ jacobian_vector)
      self.rhs.backwards(self.calculate_jacobian(self.rhs.value, self.value, self.lhs.value, self.dot_wrt_rhs_derivative_fn) @ jacobian_vector)

    if self.grad_fn == 'add':
      self.lhs.backwards(self.calculate_jacobian(self.lhs.value, self.value, None, self.add_derivative_fn) @ jacobian_vector)
      if isinstance(self.rhs, Tensor):
        self.rhs.backwards(self.calculate_jacobian(self.lhs.value, self.value, None, self.add_derivative_fn) @ jacobian_vector)

    if self.grad_fn == 'mul':
      self.lhs.backwards(self.calculate_jacobian(self.lhs.value, self.value, self.rhs.value if isinstance(self.rhs, Tensor) else self.rhs, self.mul_derivative_fn) @ jacobian_vector)
      if isinstance(self.rhs, Tensor):
        self.rhs.backwards(self.calculate_jacobian(self.rhs.value, self.value, self.lhs.value if isinstance(self.lhs, Tensor) else self.lhs, self.mul_derivative_fn) @ jacobian_vector)

    if self.grad_fn == 'mean':
      self.lhs.backwards(self.calculate_jacobian(self.lhs.value, self.value, None, self.mean_derivative_fn) @ jacobian_vector)

  def calculate_jacobian(self, x, y, other, derivative_fn):
    # y = f(x, other_tensor)
    # We want to compute dy / dx
    # y is splat horizontally, and x vertically:

    # x
    lin_x = x.shape[0]
    col_x = x.shape[1]

    # y
    lin_y = y.shape[0]
    col_y = y.shape[1]

    # Number of lines and columns of the jacobian
    lin = lin_x * col_x
    col = lin_y * col_y

    jacobian = np.zeros((lin, col))
    for i in range(lin):
      x_i = i // col_x
      x_j = i % col_x

      for j in range(col):
        y_i = j // col_y
        y_j = j % col_y

        jacobian[i, j] = derivative_fn(x, y, other, x_i, x_j, y_i, y_j)

    return jacobian

  def dot_wrt_lhs_derivative_fn(self, x, y, other, x_i, x_j, y_i, y_j):
    if x_i != y_i:
      return 0
    else:
      return other[x_j, y_j]

  def dot_wrt_rhs_derivative_fn(self, x, y, other, x_i, x_j, y_i, y_j):
    if x_j != y_j:
      return 0
    else:
      return other[y_i, x_i]

  def add_derivative_fn(self, x, y, other, x_i, x_j, y_i, y_j):
    if x_i == y_i and x_j == y_j:
      return 1
    else:
      return 0

  def mul_derivative_fn(self, x, y, other, x_i, x_j, y_i, y_j):
    if x_i == y_i and x_j == y_j:
      return other[x_i, x_j] if isinstance(other, np.ndarray) else other
    else:
      return 0

  def mean_derivative_fn(self, x, y, other, x_i, x_j, y_i, y_j):
    return 1. / x.size

