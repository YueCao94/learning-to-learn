# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Learning to learn (meta) optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import os

import mock
import sonnet as snt
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.util import nest

import networks

import pdb


def _nested_assign(ref, value):
  """Returns a nested collection of TensorFlow assign operations.

  Args:
    ref: Nested collection of TensorFlow variables.
    value: Values to be assigned to the variables. Must have the same structure
        as `ref`.

  Returns:
    Nested collection (same structure as `ref`) of TensorFlow assign operations.

  Raises:
    ValueError: If `ref` and `values` have different structures.
  """
  if isinstance(ref, list) or isinstance(ref, tuple):
    if len(ref) != len(value):
      raise ValueError("ref and value have different lengths.")
    result = [_nested_assign(r, v) for r, v in zip(ref, value)]
    if isinstance(ref, tuple):
      return tuple(result)
    return result
  else:
    return tf.assign(ref, value)


def _nested_variable(init, name=None, trainable=False):
  """Returns a nested collection of TensorFlow variables.

  Args:
    init: Nested collection of TensorFlow initializers.
    name: Variable name.
    trainable: Make variables trainable (`False` by default).

  Returns:
    Nested collection (same structure as `init`) of TensorFlow variables.
  """
  if isinstance(init, list) or isinstance(init, tuple):
    result = [_nested_variable(i, name, trainable) for i in init]
    if isinstance(init, tuple):
      return tuple(result)
    return result
  else:
    return tf.Variable(init, name=name, trainable=trainable)


def _wrap_variable_creation(func, custom_getter):
  """Provides a custom getter for all variable creations."""
  original_get_variable = tf.get_variable
  def custom_get_variable(*args, **kwargs):
    if hasattr(kwargs, "custom_getter"):
      raise AttributeError("Custom getters are not supported for optimizee "
                           "variables.")
    return original_get_variable(*args, custom_getter=custom_getter, **kwargs)

  # Mock the get_variable method.
  with mock.patch("tensorflow.get_variable", custom_get_variable):
    return func()


def _get_variables(func):
  """Calls func, returning any variables created, but ignoring its return value.

  Args:
    func: Function to be called.

  Returns:
    A tuple (variables, constants) where the first element is a list of
    trainable variables and the second is the non-trainable variables.
  """
  variables = []
  constants = []

  def custom_getter(getter, name, **kwargs):
    trainable = kwargs["trainable"]
    kwargs["trainable"] = False
    variable = getter(name, **kwargs)
    if trainable:
      variables.append(variable)
    else:
      constants.append(variable)
    return variable

  with tf.name_scope("unused_graph"):
    _wrap_variable_creation(func, custom_getter)

  return variables, constants


def _make_with_custom_variables(func, variables):
  """Calls func and replaces any trainable variables.

  This returns the output of func, but whenever `get_variable` is called it
  will replace any trainable variables with the tensors in `variables`, in the
  same order. Non-trainable variables will re-use any variables already
  created.

  Args:
    func: Function to be called.
    variables: A list of tensors replacing the trainable variables.

  Returns:
    The return value of func is returned.
  """
  variables = collections.deque(variables)

  def custom_getter(getter, name, **kwargs):
    if kwargs["trainable"]:
      return variables.popleft()
    else:
      kwargs["reuse"] = True
      return getter(name, **kwargs)

  return _wrap_variable_creation(func, custom_getter)


MetaLoss = collections.namedtuple("MetaLoss", "loss, update, reset, fx, x")
MetaStep = collections.namedtuple("MetaStep", "step, update, reset, fx, x")


def _make_nets(variables, config, net_assignments):
  """Creates the optimizer networks.

  Args:
    variables: A list of variables to be optimized.
    config: A dictionary of network configurations, each of which will be
        passed to networks.Factory to construct a single optimizer net.
    net_assignments: A list of tuples where each tuple is of the form (netid,
        variable_names) and is used to assign variables to networks. netid must
        be a key in config.

  Returns:
    A tuple (nets, keys, subsets) where nets is a dictionary of created
    optimizer nets such that the net with key keys[i] should be applied to the
    subset of variables listed in subsets[i].

  Raises:
    ValueError: If net_assignments is None and the configuration defines more
        than one network.
  """
  # create a dictionary which maps a variable name to its index within the
  # list of variables.
  name_to_index = dict((v.name.split(":")[0], i)
                       for i, v in enumerate(variables))

  if net_assignments is None:
    if len(config) != 1:
      raise ValueError("Default net_assignments can only be used if there is "
                       "a single net config.")

    with tf.variable_scope("vars_optimizer"):
      key = next(iter(config))
      kwargs = config[key]
      net = networks.factory(**kwargs)

    nets = {key: net}
    keys = [key]
    subsets = [range(len(variables))]
  else:
    nets = {}
    keys = []
    subsets = []
    with tf.variable_scope("vars_optimizer"):
      for key, names in net_assignments:
        if key in nets:
          raise ValueError("Repeated netid in net_assigments.")
        nets[key] = networks.factory(**config[key])
        subset = [name_to_index[name] for name in names]
        keys.append(key)
        subsets.append(subset)
        print("Net: {}, Subset: {}".format(key, subset))

  # subsets should be a list of disjoint subsets (as lists!) of the variables
  # and nets should be a list of networks to apply to each subset.
  return nets, keys, subsets


class MetaOptimizer(object):
  """Learning to learn (meta) optimizer.

  Optimizer which has an internal RNN which takes as input, at each iteration,
  the gradient of the function being minimized and returns a step direction.
  This optimizer can then itself be optimized to learn optimization on a set of
  tasks.
  """

  def __init__(self, **kwargs):
    """Creates a MetaOptimizer.

    Args:
      **kwargs: A set of keyword arguments mapping network identifiers (the
          keys) to parameters that will be passed to networks.Factory (see docs
          for more info).  These can be used to assign different optimizee
          parameters to different optimizers (see net_assignments in the
          meta_loss method).
    """
    self._nets = None
    self.num_lstm = 2

    if not kwargs:
      # Use a default coordinatewise network if nothing is given. this allows
      # for no network spec and no assignments.
      self._config = {
          "coordinatewise": {
              "net": "CoordinateWiseDeepLSTM",
              "net_options": {
                  "layers": (20, 20),
                  "preprocess_name": "LogAndSign",
                  "preprocess_options": {"k": 5},
                  "scale": 0.01,
              }}}
    else:
      self._config = kwargs

  def save(self, sess, path=None):
    """Save meta-optimizer."""
    result = {}
    for k, net in self._nets.items():
      if path is None:
        filename = None
        key = k
      else:
        filename = os.path.join(path, "{}.l2l".format(k))
        key = filename
      net_vars = networks.save(net, sess, filename=filename)
      result[key] = net_vars
    return result

  def meta_loss(self,
                make_loss,
                len_unroll,
                net_assignments=None,
                second_derivatives=False):
    """Returns an operator computing the meta-loss.

    Args:
      make_loss: Callable which returns the optimizee loss; note that this
          should create its ops in the default graph.
      len_unroll: Number of steps to unroll.
      net_assignments: variable to optimizer mapping. If not None, it should be
          a list of (k, names) tuples, where k is a valid key in the kwargs
          passed at at construction time and names is a list of variable names.
      second_derivatives: Use second derivatives (default is false).

    Returns:
      namedtuple containing (loss, update, reset, fx, x)
    """

    # Construct an instance of the problem only to grab the variables. This
    # loss will never be evaluated.
    
    
    sub_x, sub_constants=_get_variables(make_loss)
    x=[sub_x for i in range(self.num_lstm)]
    constants=[sub_constants for i in range(self.num_lstm)]
    print("x.length",len(x))
    print("Optimizee variables")
    print([op.name for op in x[0]])
    print("Problem variables")
    print([op.name for op in constants[0]])

    # Create the optimizer networks and find the subsets of variables to assign
    # to each optimizer.
    nets, net_keys, subsets = _make_nets(x[0], self._config, net_assignments) 

    # Store the networks so we can save them later.
    self._nets = nets

    # Create hidden state for each subset of variables.
    state = []
    for z in range(self.num_lstm):
      single_state = []
      with tf.name_scope("states"):
        for i, (subset, key) in enumerate(zip(subsets, net_keys)):
          net = nets[key]
          with tf.name_scope("state_{}".format(i)):
            single_state.append(_nested_variable(
                [net.initial_state_for_inputs(x[z][j],dtype=tf.float32)
                for j in subset],
                name="state", trainable=False))
      state.append(single_state) 
   
    def update(net, fx, x, state):
      """Parameter and RNN state update."""
      with tf.name_scope("gradients"):
        gradients = tf.gradients(fx, x)

        # Stopping the gradient here corresponds to what was done in the
        # original L2L NIPS submission. However it looks like things like
        # BatchNorm, etc. don't support second-derivatives so we still need
        # this term.
        if not second_derivatives:
          gradients = [tf.stop_gradient(g) for g in gradients]

      with tf.name_scope("deltas"):
        deltas, state_next = zip(*[net(g, s) for g, s in zip(gradients, state)])
        state_next = list(state_next)

      return deltas, state_next
    
    def inter_step_init(t,sub_fx_array, x, state):
#      pdb.set_trace()
      '''
      inter过程中对单个lstm初始化
      '''
      x_now = list(x)
      x_next = list(x)
      sub_state_next = []
      #获得变量
      with tf.name_scope("fx"):
        fx = _make_with_custom_variables(make_loss, x)
        sub_fx_array = sub_fx_array.write(t, fx)
      #求导
      with tf.name_scope("dx"):
        for subset, key, s_i in zip(subsets, net_keys, state):
          x_i = [x[j] for j in subset]
          deltas, s_i_next = update(nets[key], fx, x_i, s_i)

          for idx, j in enumerate(subset):
            x_next[j] += deltas[idx]
          sub_state_next.append(s_i_next)
      return sub_fx_array, x_now, x_next, sub_state_next

    
    def time_step(t, fx_array, x, state):
      """While loop body."""
#      pdb.set_trace()
#      print("x.length",len(x))
#      print("fx_array.length",len(fx_array))
#      print("state.length",len(state))
      Fx_array = []#得到所有lstm对应的mnist的fx的数组
      x_now=[]#当前的x
      x_next = []#迭代后的x
      state_next=[]#下一个时间的状态
      print(fx_array)
      for i in range(self.num_lstm):
#        pdb.set_trace()
        sub_fx_array, sub_x_now, sub_x_next ,sub_state_next = inter_step_init(t,fx_array[i], x[i], state[i])
        Fx_array.append(sub_fx_array)
        x_now.append(sub_x_now)
        x_next.append(sub_x_next)
        state_next.append(sub_state_next)
      
      def convert_to_mat(list_x):
#        pdb.set_trace()
#        print(list_x)
      #将 变量数组转化为矩阵方便inter计算
      #mnist是mlp模型共两层
      #第一层的weights
        mat_1=tf.concat([tf.reshape(mat[0],[1,15680]) for mat in list_x],axis=0)
      
        print('mat_1.shape',mat_1)
        #第一层的bias
        mat_2=tf.concat([tf.reshape(mat[1],[1,20]) for mat in list_x],axis=0)
        print('mat_2.shape',mat_2)
        #第二层的weights
        mat_3=tf.concat([tf.reshape(mat[2],[1,200]) for mat in list_x],axis=0)
        print('mat_3.shape',mat_3)
        第二层的bias
        mat_4=tf.concat([tf.reshape(mat[3],[1,10]) for mat in list_x],axis=0)
        print('mat_4.shape',mat_4)
        return [mat_1, mat_2 ,mat_3, mat_4]
      
      def convert_to_Variable(list_x):
      #将矩阵转化为变量数组进行迭代
#        pdb.set_trace()
        shape=([784,20],[-1],[20,10],[-1])
        Variable=[]
        for j in range(self.num_lstm):
          variable=[]
          for i in range(4):
            mat_x=list_x[i]
            variable.append(tf.reshape(mat_x[j],shape[i]))
            print(variable)
          Variable.append(variable)
        return Variable
      
      
      def inter_atten(mat_a,mat_b):
#        pdb.set_trace()
        #inter attention 的核心组件
        l=1
        grad = mat_b#梯度
        origin = mat_a#初始变量
        def attention(Mat_a,Mat_b):
          result = tf.matmul(Mat_a,Mat_b)
          Mat_b = tf.add(Mat_b,result)
          return Mat_b
        matmul1 = tf.matmul(grad,tf.transpose(grad))
        matmul2 = tf.matmul(origin,tf.transpose(origin))
        matmul2 = tf.exp(matmul2/2*l)
        softmax = tf.nn.softmax(matmul1)
        input_mul_x = tf.multiply(softmax,matmul2)
        e_ij = attention(input_mul_x,grad)
        return e_ij
      #将变量数组转化为矩阵
      mat_x_now=convert_to_mat(x_now)
      mat_x_next=convert_to_mat(x_next)
      #inter attention计算
      inter_mat=[inter_atten(mat_x_now[i],mat_x_next[i]) for i in range(4)]
      print(inter_mat)
      #将计算后的矩阵转成变量数组
      x_next=convert_to_Variable(inter_mat)
      print('x_next',x_next)

      with tf.name_scope("t_next"):
        t_next = t + 1
      print('fx_array',fx_array)
      print('x_next',x_next)
      print('state_next',state_next)

      return t_next, Fx_array, x_next, state_next

    # Define the while loop.
    fx_array = [tf.TensorArray(tf.float32, size=len_unroll + 1,
                              clear_after_read=False) for i in range(self.num_lstm)]
#    print(fx_array[1])
    _, fx_array, x_final, s_final = tf.while_loop(
        cond=lambda t, *_: t < len_unroll,
        body=time_step,
        loop_vars=(0, fx_array, x, state),
        parallel_iterations=1,
        swap_memory=True,
        name="unroll")

    with tf.name_scope("fx"):
#      pdb.set_trace()
      print('x_final',x_final)
      fx_final = [_make_with_custom_variables(make_loss, sub_x_final) for sub_x_final in x_final]
      print('fx_final',fx_final)
      print(len(fx_final))
#     print('fx_array[1]',fx_array[1])
      for i in range(self.num_lstm):
        fx_array[i].write(len_unroll, fx_final[i])
#    pdb.set_trace()
    print(fx_array)
    loss = [tf.reduce_sum(sub_fx_array.stack(), name="loss") for sub_fx_array in fx_array]

    # Reset the state; should be called at the beginning of an epoch.
    with tf.name_scope("reset"):
      reset=[]
      for i in range(self.num_lstm):

        variables = (nest.flatten(state[i]) +
                    x[i] + constants[i])
      # Empty array as part of the reset process.
        reset.append([tf.variables_initializer(variables), fx_array[i].close()])

    # Operator to update the parameters and the RNN state after our loop, but
    # during an epoch.
    with tf.name_scope("update"):
#      pdb.set_trace()
      update = []
      for i in range(self.num_lstm):
          update.append((nest.flatten(_nested_assign(x[i],  x_final[i])) +
                nest.flatten(_nested_assign(state[i], s_final[i]))))

    # Log internal variables.
    for k, net in nets.items():
      

      print("Optimizer '{}' variables".format(k))
      print([op.name for op in snt.get_variables_in_module(net)])
    
    print(fx_final)
    return MetaLoss(loss, update, reset, fx_final, x_final)

  def meta_minimize(self, make_loss, len_unroll, learning_rate=0.01, **kwargs):
    """Returns an operator minimizing the meta-loss.

    Args:
      make_loss: Callable which returns the optimizee loss; note that this
          should create its ops in the default graph.
      len_unroll: Number of steps to unroll.
      learning_rate: Learning rate for the Adam optimizer.
      **kwargs: keyword arguments forwarded to meta_loss.

    Returns:
      namedtuple containing (step, update, reset, fx, x)
    """
    info = self.meta_loss(make_loss, len_unroll, **kwargs)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step = [optimizer.minimize(info.loss[i]) for i in range(self.num_lstm)]
    return MetaStep(step, *info[1:])
