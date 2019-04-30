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


MetaLoss = collections.namedtuple("MetaLoss", "loss, update, reset, fx, x, test")
MetaStep = collections.namedtuple("MetaStep", "step, update, reset, fx, x, test, grad")


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
#    pdb.set_trace()
#    trainable = tf.constant(False, dtype = tf.bool)
    #intra attention x_minimal verification 
    self.fx_minimal = [tf.Variable(float("inf"),trainable = False) for i in range(self.num_lstm)]
    #intra attention x_minimal if fx_minimal is satisfied
    
    delta_shape=([784,20],[20,],[20,10],[10,])
    tensors=[]
    for i in range(4):
        tensor = tf.Variable(tf.zeros(delta_shape[i]), trainable = False)
        tensors.append(tensor)
    self.x_minimal = [tensors for i in range(self.num_lstm)]
    self.pre_deltas = [tensors for i in range(self.num_lstm)]
    self.pre_gradients = [tensors for i in range(self.num_lstm)]
    self.intra_features = 3
    self.fc_kernel = []
    self.fc_bias = []
    self.fc_va = []
    fc_kernel_shape = ([20, 15680*2], [20, 20*2], [20, 200*2], [10, 10*2])
    fc_bias_shape = ([20, self.intra_features ], [20, self.intra_features], [20, self.intra_features], [10, self.intra_features])
    fc_va_shape=([1,20],[1,20],[1,20],[1,10])
    for i in range(4):
      sub_fc_kernel = tf.Variable(tf.random_normal(fc_kernel_shape[i]))
      sub_fc_bias = tf.Variable(tf.random_normal(fc_bias_shape[i]))
      sub_fc_va = tf.Variable(tf.ones(fc_va_shape[i]), trainable = False)
      self.fc_kernel.append(sub_fc_kernel)
      self.fc_bias.append(sub_fc_bias)
      self.fc_va.append(sub_fc_va)
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
#    pdb.set_trace()
#    print(sub_x)
    x=[sub_x for i in range(self.num_lstm)]
    constants=[sub_constants for i in range(self.num_lstm)]
    print("x.length",len(x))
    print("Optimizee variables")
    print([op.name for op in x[0]])
    print("Problem variables")
    print([op.name for op in sub_constants])
#    x_loop = x
    # Create the optimizer networks and find the subsets of variables to assign
    # to each optimizer.
    nets, net_keys, subsets = _make_nets(sub_x, self._config, net_assignments) 

    # Store the networks so we can save them later.
    self._nets = nets

    # Create hidden state for each subset of variables.
    '''
    if len(subsets) > 1:
      state = []
    else:
      state=[[] for i in range(len(subsets))]
    '''
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
      '''
      if len(subsets) > 1:
        state.append(single_state) 
      else:
        for i in range(len(subsets))
          state[i].append(single_state[i]) 
      '''
#    pdb.set_trace()
    
    def update(t, net, fx, x, state):
      """Parameter and RNN state update."""
      
      def convert_to_mat(list_x):
#        pdb.set_trace()
#        print(list_x)
#        haha=[tf.reshape(mat[0],[1,15680]) for mat in list_x]
#        hehe=tf.concat(haha[:self.num_lstm],axis=0)
#        print(hehe)
        mat_1=tf.concat([tf.reshape(mat[0],[1,15680]) for mat in list_x],axis=0)
        
#        print('mat_1.shape',mat_1)
        mat_2=tf.concat([tf.reshape(mat[1],[1,20]) for mat in list_x],axis=0)
#        print('mat_2.shape',mat_2)
        mat_3=tf.concat([tf.reshape(mat[2],[1,200]) for mat in list_x],axis=0)
#        print('mat_3.shape',mat_3)
        mat_4=tf.concat([tf.reshape(mat[3],[1,10]) for mat in list_x],axis=0)
#        print('mat_4.shape',mat_4)
        
        return [mat_1, mat_2 ,mat_3, mat_4]
#将转化后的矩阵再转变成同样结构mnist网络的变量，进行梯度更新     
      def convert_to_Variable(list_x):
#此处的mnist网络结构由mlp构成，有四组参数，第一层的wights，第一层的bias，第二层的wights，第二层的bias，shape如下shape#所示
#        pdb.set_trace()
        shape=([784,20],[-1],[20,10],[-1])
#Variable的长度是lstm的个数，每个元素都包含对应mnist网络的变量参数
        Variable=[]
        for j in range(self.num_lstm):
          variable=[]
          for i in range(4):
            mat_x=list_x[i]
            variable.append(tf.reshape(mat_x[j],shape[i]))
#            print(variable)
          Variable.append(variable)
        return Variable
      
      
      def inter_atten(mat_a,mat_b):
#        pdb.set_trace()
        l=1
        gama=1/self.num_lstm
        origin = mat_a
#        print(origin)
        grad = mat_b 
#        print(grad)
        def attention(Mat_a,Mat_b):
#这里将inter——attention核心部分注释掉，相当于没有inter——attention
          result = gama*tf.matmul(Mat_a,Mat_b)
          Mat_b = tf.add(Mat_b,result)
          return Mat_b
        matmul1 = tf.matmul(grad,tf.transpose(grad))
#        print(matmul1)
        matmul2 = tf.matmul(origin,tf.transpose(origin))
        matmul2 = tf.exp(matmul2/2*l)
#        print(matmul2)
        softmax_grad = tf.nn.softmax(tf.transpose(matmul1))
#        print(softmax_grad)
#        softmax_grad = tf.transpose(softmax_grad)
        softmax_origin = tf.nn.softmax(tf.transpose(matmul2))
#        print(softmax_origin)
#        softmax_origin = tf.transpose(softmax_origin)
#        input_mul_x = tf.multiply(softmax_grad,softmax_origin)
#        e_ij = attention(input_mul_x,grad)
        e_ij = attention(softmax_grad,grad)
#        print(e_ij)
        return e_ij
      
      def intra_attention( grads,pre_grads, x, x_min, ht):
#        pdb.set_trace()
#        print(x)
        def convert2vector(var, shape):
          return tf.reshape(var, shape)
      
        shape=([1,15680],[1,20],[1,200],[1,10])
        reshape=([784,20],[-1],[20,10],[-1])
        Gradient = []
        beta = 0.9
        for i in range(4):
            
          sub_grad = convert2vector(grads[i], shape[i])
          sub_moment = beta*convert2vector(pre_grads[i], shape[i])
          sub_x = convert2vector(x[i], shape[i])
          sub_x_min = convert2vector(x_min[i], shape[i])
          sub_ht = convert2vector(ht[i], shape[i])
          x_res = sub_x-sub_x_min
          intra_feature=tf.concat([sub_grad,sub_moment,x_res],axis=0)
          grad_concat = tf.concat([sub_grad,sub_ht],axis=1)            
          moment_concat = tf.concat([sub_moment,sub_ht],axis=1)
          x_res_concat = tf.concat([x_res,sub_ht],axis=1)
          intra_concat = tf.concat([grad_concat,moment_concat,x_res_concat],axis=0)
          intra_concat = tf.transpose(intra_concat)
          intra_fc=tf.tanh(tf.matmul(self.fc_kernel[i],intra_concat) + self.fc_bias[i])
          va = self.fc_va[i]
          b_ij = tf.matmul(va,intra_fc)
          p_ij = tf.nn.softmax(b_ij)
          gradient = tf.matmul(p_ij, intra_feature)
          gradient = tf.reshape(gradient, reshape[i])
          Gradient.append(gradient)
        return Gradient

      with tf.name_scope("gradients"):
#        pdb.set_trace()
        gradients=[]
        for i in range(self.num_lstm):
          print(fx[i])
          print(x[i])
          sub_gradients = tf.gradients(fx[i], x[i])
          print(sub_gradients)
            
        # Stopping the gradient here corresponds to what was done in the
        # original L2L NIPS submission. However it looks like things like
        # BatchNorm, etc. don't support second-derivatives so we still need
        # this term.
          if not second_derivatives:
            sub_gradients = [tf.stop_gradient(g) for g in sub_gradients]
          with tf.name_scope("intra_attention"):
            x_min = self.x_minimal[i]
            ht = self.pre_deltas[i]
            pre_grads = self.pre_gradients[i]
            print(sub_gradients)
            print(x[i])
            print(x_min)
            print(ht)
            sub_gradients = intra_attention(sub_gradients, pre_grads, x[i], x_min, ht)
            
          gradients.append(sub_gradients)
        self.pre_gradients=gradients
#      pdb.set_trace()
#      print(gradients)
      with tf.name_scope("inter_attention"):
##x to matrix
        mat_x=convert_to_mat(x)
        mat_grads=convert_to_mat(gradients)
##inter-attention
        inter_mat=[inter_atten(mat_x[i],mat_grads[i]) for i in range(4)]
#        print(inter_mat)
##matrix to x
        gradients=convert_to_Variable(inter_mat)
#        print('mnist_gradients',gradients)


      with tf.name_scope("deltas"):
        deltas=[]
        state_next=[]
#        pdb.set_trace()
        for i in range(self.num_lstm):
          sub_deltas, sub_state_next = zip(*[net(g, s) for g, s in zip(gradients[i], state[i])])
          
          self.pre_deltas[i]=sub_deltas
          sub_state_next = list(sub_state_next)
          deltas.append(sub_deltas)
          state_next.append(sub_state_next)
          #注意此处的delats和gradients未考虑subsets的影响
      
#      print(state_next)
      return deltas, state_next
#time_step的参数初始化返回的x_now代表当前mnist网络的x，x_next代表用lstm进行梯度更新后的mnist网络x   
#intra&inter time-step
#    pdb.set_trace()
#    print(x)
    def time_step(t, fx_array, x, state):
      """While loop body."""

#      pdb.set_trace()
#      print(x)
      def X_next(x):
        X_next = []
        for i in range(self.num_lstm):
            x_list=list(x[i])
            X_next.append(x_list)
        return X_next
      x_next = X_next(x)
      
      print(x_next)
      state_next = [[] for z in range(self.num_lstm)]
      Fx_array = []
      with tf.name_scope("fx"):
        update_fx=[]
        for i in range(self.num_lstm):
          fx = _make_with_custom_variables(make_loss, x[i])
          def f1(): return self.fx_minimal[i], self.x_minimal[i]
          def f2(): return fx, x[i]
          self.fx_minimal[i], self.x_minimal[i] = tf.cond(tf.greater(fx, self.fx_minimal[i]), lambda:f1(), lambda:f2()) 
          sub_fx_array = fx_array[i].write(t, fx)
          Fx_array.append(sub_fx_array)
          update_fx.append(fx)

      with tf.name_scope("dx"):
        update_state = []
        update_x = []
        for i in range(self.num_lstm):
          sub_state = state[i][0]
          print(subsets[0])
          sub_x = [x[i][z] for z in subsets[0]]
          update_state.append(sub_state)
          update_x.append(sub_x)
#          print('update_x',update_x)
#          print('update_state',update_state)
          print(update_x)
        deltas, s_i_next = update(t, nets[key], update_fx, update_x, update_state)
        for m in range(self.num_lstm):
          for idx, n in enumerate(subsets[0]):
            x_next[m][n] += deltas[m][idx]
          state_next[m].append(s_i_next[m])
#      pdb.set_trace()
#      print(x)
#      print(state_next)
      with tf.name_scope("t_next"):
          t_next = t + 1
      
        
      return t_next, Fx_array, x_next, state_next

    
    # Define the while loop.
    fx_array = [tf.TensorArray(tf.float32, size=len_unroll + 1,
                              clear_after_read=False) for i in range(self.num_lstm)]

#    t_test, fx_test, x_test, s_test=time_step(0, fx_array, x, state)
#    t_test, fx_test, x_test, s_test=time_step(t_test, fx_test, x_test, s_test)
#    t_test, fx_test, x_test, s_test=time_step(t_test, fx_test, x_test, s_test)
#    fx_test = [_make_with_custom_variables(make_loss, sub_x_final) for sub_x_final in x_test]
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
#      print('x_final',x_final)
      fx_final = [_make_with_custom_variables(make_loss, sub_x_final) for sub_x_final in x_final]
#      print('fx_final',fx_final)
#      print(len(fx_final))
#     print('fx_array[1]',fx_array[1])
      for i in range(self.num_lstm):
        fx_array[i].write(len_unroll, fx_final[i])
#    pdb.set_trace()
#    print(fx_array)
#    pdb.set_trace()
    loss = sum([tf.reduce_sum(sub_fx_array.stack(), name="loss") for sub_fx_array in fx_array])

    # Reset the state; should be called at the beginning of an epoch.
    
    with tf.name_scope("reset"):
#      pdb.set_trace()
      print(x) 
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
#这里最后一个x的位置是为了传值到train中的sess里面验证，没有实际意义
    return MetaLoss(loss, update, reset, fx_final, x_final, x)

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
    regular = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
    gradients = optimizer.compute_gradients(info.loss+regular)
#    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
#    train_op = optimizer.apply_gradients(capped_gradients)
#    train_op = optimizer.apply_gradients(gradients)
    step = optimizer.minimize(info.loss)
    return MetaStep(step, *info[1:],gradients)
