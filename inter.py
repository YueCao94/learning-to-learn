def update(net, fx, x, state):
      """Parameter and RNN state update."""
      
      def convert_to_mat(list_x):
#        pdb.set_trace()
#        print(list_x)
#        haha=[tf.reshape(mat[0],[1,15680]) for mat in list_x]
#        hehe=tf.concat(haha[:self.num_lstm],axis=0)
#        print(hehe)
        Mat=[]
        for z in range(len(subsets)):
          mat_1=tf.concat([tf.reshape(mat[z][0],[1,15680]) for mat in list_x],axis=0)
        
          print('mat_1.shape',mat_1)
          mat_2=tf.concat([tf.reshape(mat[z][1],[1,20]) for mat in list_x],axis=0)
          print('mat_2.shape',mat_2)
          mat_3=tf.concat([tf.reshape(mat[z][2],[1,200]) for mat in list_x],axis=0)
          print('mat_3.shape',mat_3)
          mat_4=tf.concat([tf.reshape(mat[z][3],[1,10]) for mat in list_x],axis=0)
          print('mat_4.shape',mat_4)
          Mat.append([mat_1, mat_2 ,mat_3, mat_4])
        return Mat
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
            print(variable)
          Variable.append(variable)
        return Variable
      
      
      def inter_atten(mat_a,mat_b):
#        pdb.set_trace()
        l=1
        gama=1/self.num_lstm
        origin = mat_a
#        print(origin)
        next_x = mat_b 
#        print(next_x)
        grad = tf.subtract(next_x, origin)
#        print(grad)
        def attention(Mat_a,Mat_b):
#这里将inter——attention核心部分注释掉，相当于没有inter——attention
#          result = gama*tf.matmul(Mat_a,Mat_b)
#          Mat_b = tf.add(Mat_b,result)
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
        e_ij+=origin
#        print(e_ij)
        return e_ij
      
      with tf.name_scope("gradients"):
        Gradients=[]
        for i in range(self.num_lstm):
          gradients=[]
          for subset, key, s_i in zip(subsets, net_keys, state):
          #make sure x has same length as subset range
            x_i = [x[i][j] for j in subset]
            sub_gradients = tf.gradients(fx[i], x_i)
            

        # Stopping the gradient here corresponds to what was done in the
        # original L2L NIPS submission. However it looks like things like
        # BatchNorm, etc. don't support second-derivatives so we still need
        # this term.
            if not second_derivatives:
              sub_gradients = [tf.stop_gradient(g) for g in gradients]
            gradients.append(sub_gradients)
          Gradients.append(gradients)
      with tf.name_scope("inter_attention"):
        for i in range(self.num_lstm):
          for j in range(len(subsets)):


      with tf.name_scope("deltas"):
        deltas=[]
        state_next=[]
        for i in range(self.num_lstm):
          sub_deltas, sub_state_next = zip(*[net(g, s) for g, s in zip(gradients, state)])
          state_next = list(state_next)
          deltas.append(sub_deltas)
          state_next.append(sub_state_next)

      return deltas, state_next
#time_step的参数初始化返回的x_now代表当前mnist网络的x，x_next代表用lstm进行梯度更新后的mnist网络x   

#intra&inter time-step
    def time_step(t, fx_array, x, state):
      """While loop body."""
      x_next = [list(sub_x) for sub_x in x]
      state_next = []

      with tf.name_scope("fx"):
        for i in range(self.num_lstm):
          fx = _make_with_custom_variables(make_loss, x)
          fx_array = fx_array.write(t, fx)

      with tf.name_scope("dx"):
        for subset, key, s_i in zip(subsets, net_keys, state):
          #make sure x has same length as subset range
          x_i = [x[j] for j in subset]
          deltas, s_i_next = update(nets[key], fx, x_i, s_i)

          for idx, j in enumerate(subset):
            x_next[j] += deltas[idx]
          state_next.append(s_i_next)

      with tf.name_scope("t_next"):
        t_next = t + 1

      return t_next, fx_array, x_next, state_next


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

    def time_step(t, fx_array, x, state):
      """While loop body."""
      x_next = list(x)
      state_next = []

      with tf.name_scope("fx"):
        fx = _make_with_custom_variables(make_loss, x)
        fx_array = fx_array.write(t, fx)

      with tf.name_scope("dx"):
        for subset, key, s_i in zip(subsets, net_keys, state):
          x_i = [x[j] for j in subset]
          deltas, s_i_next = update(nets[key], fx, x_i, s_i)

          for idx, j in enumerate(subset):
            x_next[j] += deltas[idx]
          state_next.append(s_i_next)

      with tf.name_scope("t_next"):
        t_next = t + 1

      return t_next, fx_array, x_next, state_next




        