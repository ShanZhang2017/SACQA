import numpy as np
import tensorflow as tf
import config
import random
class Model:
    def __init__(self,vocab_size):
        embedding_lines=open('datasets/embedding.txt','r',encoding='utf-8').readlines()
        ns=0
        # matrix=[]
        print("embeddding initialize")
        for line in embedding_lines:
            # print(ns)
            line=line.strip('\n').split(' ')
            nums=[np.float32(x)for x in line]
            # matrix.append(nums)
        # matrix= np.array(matrix).T
            if ns==0:
                matrix = np.array(nums)
            else:
                matrix=np.c_[matrix,np.array(nums)]
            ns+=1
        matrix=matrix.T
        with tf.name_scope('read_inputs') as scope :
            self.Text_p=tf.placeholder(tf.int32,[config.batch_size,config.MAX_LEN],name='Tp')
            self.Text_q=tf.placeholder(tf.int32,[config.batch_size,config.MAX_LEN],name='Tq')
            self.Text_r=tf.placeholder(tf.int32,[config.batch_size,config.MAX_LEN],name='Tr')
            self.ys=tf.placeholder(tf.float32,[config.batch_size],name='ys')
            # self.rs=tf.placeholder(tf.float32,[config.batch_size,800],name='rs')
            # self.ws=tf.placeholder(tf.float32,[config.batch_size,800],name='ws')
        # with tf.name_scope('initialize_embedding') as scope :
        #     self.text_embed=tf.Variable(tf.truncated_normal([vocab_size,config.embed_size],stddev=0.3))


        with tf.name_scope('initialize_embedding') as scope :
            self.text_embed=tf.constant(matrix)
            # print (tf.shape(self.text_embed))
            # print(self.text_embed.shape)

        with tf.name_scope('looking_embeddings') as scope :
            self.TP=tf.nn.embedding_lookup(self.text_embed,self.Text_p) # [batch_size, MAX_LEN, self.embedding_size]
            self.T_P=tf.expand_dims(self.TP,-1) # [batch_size, MAX_LEN, self.embedding_size,1]

            self.TQ=tf.nn.embedding_lookup(self.text_embed,self.Text_q)
            self.T_Q=tf.expand_dims(self.TQ,-1)

            self.TR = tf.nn.embedding_lookup(self.text_embed, self.Text_r)
            self.T_R = tf.expand_dims(self.TR, -1)

            self.y=tf.expand_dims(self.ys, -1)

        self.p, self.q, self.r1, self.r2  = self.conv()
        self.loss,self.prediction,self.collection =self.compute_loss_acc()

    def conv(self):
        regularizer = tf.contrib.layers.l2_regularizer(scale=config.l2_reg)

        W1 = tf.get_variable('weight', shape=[2,config.embed_size,1,config.embed_size], initializer=tf.truncated_normal_initializer(stddev=0.3),regularizer=regularizer)
        W2 = tf.get_variable('weight2', shape=[2, config.embed_size, 1, config.embed_size],
                             initializer=tf.truncated_normal_initializer(stddev=0.3), regularizer=regularizer)
        W3 = tf.get_variable('weight3', shape=[2, config.embed_size, 1, config.embed_size],
                             initializer=tf.truncated_normal_initializer(stddev=0.3), regularizer=regularizer)
        rand_matrix1 = tf.get_variable('attention1', shape=[config.embed_size,config.embed_size],
                             initializer=tf.truncated_normal_initializer(stddev=0.3), regularizer=regularizer)
        rand_matrix2 = tf.get_variable('attention2', shape=[config.embed_size, config.embed_size],
                                      initializer=tf.truncated_normal_initializer(stddev=0.3), regularizer=regularizer)
        #
        # W1=tf.Variable(tf.truncated_normal([2,config.embed_size,1,config.embed_size],stddev=0.3))
        # W2 = tf.Variable(tf.truncated_normal([2, config.embed_size, 1, config.embed_size], stddev=0.3))
        # W3 = tf.Variable(tf.truncated_normal([2, config.embed_size, 1, config.embed_size], stddev=0.3))
        # rand_matrix=tf.Variable(tf.truncated_normal([config.embed_size,config.embed_size],stddev=0.3))

        convP=tf.nn.conv2d(self.T_P,W1,strides=[1,1,1,1],padding='VALID')   # [batch_size, MAX_LEN-1, 1, embed_size]
        convQ=tf.nn.conv2d(self.T_Q,W2,strides=[1,1,1,1],padding='VALID')
        convR=tf.nn.conv2d(self.T_R,W3,strides=[1,1,1,1],padding='VALID')

        hP=tf.tanh(tf.squeeze(convP)) #[batch_size,MAX_LEN-1,embed_size]
        hQ=tf.tanh(tf.squeeze(convQ))
        hR=tf.tanh(tf.squeeze(convR))

        tmphP=tf.reshape(hP,[config.batch_size*(config.MAX_LEN-1),config.embed_size]) # [batch_size * MAX_LEN-1, embed_size]
        tmphQ=tf.reshape(hQ,[config.batch_size*(config.MAX_LEN-1),config.embed_size])
        hp_mul_rand=tf.reshape(tf.matmul(tmphP,rand_matrix1),[config.batch_size,config.MAX_LEN-1,config.embed_size])
        hq_mul_rand=tf.reshape(tf.matmul(tmphQ,rand_matrix2),[config.batch_size,config.MAX_LEN-1,config.embed_size])

        r1=tf.matmul(hp_mul_rand,hQ,adjoint_b=True) #[batch_size,MAX_LEN-1,MAX_LEN-1]
        att1=tf.expand_dims(r1,-1) #[batch_size,MAX_LEN-1,MAX_LEN-1,1]
        att1=tf.tanh(att1)

        pooled_p=tf.reduce_mean(att1,2) #[batch_size,MAX_LEN-1,1]
        pooled_q=tf.reduce_mean(att1,1)

        p_flat=tf.squeeze(pooled_p)
        q_flat=tf.squeeze(pooled_q)

        a_p=tf.nn.softmax(p_flat) #[batch_size,MAX_LEN-1]
        a_p=tf.expand_dims(a_p,-1) #[batch_size,MAX_LEN-1,1]
        a_q=tf.nn.softmax(q_flat)
        a_q=tf.expand_dims(a_q,-1)
# P和正确答案
        r2=tf.matmul(hp_mul_rand,hR,adjoint_b=True) #[batch_size,MAX_LEN-1,MAX_LEN-1]
        r2=tf.multiply(r2,a_p) #modify interaction matrix
        att2=tf.expand_dims(r2,-1)
        att2=tf.tanh(att2) #[batch_size,MAX_LEN-1,MAX_LEN-1,1]

        pooled_rp=tf.reduce_mean(att2,2) #[batch_size,MAX_LEN-1,1]
        pooled_r1=tf.reduce_mean(att2,1)
        rp_flat=tf.squeeze(pooled_rp)#[batch_size,MAX_LEN-1]
        r1_flat=tf.squeeze(pooled_r1)

        w_rp=tf.nn.softmax(rp_flat)#[batch_size,MAX_LEN-1]
        w_rp=tf.expand_dims(w_rp,-1)#[batch_size,MAX_LEN-1,1]
        w_r1=tf.nn.softmax(r1_flat)
        w_r1=tf.expand_dims(w_r1,-1)

        h_P=tf.transpose(hP,perm=[0,2,1]) #[batch_size,embed_size,MAX_LEN-1]
        h_R=tf.transpose(hR,perm=[0,2,1])

        rep_p=tf.matmul(h_P,w_rp)#[batch_size,embed_size,1]
        rep_p=tf.squeeze(rep_p)#[batch_size,embed_size]
        rep_r1=tf.matmul(h_R,w_r1)
        rep_r1=tf.squeeze(rep_r1)
# Q和正确答案
        r3=tf.matmul(hq_mul_rand,hR,adjoint_b=True)
        r3=tf.multiply(r3,a_q)
        att3 = tf.expand_dims(r3, -1)
        att3 = tf.tanh(att3)  # [batch_size,MAX_LEN-1,MAX_LEN-1,1]

        pooled_rq = tf.reduce_mean(att3, 2)
        pooled_r2 = tf.reduce_mean(att3, 1)
        rq_flat = tf.squeeze(pooled_rq)
        r2_flat = tf.squeeze(pooled_r2)

        w_rq = tf.nn.softmax(rq_flat)
        w_rq = tf.expand_dims(w_rq, -1)
        w_r2 = tf.nn.softmax(r2_flat)
        w_r2 = tf.expand_dims(w_r2, -1)

        h_Q = tf.transpose(hQ, perm=[0, 2, 1])

        rep_q = tf.matmul(h_Q, w_rq)
        rep_q = tf.squeeze(rep_q)
        rep_r2 = tf.matmul(h_R, w_r2)
        rep_r2 = tf.squeeze(rep_r2)

        return rep_p,rep_q,rep_r1,rep_r2

    def multilayer_perception(self,inputs_1,in_size, out_size, activation_function=None,num_classes=1):
        # weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0, stddev=0.3))
        # biases = tf.Variable(tf.zeros([1, out_size]))
        # out1 = tf.matmul(inputs_1, weights) + biases
        estimation=tf.contrib.layers.fully_connected(
            inputs=inputs_1,
            num_outputs=num_classes,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=config.l2_reg),
            biases_initializer=tf.constant_initializer(1e-04),
            scope="FC"
        )

        # if activation_function is None:
        #     output1 = out1
        #
        # else:
        #     output1 = activation_function(out1)

        return estimation


    def compute_loss_acc(self):
        co_zero = tf.zeros([config.batch_size, 1])
        co_one = tf.fill([config.batch_size, 1], 0.5)


        rs = tf.concat([self.p, self.q, self.r1, self.r2],1)#[batch_size,4 * embed_size]
        # rs=np.random.normal(loc=0.0, scale=1.0, size=(config.batch_size, 800)).astype(np.float32)
        # ws = np.random.normal(loc=0.0, scale=1.0, size=(config.batch_size, 800)).astype(np.float32)

        # l_r1 = self.multilayer_perception(rs, 800, 256, activation_function=tf.nn.sigmoid)  # 隐藏层
        # l_r2 = self.multilayer_perception(l_r1, 256, 256, activation_function=tf.nn.sigmoid)
        # prediction_r = self.multilayer_perception(l_r2, 256, 1, activation_function=tf.nn.sigmoid)
        #
        # l_w1 = self.multilayer_perception(ws, 800, 256, activation_function=tf.nn.sigmoid)  # 隐藏层
        # l_w2 =self.multilayer_perception(l_w1, 256, 256, activation_function=tf.nn.sigmoid)
        # prediction_w  = self.multilayer_perception(l_w2, 256, 1, activation_function=tf.nn.sigmoid)

        # prediction_r,prediction_w = self.multilayer_perception(rs, ws,800, 1, activation_function=tf.nn.tanh)



        # l_r1 = self.multilayer_perception(rs,800, 256, activation_function=tf.nn.sigmoid)  # 隐藏层
        # l_r2  = self.multilayer_perception(l_r1, 256, 256, activation_function=tf.nn.sigmoid)
        # prediction_r = self.multilayer_perception(l_r2, 256, 1, activation_function=None)
        prediction_r = self.multilayer_perception(rs, 256, 1, activation_function=None,num_classes=1)
        # prediction=tf.contrib.layers.softmax(prediction_r)[:, 1:2]
        prediction=tf.nn.sigmoid(prediction_r)
        print(np.shape(prediction))
        #
        # cross_entropy =tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=prediction_r)
        # cross_entropy= tf.squeeze(cross_entropy)
        # # cost_t=tf.reduce_mean(cross_entropy)
        cost = tf.add(
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=prediction_r)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="cost")
        co=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)



        # prediction_gaps=prediction_w-prediction_r+co_one
        # prediction_gap=tf.concat([prediction_gaps,co_zero],1)
        # prediction_gap=tf.reduce_max(prediction_gap,1)
        # cost = tf.reduce_mean(prediction_gap)

        return  cost,prediction,co
      #
        #
        # # 计算最后一层是softmax层的cross entropy
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
        # # 使用AdamOptimizer进行梯度下降
        # train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
















