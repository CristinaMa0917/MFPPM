import tensorflow as tf
import numpy as np

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True) # 保证输入的尺度不变化，避免梯度爆炸或者弥散
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

C_RIDAR = 2 # <10
C_AGENT = 3 #< 14
C_ACTION = 1 #< 4
EMB_DIM_0 = 10
EMB_DIM_1 =  20
# conv1=ridar_dim - ridar_num-1 = 4
# conv2=agent_dim - agent_num-1 = 3
# conv3=action_dim - action_num -1 = 1

###1. 自注意力机制
def state_embbeding(inputs):
    '''
    :param inputs: [batch,step,state_dims]
    :param conv1: conv_size for state_ridar
    :param conv2: conv_size for state_agent
    :param emb_size: final emb_size
    :return: outputs:[batch,step,state_num,emb_size]
    '''
    state_ridar,state_agent,action_input = inputs
    with tf.name_scope('sa_emb') as s_emb:
        # 2 layer for ridar emb
        wR_conv0 = tf.get_variable(name='wR_conv0',shape=[1,1,1,EMB_DIM_0],initializer=initializer_relu(),regularizer=regularizer)
        bR_conv0 = tf.get_variable(name='bR_conv0',shape=[EMB_DIM_0],initializer=tf.zeros_initializer(),regularizer=regularizer)
        ridar_emb_0 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tf.expand_dims(state_ridar,axis=-1),filter=wR_conv0,strides=[1,1,1,1],padding='VALID'),bR_conv0)) #[batch,step,ridar_num,emb]

        conv1 = C_RIDAR
        wR_conv1 = tf.get_variable(name='wR_conv1',shape=[1,conv1,EMB_DIM_0,EMB_DIM_1],initializer=initializer_relu(),regularizer=regularizer)
        bR_conv1 = tf.get_variable(name='bR_conv1',shape=[EMB_DIM_1],initializer=tf.zeros_initializer(),regularizer=regularizer)
        ridar_emb = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ridar_emb_0,filter=wR_conv1,strides=[1,1,1,1],padding='VALID'),bR_conv1)) #[batch,step,ridar_num,emb]

        # 2 layer for agent emb
        wA_conv0 = tf.get_variable('wA_conv0',[1,1,1,EMB_DIM_0],initializer=initializer_relu(),regularizer=regularizer)
        bA_conv0 = tf.get_variable('bA_conv0',[EMB_DIM_0],initializer=tf.zeros_initializer(),regularizer=regularizer)
        agent_emb_0 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tf.expand_dims(state_agent,axis=-1),filter=wA_conv0,strides=[1,1,1,1],padding='VALID'),bA_conv0))

        conv1 = C_AGENT
        wA_conv1 = tf.get_variable(name='wA_conv1',shape=[1,conv1,EMB_DIM_0,EMB_DIM_1],initializer=initializer_relu(),regularizer=regularizer)
        bA_conv1 = tf.get_variable(name='bA_conv1',shape=[EMB_DIM_1],initializer=tf.zeros_initializer(),regularizer=regularizer)
        agent_emb = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(agent_emb_0,filter=wA_conv1,strides=[1,1,1,1],padding='VALID'),bA_conv1)) #[batch,step,ridar_num,emb]

        # 2 layer for action emb
        wB_conv0 = tf.get_variable('wB_conv0', [1, 1, 1, EMB_DIM_0],initializer=initializer_relu(),regularizer=regularizer)
        bB_conv0 = tf.get_variable('bB_conv0', [EMB_DIM_0],initializer=tf.zeros_initializer(),regularizer=regularizer)
        action_emb_0 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tf.expand_dims(action_input,axis=-1), filter=wB_conv0, strides=[1, 1, 1, 1], padding='VALID'),bB_conv0))

        conv1 = C_ACTION
        wB_conv1 = tf.get_variable(name='wB_conv1',shape=[1,conv1,EMB_DIM_0,EMB_DIM_1],initializer=initializer_relu(),regularizer=regularizer)
        bB_conv1 = tf.get_variable(name='bB_conv1',shape=[EMB_DIM_1],initializer=tf.zeros_initializer(),regularizer=regularizer)
        action_emb = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(action_emb_0,filter=wB_conv1,strides=[1,1,1,1],padding='VALID'),bB_conv1)) #[batch,step,ridar_num,emb]

    print('ridar_emb', ridar_emb.get_shape())
    print('agent_emb', agent_emb.get_shape())
    print('action_emb', action_emb.get_shape())

    input_emb = tf.concat([ridar_emb,agent_emb,action_emb],axis=2)
    print('input_emb', input_emb.get_shape())
    return input_emb

def norm_bn(x,episilon=1e-6):
    '''
    :param x: [batch,step,num,emb_dim]
    :param episilon:
    :return: norm_x [batch,step,num,emb_dim)
    '''
    mean = tf.reduce_mean(x,axis=[-1],keepdims=True)
    variance = tf.reduce_sum(tf.square(mean-x),axis=-1,keepdims=True)

    filter = mean.get_shape().as_list()
    with tf.variable_scope('normalization') as norm:
        scale = tf.get_variable('bn_scale',shape=filter,initializer=tf.ones_initializer(),regularizer=regularizer)
        bias = tf.get_variable('bn_bias',shape=filter,initializer=tf.zeros_initializer(),regularizer=regularizer)

    print('x shape',x.get_shape())
    print('mean shape',mean.get_shape())
    print('varience shape',tf.rsqrt(variance+episilon))

    norm_x = (x-mean)*tf.rsqrt(variance+episilon) #tf.rsqrt 平方根倒数
    return norm_x*scale+bias

def split_last_dimensions(x,n):
    old_shape = x.get_shape().as_list()
    last_dim = old_shape[-1]
    new_shape = old_shape[:-1]+[n]+[last_dim//n]
    ret = tf.reshape(x,new_shape)
    ret.set_shape(new_shape)
    return ret

def combine_last_two_dims(x):
    shape = x.get_shape().as_list()
    last_dim = shape[-1]*shape[-2]
    ret = tf.reshape(x,shape[:-2]+[last_dim])
    ret.set_shape(shape[:-2]+[last_dim])
    return ret

def dot_product_attention(q,k,v,bias,dropout=0.):
    logits = tf.matmul(q,k,transpose_b=True) #表示对后一个输入进行转置
    if bias:
        b = tf.get_variable('attention_bias',shape=logits.get_shape().as_list(),initializer=tf.zeros_initializer(),regularizer=regularizer)
        logits = logits+b
    weights = tf.nn.dropout(tf.nn.softmax(logits),keep_prob=1-dropout)
    return tf.matmul(weights,v)

def multihead_attention(input,head):
    '''
    :param input: [batch,step,num,emb_dim]
    :param head:
    :return: [batch,step,num,emb_dim]
    '''
    input = norm_bn(input)
    memory = input
    query = input
    in_channel = input.get_shape().as_list()[-1]
    with tf.name_scope('multihead_attention') as ma:
        conv_m = tf.get_variable('conv_m',[1,1,in_channel,2*in_channel],initializer=initializer_relu(),regularizer=regularizer)
        bias_m = tf.get_variable('bias_m',[2*in_channel],initializer=tf.zeros_initializer(),regularizer=regularizer)
        memory = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(memory,conv_m,strides=[1,1,1,1],padding='VALID'),bias_m))

        conv_q = tf.get_variable('conv_q',[1,1,in_channel,in_channel],initializer=initializer_relu(),regularizer=regularizer)
        bias_q = tf.get_variable('bias_q',[in_channel],initializer=tf.zeros_initializer(),regularizer=regularizer)
        query = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(query,conv_q,strides=[1,1,1,1],padding='VALID'),bias_q))

    Q = split_last_dimensions(query,head)
    K,V = [split_last_dimensions(tensor,head) for tensor in tf.split(memory,2,axis=-1)]

    ret = dot_product_attention(Q,K,V,bias=True,dropout=0.02)
    ret = combine_last_two_dims(ret) # batch,step,28,20
    return ret

if __name__=='__main__':
    batch = 10
    step = 6
    ridar_dim = 10
    agent_dim = 14
    action_dim = 4
    ridar_info = tf.constant(np.arange(batch*step*ridar_dim),shape=[batch,step,ridar_dim],dtype=tf.float32)
    agent_info = tf.constant(np.arange(batch*step*agent_dim),shape=[batch,step,agent_dim],dtype=tf.float32)
    action_info = tf.constant(np.arange(batch*step*action_dim),shape=[batch,step,action_dim],dtype=tf.float32)

    embeddings = state_embbeding([ridar_info,agent_info,action_info])
    ret = multihead_attention(embeddings,4)
    print(ret.get_shape())
