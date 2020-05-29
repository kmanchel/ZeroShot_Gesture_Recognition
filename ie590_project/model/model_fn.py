"""include functions to build model networks"""

import tensorflow as tf

######################################################################################
#        Modules        ##############################################################
######################################################################################
def _conv2d(inp, filters, kernel_size, strides=1, activation=tf.nn.relu, reg_scale=0.001):
    """2D convolutional layer applying same-padding, he initializer, and regularizer """
    regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
    
    out = tf.compat.v1.layers.conv2d(inp, filters=filters, kernel_size=kernel_size, strides=strides,
                          padding='same', activation=activation, 
                          kernel_initializer=tf.compat.v1.initializers.he_normal(),
                          kernel_regularizer=regularizer, bias_regularizer=regularizer
                          )
    
    return out

def _dense(inp, nout, activation=tf.nn.relu, reg_scale=0.001):
    """Fully connected layer applying he initializer and regularizer """
    regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
    
    out = tf.compat.v1.layers.dense(inp, units=nout, activation=activation,
                         kernel_initializer=tf.compat.v1.initializers.he_normal(),
                         kernel_regularizer=regularizer, bias_regularizer=regularizer
                         )
    
    return out
   
def _maxpool2d(inp, ksize, strides=None):
    """Max pooling layer applying valid-padding and square 2D window """
    if strides is None:
        strides = ksize
    
    out = tf.nn.max_pool2d(inp, ksize=[1,ksize,ksize,1], strides=[1,strides,strides,1], padding='VALID')
    
    return out

def _embedding(inp, nout):
    pass

    return out

def _resnet_module(inp, filters, kernel_size, is_training, downsample, params, activation=tf.nn.relu, reg_scale=0.001):
    """resnet module: 2 conv filters + skip connection
    Args:
        downsample: (bool) whether down-sample or not
        filters: (list) a list of 3 filter sizes in order
    """
    f1, f2, f3 = filters
    
    if downsample: #convolutional mapping
        out = _conv2d(inp, filters=f1, kernel_size=(1,1), strides=2, activation=None, reg_scale=params.reg_scale)
        out = tf.compat.v1.layers.batch_normalization(out, training=is_training)
        out = tf.nn.relu(out)
        identity = _conv2d(inp, filters=f3, kernel_size=(1,1), use_bias=False)
        identity = tf.compat.v1.layers.batch_normalization(identity, training=is_training)
    else: #identity mapping
        out = _conv2d(inp, filters=f1, kernel_size=(1,1), activation=activation, reg_scale=params.reg_scale)
        out = tf.compat.v1.layers.batch_normalization(out, training=is_training)
        out = tf.nn.relu(out)
        identity = inp
    
    out = _conv2d(out, filters=f2, kernel_size=kernel_size, activation=None, reg_scale=params.reg_scale)
    out = tf.compat.v1.layers.batch_normalization(out, training=is_training)
    out = _conv2d(out, filters=f2, kernel_size=(1,1), activation=None, reg_scale=params.reg_scale)
    out = tf.compat.v1.layers.batch_normalization(out, training=is_training)
    out = tf.add(out, identity)
    out = tf.nn.relu(out)
    
    return out

######################################################################################
#        (end)       #################################################################
######################################################################################


######################################################################################
#        Networks        #############################################################
######################################################################################
def _vgg19(is_training, inputs, params): #TODO included:1
    """the model architecture
    """
    
    images = inputs['images']
    
    assert images.get_shape().as_list() == [None, params.image_height, params.image_width, params.num_ch] #TODO: errors here
    net = images
                     
    fsizes = [16, 32, 64, 128, 256]
    for i, fsize in enumerate(fsizes):
        with tf.compat.v1.variable_scope('block_{}'.format(i+1)):
            net = _conv2d(net, fsize, 3, reg_scale=params.reg_scale)
            net = _conv2d(net, fsize, 3, reg_scale=params.reg_scale)
            if i > 1:
                net = _conv2d(net, fsize, 3, reg_scale=params.reg_scale)
            net = _maxpool2d(net, 2)
    
    net = tf.compat.v1.layers.Flatten()(net)
    
    nodes = [1024, 1024]
    for i, n in enumerate(nodes):
        with tf.compat.v1.variable_scope('fc_{}'.format(i+1)):
            net = _dense(net, n, reg_scale=params.reg_scale)
    
    with tf.compat.v1.variable_scope('output'):
        net = _dense(net, 1, activation=None, reg_scale=params.reg_scale)
        
    return net

def _resnet50(is_training, inputs, params):
    images = inputs['images']
    
    assert image.get_shape().as_list() == [None, params.image_height, params.image_width, params.num_ch]
    net = images
    
    net = _conv2d(net, 64, (7,7), strides=2, activation=None, reg_scale=params.reg_scale)
    net = tf.compat.v1.layers.batch_normalization(net, is_training=is_training)
    net = tf.nn.relu(net)
    
    return net

def _phase1(inputs, params):
    pass

def _phase2(inputs, params):
    pass
######################################################################################
#        (end)       #################################################################
######################################################################################

def model_fn(is_training, inputs, params, reuse=tf.compat.v1.AUTO_REUSE): #TODO included:2
    """
    Args:
        inputs: (dict) input images, targets, and dataset initializer. output of input_fn() in input_fn.py
        params: (Params)
    Return:
        model_spec: (dict) tensorflow ops for init/training and evaluation
    """
    
    targets = inputs['targets']
    ghi_mean, ghi_sd = params.ghi_stats
    
    with tf.compat.v1.variable_scope('model', reuse=reuse):
        out = _small_vgg(is_training, inputs, params) #apply a network architecture
        predictions = (out * ghi_sd) + ghi_mean #reverse-standardize to get original scale prediction
    
    mse = tf.compat.v1.losses.mean_squared_error(targets, out) #MSE for loss
    reg_loss = tf.reduce_mean(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)) #regularizer loss
    loss = mse + reg_loss #total loss
    
    if is_training:
        optimizer = tf.compat.v1.train.AdamOptimizer(params.learning_rate)
        global_step = tf.compat.v1.train.get_or_create_global_step() # TODO
        
        train_optim_op = optimizer.minimize(loss, global_step=global_step)
    
    # Metrics for evaluation using tf.metrics (average over whole dataset in case SGD)
    with tf.compat.v1.variable_scope("metrics"):
        metrics = {
            'MSE': tf.compat.v1.metrics.mean(mse), 
            'reg_loss': tf.compat.v1.metrics.mean(reg_loss), # for debugging
            'loss': tf.compat.v1.metrics.mean(loss),
        }
        
    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.compat.v1.variables_initializer(metric_variables)
        
    # summary
    tf.compat.v1.summary.scalar('loss', loss)
    
    model_spec = inputs
    model_spec['variable_init_op'] = tf.compat.v1.global_variables_initializer()
    model_spec['predictions'] = predictions
    model_spec['loss'] = loss
    model_spec['reg_loss'] = reg_loss #for debugging
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics_op'] = update_metrics_op
    model_spec['summary_op'] = tf.compat.v1.summary.merge_all()
    if is_training:
        model_spec['train_optim_op'] = train_optim_op
        
    return model_spec
        
        