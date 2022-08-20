import tensorflow as tf

def custom_loss_function_bb(y_true, y_pred):
    gt_box = y_true[:,0:4]
    pred_box = y_pred[:,0:4] 
    mse = tf.losses.mean_squared_error(y_true=gt_box,y_pred=pred_box)
    return mse

def custom_loss_function_softmax(y_true, y_pred):
    gt_class = y_true[:,4:]  
    pred_class = y_pred[:,:]
    cce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=gt_class, logits=pred_class)
    return cce_loss