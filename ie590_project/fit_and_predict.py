import logging
import os

from tqdm import trange
import tensorflow as tf

from model.utils import save_dict_to_json

def _train_sess(sess, model_spec, num_steps, writer, params):
    """
    Args:
        sess: (tf.Session) current session
        model_spec: (dict)
        num_steps: (int) total number of batches per epoch
        writer: (tf.summary.FileWriter)
        params: (Params)
    """
    iterator_init_op = model_spec['iterator_init_op']
    metrics_init_op = model_spec['metrics_init_op']
    loss = model_spec['loss']
    reg_loss = model_spec['reg_loss'] #for debugging
    train_optim_op = model_spec['train_optim_op']
    update_metrics_op = model_spec['update_metrics_op']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.compat.v1.train.get_global_step()
    
    # initialize parameter & metric tensors
    sess.run([iterator_init_op, metrics_init_op])
    
    t = trange(num_steps)
    for i in t: 
        if i % params.save_summary_steps == 0:
            # this includes 'summary_op'
            _, _, loss_eval, reg_loss_eval, summary_eval, global_step_eval = sess.run([train_optim_op, update_metrics_op, 
                                                                        loss, reg_loss, summary_op, global_step])
            writer.add_summary(summary_eval, global_step_eval)
        else:
            _, _, loss_eval, reg_loss_eval = sess.run([train_optim_op, update_metrics_op, loss, reg_loss])
        
        # log the loss
        t.set_postfix(loss='{:05.3f}'.format(loss_eval), reg_loss='{:05.3f}'.format(reg_loss_eval))
        
    metrics_op = {k: v[0] for k, v in metrics.items()}
    metrics_eval = sess.run(metrics_op)
    metrics_log = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_eval.items())
    logging.info("-> Train metrics: " + metrics_log)


def _test_sess(sess, model_spec, num_steps, writer=None, params=None):
    """
    Args:
        sess: (tf.Session) current session
        model_spec: (dict)
        num_steps: (int) total number of batches per epoch
        writer: (tf.summary.FileWriter) None when doing predict() only
        params: (Params)
    """
    iterator_init_op = model_spec['iterator_init_op']
    metrics_init_op = model_spec['metrics_init_op']
    update_metrics_op = model_spec['update_metrics_op']
    metrics = model_spec['metrics']
    global_step = tf.compat.v1.train.get_global_step()
    
    # initialize parameter & metric tensors
    sess.run([iterator_init_op, metrics_init_op])
    
    for _ in range(num_steps):
        sess.run(update_metrics_op)
        
    metrics_op = {k: v[0] for k, v in metrics.items()}
    metrics_eval = sess.run(metrics_op)
    metrics_log = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_eval.items())
    logging.info("-> Test metrics: " + metrics_log)
    
    return metrics_eval


def fit(train_model_spec, test_model_spec, model_save_dir, params, config=None, restore_from=None): #TODO included:2
    """
    Args:
        train_model_spec / val_model_spec: (dict) tensorflow ops and nodes needed for training/validation
        model_dir: (str) directory containing config, weights, and log
        params: (Params)
        config: (tf.ConfigProto) configuration of the tf.Session, most likely about GPU options
        restore_from: (str) directory containing weights to restore in graph-reusing mode
    """
    last_saver = tf.train.Saver() # keep 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1) # keep 1 best epoch
    
    with tf.Session(config=config) as sess:
        sess.run(train_model_spec['variable_init_op'])
        
        if restore_from is not None: # TODO
            logging.info("Restoring parameters from {}".format(restore_from))
        else:
            begin_at_epoch = 0
            
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(model_save_dir, 'train_summaries'), sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(model_save_dir, 'test_summaries'), sess.graph)
        
        best_test_mse = 10000
        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, begin_at_epoch + params.num_epochs))
            
            #### for training ###############################
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size # number of batches per epoch
            _train_sess(sess, train_model_spec, num_steps, train_writer, params)
            
            last_saver_path = os.path.join(model_save_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_saver_path, global_step=epoch+1) # save as ".../last_weights/after-epoch-{epoch+1}"
            ###  (end)     ###################################
            
            #### for testing ################################
            num_steps = (params.test_size + params.batch_size - 1) // params.batch_size # number of batches per epoch
            metrics_eval = _test_sess(sess, test_model_spec, num_steps, test_writer)
            
            test_mse = metrics_eval['MSE']
            if test_mse < best_test_mse:
                best_test_mse = test_mse
                best_saver_path = os.path.join(model_save_dir, 'best_weights', 'after-epoch')
                best_saver.save(sess, best_saver_path, global_step=epoch+1) # save as ".../best_weights/after-epoch-{epoch+1}"
                logging.info("--> Found a new best MSE, saving in {}-{}".format(best_saver_path, epoch+1))
                
                best_json_path = os.path.join(model_save_dir, "metrics_eval_at_best_weights.json")
                save_dict_to_json(metrics_eval, best_json_path)
            ###  (end)     #################################
            
            last_json_path = os.path.join(model_save_dir, "metrics_eval_at_last_weights.json")
            save_dict_to_json(metrics_eval, last_json_path) #TODO

def predict(model_spec, model_save_dir, params, restore_from): #TODO: return or save a prediction as json
    """predict with restored model
    Args:
        inp: #TODO
        model_spec: (dict)
        model_dir: (str) the directory where the config, weights and log are stored
        params: (Params)
        restore_from: (str) the directory or ckpt where the weights are stored to restore the graph
    """
    assert (os.path.isdir(model_save_dir) and os.path.exists(restore_from)), "the saver dir/file does not exits at '{}/{}'".format(model_dir, restore_from)
    
    saver = tf.train.Saver()
    
    with tf.compat.v1.Session() as sess:
        sess.run(model_spec['variable_init_op'])
        
        # reload the weights
        save_path = os.path.join(model_save_dir, restore_from)
        
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path) # If restore_from is a directory, get the latest ckpt
        saver.restore(sess, save_path)
        
        num_steps = (params.val_size + params.batch_size - 1) // params.batch_size
        metrics = _test_sess(sess, model_spec, num_steps)
        metrics_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
        save_dict_to_json(metrics, save_path)

        