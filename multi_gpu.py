
import sys,os,time
import tensorflow as tf
import numpy as np

def model_fn(features,labels,mode,params):

    net = tf.layers.dense(inputs=features, units=10,activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=net, units=1)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['gt_class'], logits=logits))

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("Creating oprimizer ...")
        global_step = tf.train.get_or_create_global_step()
        learning_rate = 0.1
        optimizer = tf.train.AdamOptimizer(learning_rate,epsilon=1e-8)
        grad_vars = optimizer.compute_gradients(loss)
        minimize_op = optimizer.apply_gradients(grad_vars, global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        ''' No train op at non-train modes '''
        train_op = None

    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=tf.nn.sigmoid(logits),
      loss=loss,
      train_op=train_op)
    
def input_fn(**kwargs):
    '''
    '''
    def _preprocessor(z):
        return (np.random.uniform(size=(100,),low=0,high=1).astype(np.float32),
                    np.random.randint(0,2,1).astype(np.float32))
    dataset = tf.data.Dataset.from_tensor_slices(tf.random.uniform((kwargs['batch_size']*kwargs['n_iters'],)))
    dataset = dataset.map(lambda x :tf.py_func(_preprocessor, [x],[tf.float32,tf.float32], stateful=False),num_parallel_calls=1)
    dataset = dataset.map(lambda *parts :(tf.reshape(parts[0],(100,)),{'gt_class':tf.reshape(parts[1],(1,))}),num_parallel_calls=1)
    dataset = dataset.batch(kwargs['batch_size'])
    return dataset

def input_fn_train(batch_size=10,n_iters=1000,**kwargs):
    ''' The input_fn takes the per replica, that is per GPU, batch size.  '''
    return input_fn(mode= tf.estimator.ModeKeys.TRAIN,
                    batch_size = batch_size,
                    n_iters=n_iters)

def input_fn_eval(batch_size=10,n_iters=100,**kwargs):
    return input_fn(mode= tf.estimator.ModeKeys.EVAL,
                    batch_size = batch_size,
                    n_iters=n_iters)

def run(outdir,n_iters):
    
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth = True))


    ''' With devices=None, MirroredStrategy will use all GPUs made availble to the process '''
    train_distribution_strategy = tf.distribute.MirroredStrategy(devices=None)
    eval_distribution_strategy = tf.distribute.MirroredStrategy(devices=None)

    run_config = tf.estimator.RunConfig(
        model_dir=outdir,
        tf_random_seed=1000,
        train_distribute=train_distribution_strategy,
        eval_distribute=eval_distribution_strategy,
        session_config=session_config,
        save_summary_steps=1,
        log_step_count_steps=10,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=100)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    
    ''' 
        The input function must be supported with the per replica/GPU batch size.
        At every update op, MirroredStrategy will run the datset object once for each replica of the model.
        For example, lets say you have one GPU and set your code to train for 10000 iterations with batch size 10,
        then if you provide 2 GPUs, the same code will run for 5000 iterations with batch size 20.
    '''
    train_spec = tf.estimator.TrainSpec(input_fn=lambda input_context=None: input_fn_train(n_iters=n_iters,max_steps=n_iters,input_context=input_context))
    eval_spec = tf.estimator.EvalSpec(input_fn= lambda: input_fn_eval(n_iters=100))
    tf.logging.info('Starting to train and evaluate ...')
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    tf.logging.info('Done!')

def get_available_gpus():
    from tensorflow.python.client import device_lib
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU' ]
    if len(gpus) == 0: return ["/cpu:0"]
    tf.logging.info( 'Availble GPUs: {}'.format(', '.join(gpus)) ) 
    return gpus

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    outdir = sys.argv[1]
    n_iters = int(sys.argv[2])
    outdir= os.path.join( '{}/{}_{}_gpus'.format(outdir, int(time.time()),len( get_available_gpus()) ) )
    os.mkdir(outdir)
    run(outdir,n_iters)    