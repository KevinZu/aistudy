import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# define xavier initialization function, mean distribution or gauss distribution
# mean = 0, var = 2/(n_in+n_out) 
def xavier_init(fan_in, fan_out, constant=1):
    """xavier initialization function
    fan_in: input node number
    fan_out: output node number"""
    low = -constant*np.sqrt(6/(fan_in+fan_out))
    high = constant*np.sqrt(6/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):   
    # define construct function
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, 
                 optimizer=tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

       # define auto-encoder net structure
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, self.n_input])
            
        with tf.name_scope('hidden_layr'):
            self.hidden = self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)),
                                                     self.weights['w1']), self.weights['b1']))
            tf.summary.histogram('hidden',self.hidden)
           # tf.summary.image('hidden_image',self.hidden)
        with tf.name_scope('output_layr'):
            self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']), self.weights['b2'])
        # define loss function
        with tf.name_scope('loss_func'):
            self.cost = 0.5*tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        
        self.optimizer = optimizer.minimize(self.cost)
        # initialize all variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.merged = tf.summary.merge_all()
        
        # parameter initialize function
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input],dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights
    
    # 1 step train function
    def partial_fit(self, X):
        cost, opt, merged = self.sess.run((self.cost, self.optimizer, self.merged),feed_dict={self.x:X, self.scale:self.training_scale})
        
        return cost, merged
    
    # loss function
    def calc_total_cost(self,X):
        return self.sess.run(self.cost, feed_dict={self.x:X, self.scale:self.training_scale})



if __name__  == '__main__':
    mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)
    logdir = './auto_encoder_logdir'
    summary_writer = tf.summary.FileWriter(logdir)
    
    with tf.Graph().as_default():
        # define standard scale fucntion
        def standard_scale(X_train, X_test):
            preprocessor = prep.StandardScaler().fit(X_train)
            X_train = preprocessor.transform(X_train)
            X_test = preprocessor.transform(X_test)
            return X_train, X_test
        
        # define get random block function
        def get_random_block_from_data(data, batch_size):
            start_index = np.random.randint(0, len(data)-batch_size)
            return data[start_index:(start_index+batch_size)]
        
        X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
        
        n_samples = int(mnist.train.num_examples)        
        training_epochs = 20
        batch_size = 128
        display_step = 2
        
        autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784, 
                                                       n_hidden = 200,
                                                       transfer_function=tf.nn.softplus,
                                                       optimizer = tf.train.AdamOptimizer(learning_rate=0.001),
                                                       scale = 0.01
                                                       )
        
        # training process
     
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(n_samples/batch_size)
            for i in range(total_batch):
                
                batch_xs = get_random_block_from_data(X_train, batch_size)
                #cost = autoencoder.partial_fit(batch_xs)
                cost, merged = autoencoder.partial_fit(batch_xs)
                summary_writer.add_summary(merged, i)
                avg_cost += cost/n_samples*batch_size 
                if epoch%display_step == 0:
                    print('Epoch:','%04d'%(epoch+1), 'cost=','{:.9f}'.format(avg_cost))
                    
            print('Total cost:'+str(autoencoder.calc_total_cost(X_test)))
        summary_writer.close()
