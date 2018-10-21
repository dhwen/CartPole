import tensorflow as tf

class QModel:
    def __init__(self, input_dims, dropout_drop_prob=0):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder(dtype=tf.float32, shape=(None, *input_dims), name="Input")
            self.bIsTrain = tf.placeholder_with_default(False, shape=(), name="bIsTrain")
            self.drop_prob = dropout_drop_prob
            self.build_model()
            self.build_backprop()

    def build_model(self):

        fc1 = self.DenseStack(self.input, 8, 1)
        fc2 = self.DenseStack(fc1, 9, 2)
        fc3 = self.DenseStack(fc2, 8, 3)
        fc4 = self.DenseStack(fc3, 7, 4)
        fc5 = self.DenseStack(fc4, 7, 5)
        fc6 = self.DenseStack(fc5, 6, 6)
        fc7 = self.DenseStack(fc6, 2, 7)
        self.output = tf.nn.relu(fc7, name='Output')

    def DenseStack(self, inputs, nNodes, id):
        with tf.variable_scope("DenseStack"+str(id)):
            fc = tf.layers.dense(inputs, nNodes, name="FC")
            bn = tf.layers.batch_normalization(fc, name="BN")
            leaky_relu = tf.nn.leaky_relu(bn, alpha=0.01, name='LeakyRelu')
            dropout = tf.layers.dropout(leaky_relu, rate=self.drop_prob, training=self.bIsTrain, name='DropOut')
        return dropout

    def build_backprop(self):
        self.action_taken = tf.placeholder(dtype="float32", shape=(None, 1), name="ActionTaken")
        self.output_av0, self.output_av1 = tf.split(self.output, num_or_size_splits=2, axis=1)
        self.output_av_taken= tf.maximum(tf.multiply(self.output_av0,tf.subtract(1.0,self.action_taken)), tf.multiply(self.output_av1,self.action_taken))
        self.label = tf.placeholder(dtype="float32", shape=(None,1), name="Label")

        self.loss = tf.losses.mean_squared_error(self.label,self.output_av_taken)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        self.opt = self.optimizer.minimize(self.loss)