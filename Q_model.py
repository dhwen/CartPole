import tensorflow as tf

class QModel:
    def __init__(self, drop_prob=0):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.state = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="State")
            #self.bIsTrain = tf.placeholder(dtype=tf.bool, shape=(), name="bIsTrain")

            self.bIsTrain = tf.placeholder_with_default(False, shape=(), name="bIsTrain")
            self.drop_prob = drop_prob

            self.build_layers()
            self.build_backprop()

    def build_layers(self):

        self.fc1 = self.DenseStack(self.state, 5, 1)
        self.fc2 = self.DenseStack(self.fc1, 4, 2)
        self.fc3 = self.DenseStack(self.fc2, 4, 3)
        self.fc4 = tf.layers.dense(self.fc3, 2, name="FC4")
        self.output = tf.nn.relu(self.fc4, name='Output')

    def DenseStack(self, inputs, nNodes, id):
        with tf.variable_scope("DenseStack"+str(id)):
            fc = tf.layers.dense(inputs, nNodes, name="FC")
            bn = tf.layers.batch_normalization(fc, name="BN")
            relu = tf.nn.relu(bn, name='Relu')
            dropout = tf.layers.dropout(relu, rate=self.drop_prob, training=self.bIsTrain, name='DropOut')
        return dropout

    def build_backprop(self):
        self.action_taken = tf.placeholder(dtype="float32", shape=(None, 1), name="ActionTaken")
        self.output_av0, self.output_av1 = tf.split(self.output, num_or_size_splits=2, axis=1)
        self.output_av_taken= tf.maximum(tf.multiply(self.output_av0,tf.subtract(1.0,self.action_taken)), tf.multiply(self.output_av1,self.action_taken))
        self.label = tf.placeholder(dtype="float32", shape=(None,1), name="Label")

        self.loss = tf.losses.mean_squared_error(self.label,self.output_av_taken)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        self.opt = self.optimizer.minimize(self.loss)