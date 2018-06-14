import tensorflow as tf

class QModel:
    def __init__(self, dropout_keep_prob=1):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build_layers(dropout_keep_prob)
            self.build_backprop()

    def build_layers(self,dropout_keep_prob):
        self.state = tf.placeholder(dtype="float32", shape=(None, 4), name="State")
        self.fc1 = tf.layers.dense(self.state, 4, name="FC1")
        self.bn1 = tf.layers.batch_normalization(self.fc1, name="BN1")
        self.relu1 = tf.nn.relu(self.bn1, name='Relu1')
        self.dropout1 = tf.nn.dropout(self.relu1, keep_prob=dropout_keep_prob, name='DropOut1')
        self.fc2 = tf.layers.dense(self.dropout1, 3, name="FC2")
        self.bn2 = tf.layers.batch_normalization(self.fc2, name="BN2")
        self.relu2 = tf.nn.relu(self.bn2, name='Relu2')
        self.dropout2 = tf.nn.dropout(self.relu2, keep_prob=dropout_keep_prob, name='DropOut2')
        self.fc3 = tf.layers.dense(self.dropout2, 2, name="FC3")
        self.output = tf.nn.relu(self.fc3, name='Output')

    def build_backprop(self):
        self.action_taken = tf.placeholder(dtype="float32", shape=(None, 1), name="ActionTaken")
        self.output_av0, self.output_av1 = tf.split(self.output, num_or_size_splits=2, axis=1)
        self.output_av_taken= tf.maximum(tf.multiply(self.output_av0,tf.subtract(1.0,self.action_taken)), tf.multiply(self.output_av1,self.action_taken))
        self.label = tf.placeholder(dtype="float32", shape=(None,1), name="Label")

        self.loss = tf.losses.mean_squared_error(self.label,self.output_av_taken)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        self.opt = self.optimizer.minimize(self.loss)