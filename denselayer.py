import tensorflow as tf

#atvication functions configuration
activations = {
    "sigmoid":tf.nn.sigmoid,
    "relu":tf.nn.relu,
    }

class DenseLayer(tf.keras.layers.Layer):
    """
        x: input to the layer
        n_output_nodes: number of output nodes
        x: input to the layer

        example of usage:
        >>> layer = DenseLayer(3, activation='sigmoid')
        >>> layer.build((1,2))
        >>> x_input = tf.constant([[1,2.]], shape=(1,2))
        >>> y = layer.call(x_input)
    """

    def __init__(self, n_output_nodes, activation="sigmoid"):    
        super(DenseLayer, self).__init__()
        self.n_output_nodes = n_output_nodes
        self.activation = activations[activation]

    def build(self, input_shape):
        d = int(input_shape[-1])
        # Define and initialize parameters: a weight matrix W and bias b
        # parameter initialization is random!
        self.W = self.add_weight("weight", shape=[d, self.n_output_nodes]) 
        self.b = self.add_weight("bias", shape=[1, self.n_output_nodes])

    def call(self, x):
        '''define the operation for z(z = xw + b)'''
        z = tf.matmul(x, self.W) + self.b

        '''define the operation for out(y=g(z))'''
        y = self.activation(z)
        return y
