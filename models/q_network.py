import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.losses import Huber

 # Huber loss function
@register_keras_serializable(package='Custom', name="CustomHuberLoss")  # Added decorator to keep using this "custom?" huber loss function.
class CustomHuberLoss:
    def __init__(self, clip_value=1):
        self.clip_value = float(clip_value) # changed from int to float

    def __call__(self, y_true, y_pred):
        assert self.clip_value > 0.

        x = y_true - y_pred
        clip_value_tensor = tf.constant(self.clip_value, dtype=tf.float32)
        """if tf.math.is_inf(self.clip_value): # changed from np.inf to tf.math.isinf
            # Spacial case for infinity since Tensorflow does have problems
            # if we compare `K.abs(x) < np.inf`.
            return .5 * K.square(x)"""
        
        def inf_case():
            return .5 * tf.square(x)
        
        def non_inf_case():
            condition = tf.abs(x) < clip_value_tensor
            squared_loss = .5 * tf.square(x)
            linear_loss = clip_value_tensor * (tf.abs(x) - .5 * clip_value_tensor)
            return tf.where(condition, squared_loss, linear_loss)
        
        """
        condition = tf.abs(x) < self.clip_value
        squared_loss = .5 * tf.square(x)
        linear_loss = self.clip_value * (tf.abs(x) - .5 * self.clip_value)
        return tf.where(condition, squared_loss, linear_loss)"""
        return tf.cond(tf.math.is_inf(clip_value_tensor), inf_case, non_inf_case)
    
    def get_config(self):
        return {'clip_value': self.clip_value}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def huber_loss(y_true, y_pred):
    return CustomHuberLoss()(y_true, y_pred)


class QNetwork():
    """
    Q-Network Estimator
    Represents the global model for the table
    """

    def __init__(self, obs_size, num_actions, hidden_size=100,
                 hidden_layers=1, learning_rate=.2):
        """
        Initialize the network with the provided shape
        """
        self.obs_size = obs_size
        self.num_actions = num_actions

        # Network arquitecture
        self.model = tf.keras.Sequential()
        # Add imput layer
        # self.model.add(Dense(hidden_size, input_shape=(obs_size,), # Deactivated this since keras had error: '87: UserWarning: Do not pass an input_shape/input_dim argument to a layer. When using Sequential models, prefer using an Input(shape) object as the first layer in the model instead.'
        #                     activation='relu'))
        # Add my input Layer
        self.model.add(tf.keras.Input(shape=(obs_size,)))  # use this input layer to avoid problem
        # Add hidden layers
        for layers in range(hidden_layers):
            self.model.add(tf.keras.layers.Dense(hidden_size, activation='relu')) # TODO: Check if this is the same as Dense in Keras
        # Add output layer
        self.model.add(tf.keras.layers.Dense(num_actions)) # TODO: Check if this is the same as Dense in Keras # changed from tf.keras.Layer.Dense to tf.keras.layers.Dense

        # optimizer = optimizers.SGD(learning_rate)
        # optimizer = optimizers.Adam(alpha=learning_rate)
        optimizer = optimizers.Adam(0.00025)
        # optimizer = optimizers.RMSpropGraves(learning_rate, 0.95, self.momentum, 1e-2)

        # Compilation of the model with optimizer and loss
        self.model.compile(loss=CustomHuberLoss(), optimizer=optimizer)

    def predict(self, state, batch_size=1):
        """
        Predicts action values.
        """
        return self.model.predict(state, batch_size=batch_size)

    def update(self, states, q):
        """
        Updates the estimator with the targets.

        Args:
          states: Target states
          q: Estimated values

        Returns:
          The calculated loss on the batch.
        """
        loss = self.model.train_on_batch(states, q)
        return loss

    @staticmethod
    def copy_model(model):
        """Returns a copy of a keras model."""
        model.save('tmp_model.keras')  # Added '.keras' extension here
        return tf.keras.models.load_model(
            'tmp_model.keras', custom_objects={'CustomHuberLoss': CustomHuberLoss})  # Added '.keras' extension here to avoid ValueError: Invalid filepath extension for saving. + custom_objects={'CustomHuberLoss': huber_loss}


   
# Needed for keras huber_loss locate TODO can I remove those lines?
#tf.keras.losses.huber_loss = huber_loss


""" Backup of original method / Author: gcamfer
def huber_loss(y_true, y_pred, clip_value=1):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if tf.math.isinf(clip_value): # changed from np.inf to tf.math.isinf
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = tf.abs(x) < clip_value
    squared_loss = .5 * tf.square(x)
    linear_loss = clip_value * (tf.abs(x) - .5 * clip_value)
    '''
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))
    '''
    # TensorFLow's `tf.where` replaces Keras' `tf.select`
    return tf.where(condition, squared_loss, linear_loss)
"""