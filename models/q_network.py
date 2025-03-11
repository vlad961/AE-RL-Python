import os
import tensorflow as tf
from tensorflow.python.keras.losses import Huber
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import Precision, Recall

cwd = os.getcwd()
models_dir = os.path.join(cwd, "models/trained-models/")

class QNetwork():
    """
    Q-Network Estimator
    Represents the global model for the table
    """

    def __init__(self, obs_size, num_actions, hidden_size=100,
                 hidden_layers=1, learning_rate=0.2, model_name='model'):
        """
        Initialize the network with the provided shape
        """
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.model_name = model_name

        # Network arquitecture // TODO: add tf.keras.layers.Normalization() to normalize the input
        self.model = tf.keras.Sequential(name=model_name)
        # Add input layer
        # self.model.add(Dense(hidden_size, input_shape=(obs_size,), # Deactivated this since keras had error: '87: UserWarning: Do not pass an input_shape/input_dim argument to a layer. When using Sequential models, prefer using an Input(shape) object as the first layer in the model instead.'
        #                     activation='relu')) # TODO: recheck if this is the same as Dense in Keras
        self.model.add(tf.keras.Input(shape=(obs_size,)))  # use this input layer to avoid problem
        # Add hidden layers
        for _ in range(hidden_layers):
            self.model.add(tf.keras.layers.Dense(hidden_size, activation='relu'))
        # Add output layer
        self.model.add(tf.keras.layers.Dense(num_actions)) # TODO: Check whether I achieve better results if I add a softmax activation function here.

        # optimizer = optimizers.SGD(learning_rate)
        # optimizer = optimizers.Adam(alpha=learning_rate)
        #optimizer = optimizers.Adam(0.00025)
        optimizer = optimizers.Adam(learning_rate)
        # optimizer = optimizers.RMSpropGraves(learning_rate, 0.95, self.momentum, 1e-2)

        #f1_score = F1Score() -> use this if you want to use the F1 score as a metric in the model. For my OS configs, this is not available.
        # Compilation of the model with optimizer and loss
        self.model.compile(loss=Huber(delta=1.0), optimizer=optimizer,
                            metrics=["accuracy", Precision(), Recall(), 
                                     tf.keras.metrics.AUC(num_thresholds=100, name='auc')])

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
        cloned_model = tf.keras.models.clone_model(model)
        cloned_model.set_weights(model.get_weights())
        return cloned_model