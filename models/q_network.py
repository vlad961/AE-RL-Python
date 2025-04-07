import os
import tensorflow as tf
from tensorflow.python.keras.losses import Huber
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from utils.config import TEMPORARY_MODEL_PATH
# FIXME: --> change to tf.keras.* check whether the code still works, maybe performance improves. Remove the import of the old keras library.


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
        self.model: Sequential = Sequential(name=model_name)
        # Add input layer
        self.model.add(InputLayer(input_shape=(obs_size,)))  # use this input layer to avoid problem
        # Add hidden layers
        for _ in range(hidden_layers):
            self.model.add(Dense(hidden_size, activation='relu'))
        # Add output layer
        self.model.add(Dense(num_actions)) # TODO: Check whether I achieve better results if I add a softmax activation function here.

        optimizer = Adam(learning_rate=learning_rate)

        # Compilation of the model with optimizer and loss
        self.model.compile(loss=Huber(delta=1.0), optimizer=optimizer,
                            metrics=["mse", "mae"])

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
    def copy_model(model: Sequential) -> Sequential:
        """Returns a copy of a keras model."""

        """        cloned_model = tf.keras.models.clone_model(model)
        cloned_model.set_weights(model.get_weights())
        return cloned_model"""
        model.save(TEMPORARY_MODEL_PATH)
        return tf.keras.models.load_model(TEMPORARY_MODEL_PATH)
