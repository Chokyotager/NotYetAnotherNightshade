from typing import Optional

import tensorflow as tf
from tensorflow import keras
import spektral


class Sampling (keras.layers.Layer):

    def call (self, inputs):

        z_mean, z_log_var, epochs = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        result = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # Calculate the annealing schedule
        beta = tf.math.minimum(tf.constant(1, dtype=tf.float32), tf.math.maximum(tf.constant(0, dtype=tf.float32), 0.02 * epochs - 0.2))

        loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        loss_beta = beta * loss / 15000

        self.add_loss(loss_beta)
        self.add_metric(loss, name="kl_loss")
        self.add_metric(loss_beta, name="kl_loss_beta")

        return result

class NyanModel (keras.Model):

    def __init__ (
            self,
            input_names: Optional[list[str]] = None,
            output_names: Optional[list[str]] = None,
            *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.input_names = input_names
        self.output_names = output_names

        self.epochs = tf.Variable(0, trainable=False, name="global_step", dtype=tf.float32)

    def call (self, inputs):

        if self.input_names is None:
            variables = {"x": inputs}

        elif len(self.input_names) == 1:
            variables = {self.input_names[0]: inputs}

        else:
            variables = dict(zip(self.input_names, inputs))

        variables["_epochs"] = self.epochs

        for layer in self.nyan_layers:

            if layer is None:
                continue

            if "layer" in layer:
                input_variables = list()

                for variable in layer["variables"]:
                    input_variables.append(variables[variable])

                if len(input_variables) == 1:
                    for output_variable in layer["outputs"]:
                        variables[output_variable] = layer["layer"](*input_variables)

                else:
                    for output_variable in layer["outputs"]:
                        variables[output_variable] = layer["layer"](input_variables)

            elif "concat" in layer:
                input_variables = [variables[variable] for variable in layer["concat"]]
                for output_variable in layer["outputs"]:
                    variables[output_variable] = tf.concat(input_variables, axis=-1)
            else:
                raise ValueError("Unknown layer type")

        outputs = self.output_names

        if outputs is None:
            outputs = self.nyan_layers[-1]["outputs"]

        outputs = [variables[output] for output in outputs]

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

class NyanEncoder (NyanModel):

    # Fingerprint bits for MACCS (RDKit) = 167
    def __init__ (self, latent_dim=64, batched=True):

        super().__init__(input_names=["x", "a", "e"], output_names=["z"])
        #super().__init__()

        self.masking = spektral.layers.GraphMasking() if batched else None

        self.precondition = keras.layers.Dense(16, activation=keras.layers.LeakyReLU(alpha=0.05))

        self.graphconv1 = spektral.layers.ECCConv(32, activation=keras.layers.LeakyReLU(alpha=0.05))
        self.graphconv2 = spektral.layers.ECCConv(32, activation=keras.layers.LeakyReLU(alpha=0.05))
        self.graphconv3 = spektral.layers.ECCConv(32, activation=keras.layers.LeakyReLU(alpha=0.05))

        self.pool1 = spektral.layers.GlobalSumPool()

        self.dense1 = keras.layers.Dense(256, activation=keras.layers.LeakyReLU(alpha=0.05))

        self.flatten = keras.layers.Flatten() if batched else None

        self.dense2 = keras.layers.Dense(256, activation=keras.layers.LeakyReLU(alpha=0.05))

        self.z_mean = keras.layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")
        self.latent_z = Sampling()

        self.nyan_layers = [
            {"layer": self.masking, "variables": ["x"], "outputs": ["x"]} if batched else None,

            {"layer": self.precondition, "variables": ["x"], "outputs": ["x"]},

            {"layer": self.graphconv1, "variables": ["x", "a", "e"], "outputs": ["x"]},
            {"layer": self.graphconv2, "variables": ["x", "a", "e"], "outputs": ["x"]},
            {"layer": self.graphconv3, "variables": ["x", "a", "e"], "outputs": ["x"]},

            {"layer": self.pool1, "variables": ["x"] if batched else ["x", "i"], "outputs": ["x"]},

            {"layer": self.dense1, "variables": ["x"], "outputs": ["x"]},
            {"layer": self.flatten, "variables": ["x"], "outputs": ["x"]} if batched else None,
            {"layer": self.dense2, "variables": ["x"], "outputs": ["x"]},

            {"layer": self.z_mean, "variables": ["x"], "outputs": ["z_mean"]},
            {"layer": self.z_log_var, "variables": ["x"], "outputs": ["z_log_var"]},
            {"layer": self.latent_z, "variables": ["z_mean", "z_log_var", "_epochs"], "outputs": ["z"]}
        ]

class NyanDecoder (NyanModel):

    # Fingerprint bits for MACCS (RDKit) = 167
    def __init__ (self, fingerprint_bits=167, regression=1613):

        self.fingerprint_bits = fingerprint_bits
        self.regression = regression

        # super().__init__(output_names=["fingerprint", "regression", "y"])
        super().__init__()

        self.dense3 = keras.layers.Dense(256, activation=keras.layers.LeakyReLU(alpha=0.05))

        self.fingerprint = keras.layers.Dense(fingerprint_bits, activation=keras.activations.sigmoid)
        self.regression = keras.layers.Dense(regression)

        self.nyan_layers = [
            {"layer": self.dense3, "variables": ["x"], "outputs": ["x"]},

            {"layer": self.fingerprint, "variables": ["x"], "outputs": ["fingerprint"]},
            {"layer": self.regression, "variables": ["x"], "outputs": ["regression"]},
            {"concat": ["fingerprint", "regression"], "outputs": ["y"]}
        ]

class VAE (keras.Model):

    def __init__ (self, encoder, decoder):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # Inherit
        self.epochs = encoder.epochs

    def call (self, inputs):

        z = self.encoder(inputs)
        y = self.decoder(z)

        return y

    # 1st: 2e-4, 2nd: 7e-5, 3rd: 2e-5
    def nyanCompile (self, lr=2e-4):

        """
        def loss (y_true, y_pred):

            # Create a skew to the 30th percentile to adjust for kurtosis
            weights = tf.math.abs(y_true - 50) * 0.1 + 1
            squared_difference = tf.square(y_true - y_pred)

            return tf.reduce_mean(squared_difference, axis=-1) * weights
        """

        self.learning_rate = keras.optimizers.schedules.InverseTimeDecay(lr, 300000, 0.35)

        def regression_mae (y_true, y_pred):
            fp_bits = self.decoder.fingerprint_bits
            return keras.losses.mean_absolute_error(y_true[:, fp_bits:], y_pred[:, fp_bits:])

        binary_accuracy_metric = keras.metrics.BinaryAccuracy()
        binary_accuracy_metric.reset_state()

        def binary_acc (y_true, y_pred):
            fp_bits = self.decoder.fingerprint_bits
            binary_accuracy_metric.update_state(y_true[:, :fp_bits], y_pred[:, :fp_bits])

            return binary_accuracy_metric.result()

        def combined_loss (y_true, y_pred):
            fp_bits = self.decoder.fingerprint_bits

            loss1 = keras.losses.binary_crossentropy(y_true[:, :fp_bits], y_pred[:, :fp_bits])
            loss2 = keras.losses.huber(y_true[:, fp_bits:], y_pred[:, fp_bits:], delta=0.5)

            return loss1 + loss2

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999,
                                               epsilon=1e-08)

        self.compile(loss=combined_loss, optimizer=self.optimizer, metrics=[binary_acc, regression_mae])

class EpochCounter (keras.callbacks.Callback):

    def __init__ (self, epoch):
        self.epoch = epoch

    def on_epoch_begin (self, epoch, logs):
        # Use the stored epoch instead of TensorFlow defined
        keras.backend.set_value(self.epoch, self.epoch + 1)
