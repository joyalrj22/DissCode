
from tensorflow import TensorShape
import keras.layers
import keras.backend

# Bahdanaou Attention

class ContextVector(keras.layers.Layer):

    def __init__(self, **args):
        # override build, call and compute_output_shape functions
        self.currentInputs = (None, None)
        super(ContextVector, self).__init__(**args)

    
    def compute_output_shape(self, in_shape):
        dim1 = TensorShape((in_shape[1][0], in_shape[1][1], in_shape[1][2]))
        dim2 = TensorShape((in_shape[1][0], in_shape[1][1], in_shape[0][1]))

        return [dim1, dim2]
    

    def build(self, input_shape):
        
        self.encoder_weight_matrix = self.add_weight(shape=TensorShape((input_shape[0][2], input_shape[0][2])),
            initializer=keras.initializers.lecun_uniform(),
            trainable=True)

        self.decoder_weight_matrix = self.add_weight(shape=TensorShape((input_shape[1][2], input_shape[0][2])),
            initializer=keras.initializers.lecun_uniform(),
            trainable=True)

        self.combined_vector = self.add_weight(shape=TensorShape((input_shape[0][2], 1)),
            initializer=keras.initializers.lecun_uniform(),
            trainable=True)

        super(ContextVector, self).build(input_shape)

    def calculateAttentionWeight(self, inputs, decoder_states):
        encoded_frames = keras.backend.reshape(self.currentInputs[0], (-1, self.currentInputs[0].shape[2]))
        encoded_frames_weighted = keras.backend.dot(encoded_frames, self.encoder_weight_matrix)
        encoded_frames_weighted = keras.backend.reshape(encoded_frames_weighted, (-1, self.currentInputs[0].shape[1],self.currentInputs[0].shape[2]))

        decoded_frames_weighted = keras.backend.dot(inputs, self.decoder_weight_matrix)
        decoded_frames_weighted = keras.backend.expand_dims(decoded_frames_weighted, 1)

        combined_weights = encoded_frames_weighted + decoded_frames_weighted
        combined_weights = keras.backend.reshape(combined_weights, (-1, self.currentInputs[0].shape[2]))
        activated_combined_weights = keras.backend.tanh(combined_weights)

        alpha = keras.backend.dot(activated_combined_weights, self.combined_vector)
        alpha = keras.backend.reshape(alpha, (-1, self.currentInputs[0].shape[1]))

        attentionWeight = keras.backend.softmax(alpha)

        return attentionWeight, [attentionWeight]

    def calculateContextVector(self, attentionWeight, previousStates):

        #make attentionWeight match size of previousStates
        resized_weight = keras.backend.expand_dims(attentionWeight, -1)
        context = keras.backend.sum(self.currentInputs[0]*resized_weight, axis=1)
        return context, [context]


    def call(self, currentInputs):
        #currentInputs must be accessed when calculating weights
        self.currentInputs = currentInputs
        
        initialAttention = keras.backend.sum(keras.backend.zeros_like(self.currentInputs[0]), axis=[1,2])
        initialAttention = [keras.backend.tile(keras.backend.expand_dims(initialAttention), [1, self.currentInputs[0].shape[1]])]
        weights = keras.backend.rnn(self.calculateAttentionWeight, self.currentInputs[1], initialAttention)

        initialContext = keras.backend.sum(keras.backend.zeros_like(self.currentInputs[0]), axis=[1,2])
        initialContext = [keras.backend.tile(keras.backend.expand_dims(initialContext), [1, self.currentInputs[0].shape[-1]])]
        context_states = keras.backend.rnn(self.calculateContextVector, weights[1], initialContext)

        return [context_states[1], weights[1]]

        
      


