from __future__ import absolute_import

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf

class GraphFlusing(Layer):

    def __init__(self,
                 attn_heads_reduction='attention',  # {'concat', 'average'}
                 activation= None,
                 attn_kernel_initializer='glorot_uniform',
                 attn_kernel_regularizer=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'sum', 'average','max','attention'}:
            raise ValueError('Possbile reduction methods: sum, average,max,attention')

        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        if activation:
            self.activation = activations.get(activation)  # Eq. 4 in the paper
        else:
            self.activation = None

        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        super(GraphFlusing, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]        
        # Initialize weights for each attention head
        self.attn_kernels = self.add_weight(shape=(F, 1),
                                             initializer=self.attn_kernel_initializer,
                                             regularizer=self.attn_kernel_regularizer,
                                             constraint=self.attn_kernel_constraint,
                                             name='attn_kernel_neigh')
        self.built = True

    def call(self, inputs):

        # Aggregate the heads' output according to the reduction method          
        atten_map = []
        for input0 in inputs:
#            print('ok------------',K.int_shape(input0))
#            print('fit------------',K.int_shape(self.attn_kernels))
            attention_kernel = self.attn_kernels  # Attention kernel a in the paper (F x 1)

            # Compute inputs to attention network
            single_atten = K.dot(input0, attention_kernel)  # (N x F) * (F x 1) = ( N x 1)
            print('------------',K.int_shape(single_atten))
            atten_map.append(single_atten)
                
        atten_map = K.concatenate(atten_map)
        print('atten_map------------',K.int_shape(atten_map))        
        atten_map = K.tanh(atten_map)
        atten_map = K.softmax(atten_map)  # (N x 1)
        
        output = 0
        for ID in range(len(inputs)):
            
  
#            print('put------------',K.int_shape(inputs[ID]))       
#            print('put------------',type(atten_map[:,ID]))
            output += K.expand_dims(atten_map[:,ID]) * inputs[ID]


#            print('put------------',K.int_shape(atten_map[:,ID]))

        
#        if self.attn_heads_reduction == 'sum':
#            output = K.sum(K.stack(inputs), axis=0)  # (N x F)
#        elif self.attn_heads_reduction == 'average':         
#            output = K.mean(K.stack(inputs), axis=0)  # (N x F)
#        elif self.attn_heads_reduction == 'max':         
#            output = K.max(K.stack(inputs), axis=0)  # (N x F)  



        print('out------------',K.int_shape(output))
        if self.activation:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        # Output will have shape (..., F')
#        print('shape',input_shape[0])
        output_shape = input_shape[0]#input_shape[0][0], self.output_dim
        return output_shape
