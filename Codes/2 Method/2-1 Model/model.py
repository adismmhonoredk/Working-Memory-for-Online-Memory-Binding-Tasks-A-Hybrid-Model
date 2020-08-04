#Mahdi Heidarpoor

import numpy as np
import tensorflow as tf 
from tensorflow.keras.layers import Dense
from typing import Union

# Dual-component model of working memory (DCWM)
# Model parameters can also set in jupyter file 
class DCWM(tf.keras.Model):

    def __init__(self,
                 input_dim: int, 
                 output_dim:int,
                 controller_dime:int=512,
                 interface_dim: int=350,
                 netsize: int=1000, 
                 degree: int=20, 
                 k:float=2,
                 name:str='dcwm')->None:

        """
        Initialize dcwm object.

        Parameters
        ----------
        input_dim
            Size of input vector is free
        output_dim
            Size of output vector
        interface_dim
            size of vector to random network (and from random)
        netsize
            random network netsize
        degree
            random network connection to/from each neurons
        1/k
            random network maximum synaptic strength 
        name
            Name of dcwm.

        numbers pick from valiant 2009 , 2012 alpha regima and buschman 2019 ...
        """
        
        super(DCWM,self).__init__(name=name)
        self.input_dim=input_dim
        self.output_dim=output_dim # Y = Output_dim
        self.interface_dim=interface_dim
        self.netsize=netsize
        self.degree=degree
        self.k=k
        self.neg_to_all=1/3
        self.controller_dim= controller_dime
        self.network_forget_rate=1/3    #Random network forgeting data in time, see testing random network
        self.show_interface_vector=True
        self.interface_vector_processes=[]

        # Initialize Random Network States
        self.network_state=np.zeros(self.netsize)

        #Creating Random Ntwork Wieghts -- using numpy because its faster than TF
        self.density=self.degree/self.netsize
        r=np.random.uniform(0,1,(self.netsize,self.netsize))
        r2=np.random.uniform(0,1,(self.netsize,self.netsize))
        self.randomnetwork=np.zeros((self.netsize,self.netsize))
        for row in range(self.netsize):
            for col in range(self.netsize):
                if r[row,col] < self.density:
                    if r2[row,col] <self.neg_to_all:
                        self.randomnetwork[row,col]=-2/self.k
                    else:
                        self.randomnetwork[row,col]=1/self.k
        
        # Initialize controller output and interface vector with gaussian normal
        self.output_v = tf.random.uniform([1, self.output_dim],minval=0,maxval=1) 
        self.interface = tf.random.uniform([1, self.interface_dim],minval=0,maxval=1)  

        # initialise Dense layers of the controller
        self.dense1 = Dense(
                         self.controller_dim,
                         name='dcwm_controller')
        self.dense2 = Dense(
                         2*self.controller_dim,
                         name='dcwm_controller')
        self.dense3 = Dense(
                         self.controller_dim,
                         name='dcwm_controller')
        self.dense_output = Dense(
                         self.interface_dim+self.output_dim, # last_layer_of_dense_dim
                         activation='sigmoid',
                         name='dcwm_controller')
        
        # Derived Output and interface vector from controller last layer (dense_output)       
        temp=np.zeros((self.interface_dim+self.output_dim, self.output_dim))
        temp_dim=self.interface_dim+self.output_dim
        for i in range(temp_dim):
            for j in range(self.output_dim):
                if (i==j): temp[i,j]=1
                else : temp[i,j]=0

        self.W_output = tf.Variable( #[last_layer_of_dense_dim,output_dim]
            tf.constant(temp,dtype=tf.float32),
            trainable=False,
            name='dcwm_net_output_weights'
        )

        temp=np.zeros((self.interface_dim+self.output_dim, self.interface_dim))
        temp_dim=self.interface_dim+self.output_dim
        for i in range(temp_dim):
            for j in range(self.interface_dim):
                if (i-self.output_dim==j): temp[i,j]=1
                else : temp[i,j]=0

        self.W_interface = tf.Variable(  # [last_layer_of_dense_dim,interface_dim]
            tf.constant(temp,dtype=tf.float32),
            trainable=False,
            name='dcwm_interface_weights'
        )


    # This Two functions are the same just for diferenet job!
    def Write_Vector(self):
        "Getting Write Vector from NN to RN"
        if self.show_interface_vector :
            self.interface_vector_processes.append(self.interface.numpy()[0])
    
    def Read_Vector(self):
        "Getting Read Vector from RN for NN"
        if self.show_interface_vector :
            self.interface_vector_processes.append(self.interface.numpy())
   
    def random_network_updating(self):
        """
        1- update random network states
        2- no weights change
        """

        #update fire states with interface vector
        for i in range(self.interface_dim):
            self.network_state[i]=self.network_state[i]+self.interface[0,0,i]
        
        for i in range(self.netsize):
            s=np.dot(self.network_state,self.randomnetwork[i])*self.network_forget_rate
            if s>0:
                self.network_state[i]=s
            else:
                self.network_state[i]=0
        
        #changing size
        temp=np.zeros(self.interface_dim)
        for i in range(self.interface_dim):
            temp[i]=self.network_state[i]
        
        self.interface=temp
        self.interface=tf.convert_to_tensor(self.interface.reshape(1,self.interface_dim))
        
    def step(self, x: tf.Tensor) -> tf.Tensor:
        """
        Update the controller, compute the output and interface vectors,
        write to and read from memory and compute the output.
        """

        # Read
        self.Read_Vector()
        x_in = tf.expand_dims(tf.convert_to_tensor(np.concatenate([x.numpy(), self.interface.numpy()],1)), axis=0)
       
        # Update controller
        y_out = self.dense_output(self.dense3(self.dense2(self.dense1(x_in))))

        # Compute output and interface vectors
        self.output_v = tf.matmul(y_out, self.W_output)  # [1,last_layer_of_dense_dim * [last_layer_of_dense_dim,output_dim] -> [1,output_dim]
        self.interface = tf.matmul(y_out, self.W_interface)  # [1,last_layer_of_dense_dim] * [last_layer_of_dense_dim,interface_dim] -> [1,interface_dim]
        
        # Write
        self.Write_Vector()
        self.random_network_updating()
        
        return self.output_v

    def call(self, x: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        """ Unstack the input, run through the DCWM and return the stacked output. """

        y = []
        for x_seq in tf.unstack(x, axis=0):
            x_seq = tf.expand_dims(x_seq, axis=0)
            y_seq = self.step(x_seq)
            y.append(y_seq)
        if self.show_interface_vector :
            return tf.squeeze(tf.stack(y, axis=0)) , self.interface_vector_processes
        else:
            return tf.squeeze(tf.stack(y, axis=0))

