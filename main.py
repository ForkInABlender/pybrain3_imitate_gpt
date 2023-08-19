# Dylan Kenneth Eliot & GPT-4-codeinterpreter & GPT-4-plugins
"""

This mostly works to imitate some of the logic going on. Now what is left is the fine tuning, the proper layering for each piece of GPT as a set of disaggrogated neural networks, and fine tuning. The rest is token parsing, and connect the dots on training data. (Lots of R&D..... lots..... lots.........)
 On the bright side, a more lightweight, open sourced, easily tunable based on training data, and other optimizations not shown here be used as well. A lot of consideration went into how to get a basic model close enough to be testible even on replit. The goal is to add after applying customized training data
  eeg training data and neural network architecture. This too would mean furthermore disaggrogating the parts. Each part would then need to be trained separately.

  If one is going to attempt to build AI, one must see the world as others do around the world, not just know how to translate and speak their language; culture is not one language to be simply summed up for translation or being lost. 



"""
from pybrain3.structure import SoftmaxLayer, TanhLayer, LinearLayer, SigmoidLayer, GaussianLayer, LSTMLayer
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.structure import RecurrentNetwork, FeedForwardNetwork, FullConnection
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.structure.modules.neuronlayer import NeuronLayer
import numpy as np
#import pickle

class GeLULayer(NeuronLayer):
	def _forwardImplementation(self, inbuf, outbuf):
		outbuf[:] = 0.5 * inbuf * (
		 1 + np.tanh(np.sqrt(2 / np.pi) * (inbuf + 0.044715 * inbuf**3)))
	def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
		cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (inbuf + 0.044715 * inbuf**3)))
		pd = np.sqrt(2 / np.pi) * (1 + 0.134145 * inbuf**2) * ( 1 / np.cosh(np.sqrt(2 / np.pi) * (inbuf + 0.044715 * inbuf**3)))**2
		inerr[:] = outerr * (cdf + inbuf * pd)
#
class AttentionLayer(NeuronLayer):
  def __init__(self, indim, outdim):
    super().__init__(indim, outdim)
    self.attention_weights = np.random.rand(indim, outdim)
  def _forwardImplementation(self, inbuf, outbuf):
    outbuf[:] = np.dot(inbuf, self.attention_weights)
  def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
    gradient = np.dot(inbuf.T, outerr)
    self.attention_weights -= gradient
    inerr[:] = np.dot(outerr, self.attention_weights.T)
#
class MultiHeadSelfAttention(NeuronLayer):
  def __init__(self, indim, outdim, num_heads):
    super().__init__(indim, outdim)
    self.num_heads = num_heads
    self.depth = indim // num_heads
    # Initialize weights for linear projections to Q, K, V
    self.W_q = np.random.rand(indim, indim)
    self.W_k = np.random.rand(indim, indim)
    self.W_v = np.random.rand(indim, indim)
    # Initialize weights for final linear layer
    self.W_o = np.random.rand(indim, outdim)
  def scaled_dot_product_attention(self, Q, K, V):
    matmul_qk = np.dot(Q, K.T)
    d_k = Q.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)
    # Attention weights
    attention_weights = np.softmax(scaled_attention_logits, axis=-1)
    # Output
    output = np.dot(attention_weights, V)
    return output
  def _forwardImplementation(self, inbuf, outbuf):
    self.Q = np.dot(inbuf, self.W_q)
    self.K = np.dot(inbuf, self.W_k)
    self.V = np.dot(inbuf, self.W_v)
    # Split Q, K, V into multiple heads
    self.Q = np.split(self.Q, self.num_heads, axis=1)
    self.K = np.split(self.K, self.num_heads, axis=1)
    self.V = np.split(self.V, self.num_heads, axis=1)
    attention_heads = []
    for i in range(self.num_heads):
      attention_head = self.scaled_dot_product_attention(self.Q[i], self.K[i], self.V[i])
      attention_heads.append(attention_head)
    # Concatenate attention heads and pass through the final linear layer
    self.concatenated_heads = np.concatenate(attention_heads, axis=1)
    outbuf[:] = np.dot(self.concatenated_heads, self.W_o)
		#
  def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
    # Gradient w.r.t concatenated heads
    d_concat_heads = np.dot(outerr, self.W_o.T)
    # Split the gradient for each head
    d_attention_heads = np.split(d_concat_heads, self.num_heads, axis=1)
    dQ_total, dK_total, dV_total = 0, 0, 0
    for i in range(self.num_heads):
      d_out = d_attention_heads[i]
      # Gradient w.r.t V (using attention weights from the forward pass)
      dV = np.dot(self.attention_weights[i].T, d_out)
      # Gradient w.r.t attention weights
      d_attention_weights = np.dot(d_out, self.V[i].T)
      # Gradient w.r.t Q and K (chain rule through softmax and dot product)
      # This step is highly simplified for illustration; in practice, it involves
      # handling the chain rule through the softmax function and the dot product
      dQK = d_attention_weights * (1 - self.attention_weights[i]) * self.attention_weights[i]
      dQ = np.dot(dQK, self.K[i])
      dK = np.dot(dQK, self.Q[i])
      dQ_total += dQ
      dK_total += dK
      dV_total += dV
    # Gradient w.r.t linear transformations
    inerr[:] = np.dot(dQ_total, self.W_q.T) + np.dot(dK_total, self.W_k.T) + np.dot(dV_total, self.W_v.T)    
    # Update weights
    self.W_q -= np.dot(inbuf.T, dQ_total)
    self.W_k -= np.dot(inbuf.T, dK_total)
    self.W_v -= np.dot(inbuf.T, dV_total)
    self.W_o -= np.dot(self.concatenated_heads.T, outerr)
#

# Create the network
net = FeedForwardNetwork()
# Input layer
inLayer = SigmoidLayer(50000)
net.addInputModule(inLayer)
# Multi-head self-attention layer
attentionLayer = MultiHeadSelfAttention(48, 48, num_heads=8)  # Here we assume 8 heads for example
net.addModule(attentionLayer)
# Existing GeLU layers
hidden_layers = []
for i in range(96):
    hiddenLayer = GeLULayer(4)
    net.addModule(hiddenLayer)
    hidden_layers.append(hiddenLayer)
# Output layer
outLayer = SigmoidLayer(50000)
net.addOutputModule(outLayer)
# Connections
net.addConnection(FullConnection(inLayer, attentionLayer))
net.addConnection(FullConnection(attentionLayer, hidden_layers[0]))
for i in range(95):
    net.addConnection(FullConnection(hidden_layers[i], hidden_layers[i + 1]))
net.addConnection(FullConnection(hidden_layers[-1], outLayer))
net.sortModules()
"""for saving a snapshot to disk"""
##pickle.dump(net, open("nework.pkl", "wb"))
"""for loading"""
##net2=pickle.load(open("nework.pkl", "rb"))
