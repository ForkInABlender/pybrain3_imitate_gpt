# Dylan Kenneth Eliot & GPT-4-codeinterpreter & GPT-4-plugins
"""

This mostly works to imitate some of the logic going on. Now what is left is the fine tuning, the proper layering for each piece of GPT as a set of disaggrogated neural networks, and fine tuning. The rest is token parsing, and connect the dots on training data. (Lots of R&D..... lots..... lots.........)
 On the bright side, a more lightweight, open sourced, easily tunable based on training data, and other optimizations not shown here be used as well. A lot of consideration went into how to get a basic model close enough to be testible even on replit. The goal is to add after applying customized training data
  eeg training data and neural network architecture. This too would mean furthermore disaggrogating the parts. Each part would then need to be trained separately.

  If one is going to attempt to build AI, one must see the world as others do around the world, not just know how to translate and speak their language; culture is not one language to be simply summed up for translation or being lost. 



"""
from pybrain3.structure import SoftmaxLayer, LinearLayer
from pybrain3.datasets import SupervisedDataSet
from pybrain3.structure import FeedForwardNetwork, FullConnection
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.structure.modules.neuronlayer import NeuronLayer
import numpy as np

class FeedForwardLayer(NeuronLayer):
  def __init__(self, indim, outdim):
    super().__init__(indim, outdim)
    self.weights = np.random.randn(indim, outdim)
    self.bias = np.random.randn(outdim)
  def _forwardImplementation(self, inbuf, outbuf):
    outbuf[:] = inbuf @ self.weights + self.bias
  def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
    self.weights -= inbuf[:, np.newaxis] @ outerr[np.newaxis, :]
    self.bias -= outerr
    inerr[:] = outerr @ self.weights.T

class EmbeddingLayer(LinearLayer):
  def __init__(self, vocab_size, embedding_dim):
     super().__init__(embedding_dim)
     self.embeddings = np.random.randn(vocab_size, embedding_dim)
  def _forwardImplementation(self, inbuf, outbuf):
     self.token_idx = np.argmax(inbuf)
     outbuf[:] = self.embeddings[self.token_idx]
  def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
    gradient = np.zeros_like(self.embeddings)
    gradient[self.token_idx] = outerr
    self.embeddings -= gradient
    inerr[:] = self.embeddings.T @ outerr

class GeLULayer(NeuronLayer):
	def __init__(self, dim):
		super().__init__(dim, dim)
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
def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)
#
class MultiHeadSelfAttention(NeuronLayer):
    def __init__(self, indim, outdim, num_heads):
        super().__init__(indim, outdim)
        self.num_heads = num_heads
        self.depth = indim // num_heads
        self.W_q = np.random.rand(indim, indim)
        self.W_k = np.random.rand(indim, indim)
        self.W_v = np.random.rand(indim, indim)
        self.W_o = np.random.rand(indim, outdim)

    def scaled_dot_product_attention(self, Q, K, V):
        matmul_qk = np.dot(Q, K.T)
        d_k = Q.shape[-1]
        scaled_attention_logits = matmul_qk / np.sqrt(d_k)
        attention_weights = softmax(scaled_attention_logits, axis=-1)
        return np.dot(attention_weights, V)
    def _forwardImplementation(self, inbuf, outbuf):
        if len(inbuf.shape) == 1:
          inbuf = inbuf[np.newaxis, :]
        Q = np.dot(inbuf, self.W_q)
        K = np.dot(inbuf, self.W_k)
        V = np.dot(inbuf, self.W_v)
        Q = np.split(Q, self.num_heads, axis=1)
        K = np.split(K, self.num_heads, axis=1)
        V = np.split(V, self.num_heads, axis=1)
        
        attention_heads = []
        for i in range(self.num_heads):
            attention_head = self.scaled_dot_product_attention(Q[i], K[i], V[i])
            attention_heads.append(attention_head)
        
        # Concatenate attention heads and pass through the final linear layer
        concatenated_heads = np.concatenate(attention_heads, axis=1)
        outbuf[:] = np.dot(concatenated_heads, self.W_o)


    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        d_concat_heads = np.dot(outerr, self.W_o.T)
        d_attention_heads = np.split(d_concat_heads, self.num_heads, axis=1)
        dQ_total, dK_total, dV_total = 0, 0, 0
        for i in range(self.num_heads):
            d_out = d_attention_heads[i]
            dV = np.dot(self.attention_weights[i].T, d_out)
            d_attention_weights = np.dot(d_out, self.V[i].T)
            dQK = d_attention_weights * (1 - self.attention_weights[i]) * self.attention_weights[i]
            dQ = np.dot(dQK, self.K[i])
            dK = np.dot(dQK, self.Q[i])
            dQ_total += dQ
            dK_total += dK
            dV_total += dV
        inerr[:] = np.dot(dQ_total, self.W_q.T) + np.dot(dK_total, self.W_k.T) + np.dot(dV_total, self.W_v.T)
        self.W_q -= np.dot(inbuf.T, dQ_total)
        self.W_k -= np.dot(inbuf.T, dK_total)
        self.W_v -= np.dot(inbuf.T, dV_total)
        self.W_o -= np.dot(self.concatenated_heads.T, outerr)
#
class LayerNorm(NeuronLayer):
	def __init__(self, size, eps=1e-6):
		super().__init__(size, size)
		self.gamma = np.ones(size)
		self.beta = np.zeros(size)
		self.eps = eps
	def _forwardImplementation(self, inbuf, outbuf):
		mean = np.mean(inbuf)
		std = np.std(inbuf)
		outbuf[:] = self.gamma * (inbuf - mean) / (std + self.eps) + self.beta
	def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
		N = inbuf.size
		dbeta = np.sum(outerr)
		dgamma = np.sum((inbuf - np.mean(inbuf)) / (np.std(inbuf) + self.eps) * outerr)
		dinbuf = self.gamma / (N * np.std(inbuf) + self.eps) * (N * outerr - np.sum(outerr) - (inbuf - np.mean(inbuf)) / (np.std(inbuf) + self.eps) * np.sum(outerr * (inbuf - np.mean(inbuf))))
		inerr[:] = dinbuf


VOCAB_SIZE = 50257
D_MODEL = 128
NUM_BLOCKS = 96
NUM_HEADS = 64
FFN_DIM = 128
net = FeedForwardNetwork()
inLayer = LinearLayer(VOCAB_SIZE)
net.addInputModule(inLayer)
embedding = EmbeddingLayer(VOCAB_SIZE, D_MODEL)
net.addModule(embedding)
net.addConnection(FullConnection(inLayer, embedding))
attention = MultiHeadSelfAttention(D_MODEL, D_MODEL, NUM_HEADS)
net.addModule(attention)
net.addConnection(FullConnection(embedding, attention))
prev_layer = attention
for _ in range(NUM_BLOCKS):
    norm1 = LayerNorm(D_MODEL)
    net.addModule(norm1)
    net.addConnection(FullConnection(prev_layer, norm1))
    ffn1 = LinearLayer(D_MODEL, FFN_DIM)
    net.addModule(ffn1)
    net.addConnection(FullConnection(norm1, ffn1))
    gelu = GeLULayer(FFN_DIM)
    net.addModule(gelu)
    net.addConnection(FullConnection(ffn1, gelu))
    ffn2 = LinearLayer(FFN_DIM, D_MODEL)
    net.addModule(ffn2)
    net.addConnection(FullConnection(gelu, ffn2))
    norm2 = LayerNorm(D_MODEL)
    net.addModule(norm2)
    net.addConnection(FullConnection(ffn2, norm2))    
    prev_layer = norm2
outLayer = SoftmaxLayer(VOCAB_SIZE)
net.addOutputModule(outLayer)
net.addConnection(FullConnection(prev_layer, outLayer))
net.sortModules()
#print(net.activate(tuple([0]*50257)))

# Save the network to a file
with open('network.xml', 'wb') as f:
	pickle.dump(net, f)

"""
This is for those getting into AI development.

If you're looking for a place to start, here is a good place as any.

Do note that this isn't easy to do development as it is AI, token parsing, and mapping data plus testing for correctness.


"""
