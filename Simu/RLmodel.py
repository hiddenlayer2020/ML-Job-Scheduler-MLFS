import torch.nn as nn
import torch.nn.functional as F
import torch
import math
class RLnetwork(object):
    def __init__(self, numInputs, numOutputs):
        self.input_dim = numInputs
        self.output_dim = numOutputs
        self.weight = torch.Tensor(numInputs, numOutputs).normal_(0, 0.01)
        self.weight_grads = torch.Tensor(numInputs, numOutputs)
        self.bias = torch.Tensor(numOutputs).zero_()
        self.bias_grads = torch.Tensor(numOutputs)

    def forward(self, state_vector): #input: 1 x n. weight: m x n. output: weight x input.t() = m x 1

        return torch.matmul(state_vector, self.weight) + self.bias

    def backward(self, state_vector, vt, learning_rate):
      log_grad = 1/(torch.matmul(state_vector, self.weight) + self.bias)
      sv = torch.Tensor(self.input_dim,1)
      lg = torch.Tensor(1,self.output_dim)
      sv = state_vector.unsqueeze_(-1)
      lg[0][:] = log_grad
      weight_grads = torch.matmul(sv,lg) 
      self.weight_grads += learning_rate * weight_grads * vt
      self.bias_grads += learning_rate * log_grad * vt
      return self.weight_grads, self.bias_grads
    
    def zero_grad(self):
      self.weight_grads = 0
      self.bias_grads = 0
    
    def update_grads(self):
      self.weight -= self.weight_grads
      self.bias -= self.bias_grads