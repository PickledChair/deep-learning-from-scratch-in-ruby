require "numo/narray"
NM = Numo::NMath

def identity_function(x)
  return x
end

def step_function(x)
  return Numo::Int32.cast(x > 0)
end

def sigmoid(x)
  return 1 / (1 + NM.exp(-x))
end

def relu(x)
  return x.clip(0, nil)
end

def softmax(x)
  x = x - x.max
  return NM.exp(x) / NM.exp(x).sum
end
