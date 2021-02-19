require "numo/narray"
NM = Numo::NMath

def softmax(x)
  x = x - x.max
  return NM.exp(x) / NM.exp(x).sum
end
