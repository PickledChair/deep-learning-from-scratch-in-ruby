require "numo/narray"
require "./sigmoid_func"
require "./identity_func"

def init_network
  network = {}
  network[:W1] = Numo::DFloat[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
  network[:b1] = Numo::DFloat[0.1, 0.2, 0.3]
  network[:W2] = Numo::DFloat[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]
  network[:b2] = Numo::DFloat[0.1, 0.2]
  network[:W3] = Numo::DFloat[[0.1, 0.3], [0.2, 0.4]]
  network[:b3] = Numo::DFloat[0.1, 0.2]

  return network
end

def forward(network, x)
  w1, w2, w3 = network[:W1], network[:W2], network[:W3]
  b1, b2, b3 = network[:b1], network[:b2], network[:b3]

  a1 = x.dot(w1) + b1
  z1 = sigmoid(a1)
  a2 = z1.dot(w2) + b2
  z2 = sigmoid(a2)
  a3 = z2.dot(w3) + b3
  y = identity_function(a3)

  return y
end

if $0 == __FILE__
  network = init_network
  x = Numo::DFloat[1.0, 0.5]
  y = forward(network, x)
  p y
end
