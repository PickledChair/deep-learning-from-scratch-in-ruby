require "numo/narray"

def AND(x1, x2)
  x = Numo::DFloat[x1, x2]
  w = Numo::DFloat[0.5, 0.5]
  b = -0.7
  tmp = (w*x).sum + b
  return tmp <= 0 ? 0 : 1
end

def NAND(x1, x2)
  x = Numo::DFloat[x1, x2]
  w = Numo::DFloat[-0.5, -0.5]   # 重みとバイアスだけが AND と違う！
  b = 0.7
  tmp = (w*x).sum + b
  return tmp <= 0 ? 0 : 1
end

def OR(x1, x2)
  x = Numo::DFloat[x1, x2]
  w = Numo::DFloat[0.5, 0.5]   # 重みとバイアスだけが AND と違う！
  b = -0.2
  tmp = (w*x).sum + b
  return tmp <= 0 ? 0 : 1
end

def XOR(x1, x2)
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y
end

