require "numo/narray"
NM = Numo::NMath

def cross_entropy_error(y, t)
  delta = 1e-7
  return -(t * NM.log(y + delta)).sum
end

if $0 == __FILE__
  # 「2」を正解とする
  t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

  # 例1:「2」の確率が最も高い場合（0.6）
  y0 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
  p cross_entropy_error(Numo::DFloat.cast(y0), Numo::DFloat.cast(t))

  # 例2:「7」の確率が最も高い場合（0.6）
  y1 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
  p cross_entropy_error(Numo::DFloat.cast(y1), Numo::DFloat.cast(t))
end
