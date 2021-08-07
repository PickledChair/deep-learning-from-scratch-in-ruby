require "numo/narray"
require "numo/gnuplot"
require "../common/gradient.rb"

def function_2(x)
  if x.ndim == 1
    return (x**2).sum
  else
    return (x**2).sum(axis: 1)
  end
end

def tangent_line(f, x)
  d = GRAD.numerical_gradient(f, x)
  p d
  y = f.call(x) - d*x
  return proc { |t| d*t + y }
end

if $0 == __FILE__
  x0 = Numo::DFloat.new(18).seq(-2.0, 0.25)
  x1 = Numo::DFloat.new(18).seq(-2.0, 0.25)

  x = (x0.expand_dims(0) * Numo::DFloat.ones(x1.size, 1)).flatten
  y = (x1.expand_dims(1) * Numo::DFloat.ones(1, x0.size)).flatten

  grad = GRAD.numerical_gradient_2d(method(:function_2), Numo::DFloat.cast([x, y]).transpose).transpose
  File.open("gradient_2d.dat", "w") do |text|
    grad.shape[1].times { |idx|
      text.puts(
        x[idx].to_s + " " + y[idx].to_s + " " \
        + (-grad[0, idx]*0.05).to_s + " " + (-grad[1, idx]*0.05).to_s
      )
    }
  end

  Numo.gnuplot do
    set xrange: -2..2
    set yrange: -2..2
    set xlabel: "x0"
    set ylabel: "x1"
    plot "'gradient_2d.dat'", w: "vectors"
  end
  _ = gets
end
