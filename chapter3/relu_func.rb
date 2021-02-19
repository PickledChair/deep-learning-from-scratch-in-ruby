require "numo/narray"
require "charty"
NM = Numo::NMath

def relu(x)
  return x.clip(0, nil)
end


if $0 == __FILE__
  x = Numo::DFloat.new(101).seq(-5, 0.1)
  y = relu(x)

  charty = Charty::Plotter.new(:pyplot)
  curve = charty.curve do
    series x, y, label: "relu function"
    xlabel "x"
    ylabel "y"
  end

  curve.render("relu_function.png")
end
