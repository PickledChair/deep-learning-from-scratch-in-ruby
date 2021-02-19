require "numo/narray"
require "charty"

def step_function(x)
  return Numo::Int32.cast(x > 0)
end

x = Numo::DFloat.new(101).seq(-5, 0.1)
y = step_function(x)

charty = Charty::Plotter.new(:pyplot)
curve = charty.curve do
  series x, y, label: "step function"
  xlabel "x"
  ylabel "y"
end
curve.render("step_function.png")
