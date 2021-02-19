require "json"
require "numo/narray"

array_table = nil

File.open("sample_weight.json", "r") do |f|
  content = f.read
  array_table = JSON.parse(content)
end

new_table = {}

array_table.keys.each do |k|
  new_table[k] = Numo::DFloat.cast(array_table[k])
end

File.binwrite("sample_weight.dump", Marshal.dump(new_table))
