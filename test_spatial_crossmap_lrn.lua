#! /usr/local/torch/install/bin/th

package.cpath = string.format("./simile12_common/?.so;%s/src/torch7-libv2/?.so", os.getenv("HOME")) .. ";" .. package.cpath
package.path = string.format("./simile12_common/?.lua;%s/src/torch7-libv2/?.lua", os.getenv("HOME")) .. ";" .. package.path
local random_source = require "random_source"
require "nn"

function fprintf(fh, fmt, ...)
	fh:write(string.format(fmt, unpack({...})))
end

function simple_json(fh, il, array)
	local indentstr = string.rep("\t", il)
	if array:dim() == 1 then
		fprintf(fh, "%s[", indentstr)
		for i = 1, array:size(1) do
			if i > 1 then fprintf(fh, ", ") end
			fprintf(fh, "%.7f", array[i])
		end
		fprintf(fh, "]")
	else
		fprintf(fh, "%s[", indentstr)
		for i = 1, array:size(1) do
			if i > 1 then fprintf(fh, ", ") end
			fprintf(fh, "\n")
			simple_json(fh, (1+il), array[i])
		end
		fprintf(fh, "\n%s]", indentstr)
	end
end

function main()
	local size = 5
	local alpha = 0.0001
	local beta = 0.75
	local k = 1

	local minibatches = 10
	local features = 10
	local height = 10
	local width = 10

	local grab = random_source(12345, 16777216)

	local inp = grab(minibatches*features*height*width):reshape(minibatches, features, height, width)
	local lua_scmlrn = nn.SpatialCrossMapLRN(size, alpha, beta, k):float()
	local lua_out = lua_scmlrn:forward(inp)

	local fh = io.open("/tmp/lua.json", "w")
	simple_json(fh, 0, lua_out)
	fh:close()
end

main()
