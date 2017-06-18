So, basically we need to implement _most_ of what's in torch7-libv2 here.
Let's start off with this list:

	* AddGaussian.lua
	* BinaryStochastic.lua
	* HighwayLayer.lua
	* Nullable.lua
	* SpatialDropoutX21.lua

Ordered:
	* 2017-06-17: HighwayLayer.lua
	* 2017-06-17: BinaryStochastic.lua
	* 2017-06-17: AddGaussian.lua
	* 2017-06-18: Nullable.lua -- Perhaps this needs more rigorous testing tho

Not needed:
	* 2017-06-17: SpatialDropoutX21.lua
		So, the reason I needed SpatialDropoutX21 was because
		luatorch's SpatialDropout didn't premultiply by (1/(1-p))
		pytorch's nn.Dropout2d _does_ premultiply... So, this is not
		needed anymore.
	* 2017-06-18: Random.lua -- Let's see how far we can get without implementing these.
	* 2017-06-18: Zero.lua
