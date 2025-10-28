#const max_bins = 2.
#const bin_bit_width = 3.

variable(v0).
variable(v1).
variable(v2).
variable(v3).

bit_width(v0, 1).
bit_width(v1, 2).
bit_width(v2, 1).
bit_width(v3, 2).

operator(o0).
operator(o1).
operator(o2).
operator(o3).

effect(o0, v0).
effect(o0, v1).
effect(o1, v1).
effect(o1, v2).
effect(o2, v0).
effect(o2, v2).
effect(o3, v3).
