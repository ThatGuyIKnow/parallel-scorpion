%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Necessary constants:
% - max_bins defines the number of available bins. Perform exponential search 1,2,4,8,... followed by interval search to lowest value.
% Then try to increase the bound to see how the objective value changes to find a sweet spot
% - bin_bit_width defines the number of bits available in each bin.
% Necessary facts:
% - variable(V) defines that V is a variable
% - bit_width(V, W) defines the bit width W of variable V
% - operator(O) defines that O is an operator
% - effect(O, V) defines that operator O has an effect on variable V
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Minimize number of bins used by each operator
#minimize { 1@2, touched_node(O, N) : touched_node(O, N) }.
#minimize { 1@1 : touched_node(N)}.

% Generate available bins .
bin(0..max_bins-1).

% Generate variable to bin assignment
1 { throw(V, B) : bin(B) } 1 :- variable(V).

% Define used nodes by operator O
touched_node(O, B / 2) :- operator(O), effect(O,V), throw(V,B).

% Define used node by any operator O
touched_node(B / 2) :- variable(V), bin(B), throw(V, B).

% Ensure that variables fit into bin
:- bin(B), #sum { W,V : throw(V,B), bit_width(V,W) } > bin_bit_width.

#show throw/2.