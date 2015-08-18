% =========================================================================
% The reward function
% 
% u(w) = (1 / a) * (1 - exp(-a * w))
%
% Note: the worker obtains utility u(w) each period if he/she acceptes the 
% wage offer w.
%
% =========================================================================


function result = u(w)

global a;

result = (1 - exp(-a * w)) / a;