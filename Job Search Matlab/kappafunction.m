% ========================================================================
% The kappa function / weight function 
% kappa(mu, gam) = exp(-a*mu) * exp( (a^2)*(gam+gam_w)/2 ) + 1
% 
% This funtion is used for constructing the new complete metric space
% (b_kappa \Theta, rho_kappa).
%
% ========================================================================

function result = kappafunction(mu, gam)

global a gam_w;

term1 = exp(-a * mu);
term2 = exp((a ^ 2) * (gam + gam_w) / 2);

result = term1 .* term2 + 1;


