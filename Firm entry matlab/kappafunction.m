% ========================================================================
% The kappa function / weight function 
% kappa(mu, gam) = exp(-a*mu + (a^2)*(gam + gam_x)/2) + 1
% 
% This funtion is used for constructing the new complete metric space
% (b_kappa \Theta, rho_kappa).
%
% ========================================================================

function result = kappafunction(mu, gam)

global a gam_x;

result = exp(-a * mu + (a^2) * (gam + gam_x) / 2) + 1;




