% ========================================================================
% The reservation rule (wage) operator
% 
% Qphi = c*(1-beta) + beta*integral(max{u(w'),phi(mu',gam')} *f(w'|mu,gam))dw'
% where: 
%        u(w) = (1/a) * (1 - exp(-a*w))
%        f(w'|mu, gam) = N(mu, gam + gam_w)
%        gam' = 1/(1/gam + 1/gam_w)
%        mu' = gam'*(mu/gam + w'/gam_w)
% 
% The operator Q is a well-defined contraction mapping from the complete
% metric space (b_kappa \Theta, rho_kappa) into itself.
% where b_kappa \Theta is the reweighted space by the weight function
% kappa. 
%
% ========================================================================

function phi1 = ReservationRuleOperator(phi0)

global gridsize N gridpoints gam_w mugrid gamgrid c beta gamsize musize; 

phi1 = zeros(gridsize, 1);
draws = randn(N, 1);  % initial Monte Carlo samples

for i = 1:gridsize;
    
    gridi = gridpoints(i, :);
    mu = gridi(1); gam = gridi(2);
    gam_prime = 1 / (1 / gam + 1 / gam_w); % scalar
    draws_w = mu + sqrt(gam + gam_w) * draws;   % sample w' from f(w'|mu,gam)
                                                % N * 1 vector
    b1 = gam_prime / gam;
    b2 = 1 - b1;
    mu_prime = b1 * mu + b2 * draws_w;  % N * 1 vector
    gam_prime = gam_prime * ones(N, 1); % make gam' an N * 1 vector
    
    % interpolate and calculate phi(mu',gam')
    % the evaluation of the function outside the grid is set to its value 
    % at the closest grid point
    phi_interp = interpne(mugrid, gamgrid, phi0, mu_prime, gam_prime, 'linear');
    
    u_wprime = u(draws_w);  % the reward function
    
    % compute the operator (with integral in it) by the Monte Carlo method
    phi1(i) = c * (1 - beta) + beta * mean(max(u_wprime, phi_interp));
    
end;

phi1 = reshape(phi1, gamsize, musize);

