% =========================================================================
% The reservation rule (cost) operator
%
% Qphi(mu,gam) = beta*integral[ max{reward, phi(mu',gam')} * h(f')*l(y|mu,gam) ]df'dy
% where: 
%        f ~ h(f) = LN(mu_f, gam_f) : the entry cost 
%        gam' = 1/(1/gam + 1/gam_y)
%        mu' = gam'*(mu/gam + y/gam_y)
%        reward = (1/a) - (1/a)*exp[-a*mu' + (a^2)*(gam'+gam_x)/2] - f'
%        l(y|mu,gam) = N(mu, gam + gam_y)
%        
% The operator Q is a well-defined contraction mapping from the complete
% metric space (b_kappa \Theta, rho_kappa) to itself.
% where b_kappa \Theta is the reweighted space by the weight function
% kappa. 
%
% ========================================================================

function phi1 = ReservationRuleOperator(phi0)

global gridsize gridpoints gam_x gam_y mu_f gam_f a mugrid gamgrid beta gamsize musize N draws;

phi1 = zeros(gridsize, 1);

for i = 1:gridsize;
    
    gridi = gridpoints(i, :);
    mu = gridi(1); gam = gridi(2);
    gam_prime = 1 / (1 / gam + 1 / gam_y);  % scalar
    draws_y = mu + sqrt(gam + gam_y) * draws(:,1);     % sample x from l(x|mu,gam)
                                                       % N * 1 vector
    
    b1 = gam_prime / gam;
    b2 = 1 - b1;
    mu_prime = b1 * mu + b2 * draws_y;   % N * 1 vector
    gam_prime = gam_prime * ones(N, 1);  % make gam' an N * 1 vector
    
    f_prime = exp(mu_f + sqrt(gam_f) * draws(:,2));  % sample f from h(f)

    % interpolate and calculate phi(mu',gam')
    % the evaluation of the function outside the grid is set to its value 
    % at the closest grid point
    phi_interp = interpne(mugrid, gamgrid, phi0, mu_prime, gam_prime, 'linear');  
    
    reward = 1 / a - (1 / a) * exp(-a * mu_prime + (a^2) * (gam_prime + gam_x) / 2) - f_prime;
    
    % compute the operator (with integral in it) by the Monte Carlo method
    phi1(i) = beta * mean(max(reward, phi_interp));
    
end;

phi1 = reshape(phi1, gamsize, musize);

    
    
    

