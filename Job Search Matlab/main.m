% ========================================================================
% The generalized job search model
%
% The value function:
% V(w,mu,gam) = max{ u(w)/(1-beta), c+beta*E[V(w',mu',gam')|mu,gam] }
% 
% The Bayesian updating process: 
% w = theta + eps_w, eps_w ~ N(0, gam_w)
% prior: theta ~ N(mu, gam)
% posterior: theta|w' ~ N(mu', gam')
% where: gam' = 1/(1/gam + 1/gam_w) 
%        mu' = gam'*(mu/gam + w'/gam_w)
%
% Agents have constant absolute risk aversion:
% u(w) = (1/a) * (1 - exp(-a*w)) 
%
% ========================================================================


close all; clear all; clc;

global beta c a gam_w N musize mugrid gamsize gamgrid X Y gridsize gridpoints draws; 

%% Parameters

beta = 0.95;   % discount factor
c = 0.05;      % unemployment compensation
a = 2.5;       % coefficient of relative risk aversion
gam_w = 0.005; % the variance of eps_w
N = 1000;      % the number of MC draws to calculate the expectation
               % in the reservation rule operator
               
               
%% Create grid for mu
% the grid is scaled to be more dense when the absolute value of mu is
% smaller

mumax = 50;
mumin = 1e-3;
musize = 100; % make it an even number, as the state space of mu is the 
              % whole real number system. We create musize/2 positive
              % grids and musize/2 negative grids.
muscale = 4;
mugrid_positive = makegrid(mumin, mumax, musize / 2, muscale);
mugrid_negative = -mugrid_positive;
mugrid_negative = sort(mugrid_negative);
mugrid = [mugrid_negative; mugrid_positive];


%% Create grid for gamma
% the grid is scaled to be more dense when the absolute value of gamma is
% smaller

gammax = 25;
gammin = 1e-4;
gamsize = 50;
gamscale = 4;
gamgrid = makegrid(gammin, gammax, gamsize, gamscale);


%% Recombine and create grid for reservation rule operator iteration

[X Y] = meshgrid(mugrid, gamgrid);
gridsize = musize * gamsize;
gridX = reshape(X, gridsize, 1);
gridY = reshape(Y, gridsize, 1);
gridpoints = [gridX gridY];


%% Iteration based on the reservation rule operator to find the fixed point

% The fixed point is the reservation utility, which is a function of the 
% reservation wage w_bar. We can recover w_bar based on the reservation
% utility.

kappa = kappafunction(gridX, gridY);  % the kappa function (the weight
                                      % function) evaluated at the grids
kappa = reshape(kappa, gamsize, musize); 

draws = randn(N, 1);  % initial Monte Carlo samples
phi0 = ones(gamsize, musize);   % initial guess of the fixed point (the
                                % reservation wage)

Tol = 1e-5;   % stopping criterion
err = Tol + 1; count = 1; maxiter = 1000;
while err > Tol
    phi1 = ReservationRuleOperator(phi0);
    err = max(max(abs(phi1 - phi0) ./ kappa)); % update err
    format long;
    disp([err count]);
    phi0 = phi1;
    count = count + 1;
end;

if count == maxiter;
    disp('maximum iteration reached ...');
end;


%% Plot the reservation utility

% The reservation wage w_bar can be recovered from the reservation utility.

% mesh(X, Y, phi1);
% xlabel('\mu');
% ylabel('\gamma');
% xlim([-20 20]);
% title('Reservation Utility');

mesh(X(:,15:85), Y(:,15:85), phi1(:,15:85));
xlabel('\mu');
ylabel('\gamma');
title('Reservation Utility');



