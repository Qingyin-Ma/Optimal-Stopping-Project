% =========================================================================
%                       The Firm Entry Problem
% 
% Fajgelbaum, Schaal, and Taschereau-Dumouchel (2015)
% 
% x = theta + eps_x, eps_x ~ N(0, gam_x)
% where:
%       x: output, not observed when the firm makes decisions at the
%          beginning of each period
%
% y = theta + eps_y, eps_y ~ N(0, gam_y)
% where:
%       y: a public signal, observed after the firm makes decision of
%          whether to invest or not
%
%
% The Bayesian learning process:
% theta ~ N(mu, gam): prior
% theta|y ~ N(mu', gam') : posterior after observing public signal y
% where:
%        gam' = 1/(1/gam + 1/gam_y)
%        mu' = gam'*(mu/gam + y/gam_y)
%
%
% u(x) = (1/a) * (1 - exp(-a*x)) : the return to the firm when output is x
% where: a is the coefficient of absolute risk aversion.
%
%
% f ~ h(f) = LN(mu_f, gam_f) : the entry cost / the investment cost
% 
%
% The value function is
% V(f,mu,gam) = max{ E[u(x)|mu,gam]-f, beta*E[V(f',mu',gam')|mu,gam] }
% where: 
%       E[u(x)|mu,gam]-f : the value of entering the market and investing
%       beta*E[V(f',mu',gam')|mu,gam] : the expected value of waiting 
%
% =========================================================================


clear all; close all; clc;

global beta gam_x gam_y a gridpoints mu_f gam_f gridsize mugrid gamgrid gamsize musize N;


%% Parameters

beta = 0.95;   % the discount factor
gam_x = 1.0;   % the variance of eps_x, eps_x ~ N(0,gam_x)
gam_y = 1.5;   % the variance of eps_y, eps_y ~ N(0,gam_y)
a = 0.6;  % 0.99; 1.5; 2; 0.25; 1.1; 0.65; 0.75; 0.6; 0.55, 0.58     % the coefficient of absolute risk aversion
mu_f = 0.05;    % mean of the cost distribution f ~ h(f) = LN(mu_f, gam_f)
gam_f = 0.0005; %0.005;  % variance of the cost distribution f ~ h(f) = LN(mu_f, gam_f)
N = 50000;     % number of MC samples used to compute the integration


%% Make grid for mu
% the grid is scaled to be more dense when the absolute value of mu is
% smaller

mumin = 1e-4;
mumax = 50;
musize = 100; % make it an even number, as the state space of mu is the 
              % whole real number system. We create musize/2 positive
              % grids and musize/2 negative grids.
muscale = 4;
% mugrid = makegrid(mumin, mumax, musize, muscale);
mugrid_positive = makegrid(mumin, mumax, musize / 2, muscale);
mugrid_negative = -mugrid_positive;
mugrid_negative = sort(mugrid_negative);
mugrid = [mugrid_negative; mugrid_positive];


%% Make grid for gamma
% the grid is scaled to be more dense when the absolute value of gamma is
% smaller

gammin = 1e-4; 
gammax = 25;
gamsize = 50;
gamscale = 4;
gamgrid = makegrid(gammin, gammax, gamsize, gamscale);


%% Recombine and make grid for reservation rule operator iteration

[X Y] = meshgrid(mugrid, gamgrid);
gridsize = musize * gamsize;
gridX = reshape(X, gridsize, 1);
gridY = reshape(Y, gridsize, 1);
gridpoints = [gridX, gridY];


%% Iteration based on the reservation rule operator to find the fixed point

kappa = kappafunction(gridX, gridY);  % the kappa function (the weight
                                      % function) evaluated at the grids
kappa = reshape(kappa, gamsize, musize);
phi0 = ones(gamsize, musize);   % initial guess of the fixed point (the
                                % reservation cost)

Tol = 1e-4;    % stopping criterion
err = Tol + 1; count = 1; maxiter = 1000;
while err > Tol
    phi1 = ReservationRuleOperator(phi0);
    err = max(max(abs(phi1 - phi0) ./ kappa));  % update err
    format long;
    disp([err count]);
    phi0 = phi1;
    count = count + 1;
end;

if count == maxiter;
    disp('maximum iteration reached');
end;


%% Recover the reservation cost based on the fixed point
f_bar = (1 / a) * (1 - exp(-a * X + (a^2) * (Y + gam_x) / 2)) - phi1;

%% Perceived probability of investing
p_bar = cdf('logn', f_bar, mu_f, sqrt(gam_f));

%% Plot the reservation cost
subplot(1,2,1); mesh(X, Y, f_bar);
xlabel('\mu');
ylabel('\gamma');
% xlim([-10 30]);
% ylim([-1 1]);
title('Reservation Cost');

subplot(1,2,2); mesh(X(:,25:90), Y(:,25:90), p_bar(:,25:90));
xlabel('\mu');
ylabel('\gamma');
% xlim([-10 30]);
% ylim([0 25]);
title('Perceived Probability of Investment');

