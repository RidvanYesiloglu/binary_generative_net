
clear all; close all;

%%
global nbits
global npar
global popsize
global nelite
global npairs
global start_indices
global pairs
global pairs_indices
global weight
global iter_plotstep
global xlabel_str
global ylabel_str
global tag_on_str
global title_str

nbits = 32;  % bits per binary sequence
npar = 3;     % number of parameters (must be at least 2)
popsize = 400;  % number members in population
nelite = 200;     % numer elite members to refit population
obj_str = 'max_AC_CC_sqr';   % 'max_AC_CC_sqr' or 'equalweight_AC_CC_sqr'
max_iter = 5000; % maximum number of iterations
plot_iter = true;
print_final = true; % whether to print final results
print_final_seq = false; % whether to print final sequence
verbose = false;    % whether to print intermediate, iterative results


% Specify x and y labels
iter_plotstep = 20;
xlabel_str = 'Iteration number';
ylabel_str = 'Objective function value';
tag_on_str = [num2str(npar), ' seq, ', num2str(nbits), ...
                ' bits each (pop ', num2str(popsize), ')'];
title_str = ['Evolution of ', tag_on_str];


%% Set up additional global variables
npairs = nchoosek(npar, 2);

% Get all starting integers for sequences and pairs of sequences
start_indices = 1:nbits:(nbits*(npar - 1) + 1);
pairs = combnk(start_indices, 2);
pairs_indices = combnk(1:popsize, 2);

% Specify maximum cost to propagte (for ef = chooseParetoFront_minMaxCorr)
weight = [0.5, 0.5];  % Weight vector of cost
assert (sum(weight) == 1);


%% Run CE method
if obj_str == "max_AC_CC_sqr"
    ff = @ff_max_mean_sqr_auto_and_cross_corr;
elseif obj_str == "equalweight_AC_CC_sqr"
    ff = @ff_weighted_mean_sqr_auto_and_cross_corr;
else
    error(['Specified objective function string does not have ', ...
        'corresponding function defined: ', obj_str]);
end

disp(['Running cross entropy method for objective: ', obj_str]);
pvalues = CE_method(nbits, npar, popsize, ff, ...
    nelite, max_iter, plot_iter, print_final, print_final_seq, verbose);