% Find random combinations of codes

% stanfordRed = [140, 21, 21] / 255;
% stanfordGreen = [29, 121, 108] / 255;

%% Setup
global nbits; 
global npar;
global popsize;
global npairs;
global start_indices;
global pairs;

% Specify objective
% obj_str = 'equalweight_AC_CC_sqr';
obj_str = 'max_AC_CC_sqr';

% Specify taps / register initialization
nbits = 521;

% Specify number of sequences desired
npar = 31;
popsize = 10000;

npairs = nchoosek(npar, 2);
start_indices = 1:nbits:(nbits*(npar - 1) + 1);
pairs = combnk(start_indices, 2);
% pairs_indices = combnk(1:popsize, 2);
    
completeWeilCodes = getWeilCodes(nbits);
completeWeilCodes = (-1*completeWeilCodes + 1)/2;
nCodes = (nbits-1)/2;

% For each random family
results = nan*ones(popsize,2);

% Generate random family
pop = nan*ones(popsize,nbits*npar);
for i = 1:popsize
   % Generate random collection (sampling without replacement)
   currFamily = datasample(1:nCodes, npar, 'Replace', false);
   pop(i, :) = reshape(completeWeilCodes(currFamily,:)', [1,npar*nbits]);
end


% Get fitness function 
if obj_str == "equalweight_AC_CC_sqr"
    [max_obj, auto_obj, cross_obj] ...
        = ff_equalweight_mean_sqr_auto_and_cross_corr(pop);
elseif obj_str == "max_AC_CC_sqr"
    [max_obj, auto_obj, cross_obj] ...
        = ff_max_mean_sqr_auto_and_cross_corr(pop);
else
    error(['Specified objective function string does not have ', ...
        'corresponding function defined: ', obj_str]);
end

[min_cost, min_cost_i] = min(max_obj);
disp(' ');
disp(['Performing objective: ', num2str(obj_str)]);
disp(['nbits = ', num2str(nbits), ', npar = ', num2str(npar), ...
    ', num samps = ', num2str(popsize)]);
disp(' ');
disp(['Best Objective: ', num2str( max_obj(min_cost_i)) ]);
disp(['          auto comp: ', num2str( auto_obj(min_cost_i) )]);
disp(['          cross comp: ', num2str( cross_obj(min_cost_i) )]);
disp(['Log of Best Normalized Objective: ', num2str( log((1/nbits)^2 * max_obj(min_cost_i)) )]);
disp(['          auto comp: ', num2str( log((1/nbits)^2 * auto_obj(min_cost_i)) )]);
disp(['          cross comp: ', num2str( log((1/nbits)^2 *cross_obj(min_cost_i)) )]);
disp(' ');
