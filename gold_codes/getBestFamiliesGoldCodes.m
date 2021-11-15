
close all; clear all;
%%
% Find random combinations of codes
stanfordRed = [140, 21, 21] / 255;
stanfordGreen = [29, 121, 108] / 255;

%% Setup
global popsize
global nbits
global npar
global start_indices
global pairs
global npairs
global pop

% For length 63, 127, and 511 got preferred pairs for generating Gold codes
% from: https://www.gaussianwaves.com/2015/06/gold-code-generator/
% For length 31, got pair from GPS textbook
% For length 1023, got pair from GPS ICD

% Specify taps / register initialization
% nbits = 31;
% tap1 = [1, 0, 1, 0, 0];
% tap2 = [1, 0, 1, 1, 1];
% init_reg = [1, 0, 0, 0, 0]; 

% For 63-length codes
% nbits = 63;
% tap1 = [1, 0, 0, 0, 0, 1];
% tap2 = [1, 1, 0, 0, 1, 1];
% init_reg = [1, 1, 1, 1, 1, 1];

% For 127-length codes
% nbits = 127;
% tap1 = [1, 0, 0, 0, 1, 1, 1];
% tap2 = [1, 0, 0, 0, 1, 0, 0];
% init_reg = [1, 1, 1, 1, 1, 1, 1];

% For 511-length codes
% nbits = 511;
% tap1 = [1, 0, 0, 1, 0, 1, 1, 0, 0];
% tap2 = [1, 0, 0, 0, 0, 1, 0, 0, 0];
% init_reg = [1, 1, 1, 1, 1, 1, 1, 1, 1];

% For 1023-length codes
nbits = 1023;
tap1 = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0];
tap2 = [1, 1, 1, 0, 1, 0, 0, 1, 1, 0];
init_reg = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

% Specify objective
% obj_str = 'equalweight_AC_CC_sqr';
obj_str = 'max_AC_CC_sqr';

% Specify number of sequences desired
npar = 10;
popsize = 10000;

npairs = nchoosek(npar, 2);
start_indices = 1:nbits:(nbits*(npar - 1) + 1);
pairs = combnk(start_indices, 2);
    
%%


codelength = 2^length(tap1)-1;
delays = 1:codelength;
completeGoldCodes = nan*ones(codelength+2,codelength);
for delay = delays
   code = goldCode(tap1,tap2,init_reg,delay);
   completeGoldCodes(delay,:) = code;
end
completeGoldCodes(codelength+1,:) = return_LFSR_sequence(tap1, init_reg);
completeGoldCodes(codelength+2,:) = return_LFSR_sequence(tap2, init_reg);

% For each random family
results = nan*ones(popsize,2);

% Generate random family
pop = nan*ones(popsize,codelength*npar);
for i = 1:popsize
   % Generate random collection (sampling without replacement)
   currFamily = datasample(1:(codelength+2), npar, 'Replace', false);
   pop(i, :) = reshape(completeGoldCodes(currFamily,:)', [1,npar*codelength]);
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
