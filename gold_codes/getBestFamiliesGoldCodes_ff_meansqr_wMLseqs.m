
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
nbits = 127;
tap1 = [1, 0, 0, 0, 1, 1, 1];
tap2 = [1, 0, 0, 0, 1, 0, 0];
init_reg = [1, 1, 1, 1, 1, 1, 1];

% For 511-length codes
% nbits = 511;
% tap1 = [1, 0, 0, 1, 0, 1, 1, 0, 0];
% tap2 = [1, 0, 0, 0, 0, 1, 0, 0, 0];
% init_reg = [1, 1, 1, 1, 1, 1, 1, 1, 1];

% For 1023-length codes
% nbits = 1023;
% tap1 = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0];
% tap2 = [1, 1, 1, 0, 1, 0, 0, 1, 1, 0];
% init_reg = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];


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
   pop(i, :) = reshape(completeGoldCodes(currFamily,:), [npar*codelength,1])';
end


% Get fitness function (2-dimensional) and put in results
[max_obj, auto_obj, cross_obj] ...
    = ff_max_mean_sqr_auto_and_cross_corr(pop);

[min_cost, min_cost_i] = min(max_obj);
disp(['Average Mean-Sqr Auto: ', num2str(mean(auto_obj))]);
disp(['Average Mean-Sqr Cross: ', num2str(mean(cross_obj))]);
disp(' ');
disp(['Best Mean-Sqr Auto: ', num2str( auto_obj(min_cost_i) )]);
disp(['Best Mean-Sqr Cross: ', num2str( cross_obj(min_cost_i) )]);