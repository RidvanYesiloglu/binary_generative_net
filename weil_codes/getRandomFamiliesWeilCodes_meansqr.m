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

% Specify taps / register initialization
nbits = 127;

% Specify number of sequences desired
npar = 10;
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
   pop(i, :) = reshape(completeWeilCodes(currFamily,:), [npar*nbits,1])';
end


% Get fitness function (2-dimensional) and put in results
[max_obj, auto_obj, cross_obj] ...
    = ff_max_mean_sqr_auto_and_cross_corr(pop);

% for i = 1:popsize
%    % Generate random collection (sampling without replacement)
%    currFamily = datasample(1:codelength, npar, 'Replace', false);
%    
%    % Get mean auto-correlation
%    results(i,1) = mean( meanAutoCorr(currFamily) );
%    
%    % Get mean cross-correlation
%    % First generate all pairs
%    currPairs = nchoosek(currFamily,2);
%    currMeanCrossCorr = 0;
%    for j = 1:size(currPairs,1)
%       code1_ID = currPairs(j,1); code2_ID = currPairs(j,2);
%       currMeanCrossCorr ...
%           = currMeanCrossCorr + meanCrossCorr(code1_ID, code2_ID);
%    end
%    results(i,2) = currMeanCrossCorr / size(currPairs,1);
% end
% 
% figure();
% plot(results(:,1)/codelength, results(:,2)/codelength, '*', 'color', stanfordGreen);
% grid on; hold on;
% % Get Pareto points & plot these points
% [paretoPoints, ~] = paretoFront(results);
% plot(paretoPoints(:,1)/codelength, paretoPoints(:,2)/codelength, '*', 'color', stanfordRed);
% xlabel('Mean Auto-Correlation Side Peak');
% ylabel('Mean Cross-Correlation Side Peak');
% title(['Gold Code Auto- and Cross-Correlation', sprintf('\n'), ...
%     '(', num2str(npar), ' sequences, length ', num2str(codelength), ...
%     ', with ', num2str(popsize), ' random families)']);
% grid on;
[min_cost, min_cost_i] = min(max_obj);
disp(['Average Mean-Sqr Auto: ', num2str(mean(auto_obj))]);
disp(['Average Mean-Sqr Cross: ', num2str(mean(cross_obj))]);
disp(' ');
disp(['Best Mean-Sqr Auto: ', num2str( auto_obj(min_cost_i) )]);
disp(['Best Mean-Sqr Cross: ', num2str( cross_obj(min_cost_i) )]);