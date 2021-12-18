function [equalweight_cost, auto_obj, cross_obj] = ff_equalweight_mean_sqr_auto_and_cross_corr(pop)
    % x is complete population 
    global popsize
    global nbits
    global npar
    global start_indices
    global pairs
    global npairs
    
    auto_comp = calc_autocorr(pop, npar, popsize, nbits, start_indices, false);
    % Remove first row to zero to ignore zero-lag autocorrelation
    auto_comp = auto_comp(:, :, 2:nbits);
    cross_comp = calc_crosscorr(pop, popsize, nbits, pairs, npairs, false);

    % Include the mean auto- and cross-correlation cost components
    auto_obj = squeeze(mean(mean(auto_comp.*auto_comp, 2),3));
    cross_obj = squeeze(mean(mean(cross_comp.*cross_comp, 2),3));

    equalweight_cost = 0.5*(auto_obj + cross_obj);
    
end