function [max_cost, auto_obj, cross_obj] = ff_max_mean_sqr_auto_and_cross_corr(pop)

    % x is complete population 
    global nbits
    global npar
    global start_indices
    global pairs
    global npairs
    
    popsize = size(pop,1);
    
    auto_comp = calc_autocorr(pop, npar, popsize, nbits, start_indices,1);
    % Remove first row to zero to ignore zero-lag autocorrelation
    auto_comp = auto_comp(:, :, 2:nbits);
    cross_comp = calc_crosscorr(pop, popsize, nbits, pairs, npairs,1);

    % Include the mean auto- and cross-correlation cost components
    % And normalize the correlation (divide by nbits again)
    auto_obj = squeeze(mean(mean(auto_comp.*auto_comp, 2),3));
    cross_obj = squeeze(mean(mean(cross_comp.*cross_comp, 2),3)); % across nbits for curr seq
    max_cost = max(auto_obj, cross_obj);
end