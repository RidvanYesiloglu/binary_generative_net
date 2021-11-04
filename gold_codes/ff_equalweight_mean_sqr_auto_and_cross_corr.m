function [equalweight_cost, auto_obj, cross_obj] = ff_equalweight_mean_sqr_auto_and_cross_corr(pop)

    % x is complete population 
    global popsize
    global nbits
    global npar
    global start_indices
    global pairs
    global npairs
    
    cross_comp = zeros(popsize, npairs);
    auto_comp = zeros(popsize, npar);
    
    % Compute cross-correlation component
    % For each pair of sequences (1st dimension of size variable)
    for i = 1:npairs
        firstIndex = pairs(i, 1);
        secondIndex = pairs(i, 2);
        
        firstSequences ... 
            = -1*(2*pop(:, firstIndex:(firstIndex+nbits - 1)) - 1);
        secondSequences ...
            = -1*(2*pop(:, secondIndex:(secondIndex+nbits - 1)) - 1);

        % Get correlation for all individuals, for current pair of seq
        corr = abs(...
            ifft(fft( firstSequences' ).*conj(fft( secondSequences' ))) );

        % Get average squared cross-correlation, and add this result
        cross_comp(:,i) = mean(corr.*corr, 1)'; % across nbits for curr seq
        
    end
    
    % Compute auto-correlation component
    for i = 1:npar
        start_index = start_indices(i);
        currSequences ... 
            = -1*(2*pop(:, start_index:(start_index+nbits - 1)) - 1);
        
        % Get correlation component, and add to objective function
        corr = abs(...
            ifft(fft( currSequences' ).*conj(fft( currSequences' ))) );
        
        % Set first row to zero to ignore zero-lag autocorrelation
        corr(1, :) = zeros(1, popsize);
        
        % Get average sqr auto-corr, ignoring 1st, and add this result
        auto_comp(:,i) = mean(corr.*corr, 1)'; % across nbits for curr seq
    end
    
    % Include the mean auto- and cross-correlation cost components
    % And normalize the correlation (divide by nbits again)
    auto_obj = mean(auto_comp, 2);
    cross_obj = mean(cross_comp, 2);
    equalweight_cost = 0.5*(auto_obj + cross_obj);
    
end