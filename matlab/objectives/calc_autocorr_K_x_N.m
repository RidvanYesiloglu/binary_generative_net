function [autocorr] = calc_autocorr_K_x_N(codesets, normalize)
    [popsize, npar, nbits] = size(codesets);
    start_indices = linspace(1,1+(npar-1)*nbits, npar);
    pop = zeros(popsize,nbits*npar);
    for i=1:npar
        start_index = start_indices(i);
        pop(:, start_index:(start_index+nbits-1)) = squeeze(codesets(:, i, :));
    end
    autocorr = calc_autocorr(pop, npar, popsize, nbits, start_indices, normalize);
end