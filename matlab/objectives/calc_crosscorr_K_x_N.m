function [crosscorr] = calc_crosscorr_K_x_N(codesets, normalize)
    [popsize, npar, nbits] = size(codesets);
    start_indices = linspace(1,1+(npar-1)*nbits, npar);
    npairs = nchoosek(npar, 2);
    pairs = combnk(start_indices, 2);
    pairs = pairs(end:-1:1,:);
    pop = zeros(popsize,nbits*npar);
    for i=1:npar
        start_index = start_indices(i);
        pop(:, start_index:(start_index+nbits-1)) = squeeze(codesets(:, i, :));
    end
    crosscorr = calc_crosscorr(pop, popsize, nbits, pairs, npairs, normalize);
end