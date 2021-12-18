function [autocorr] = calc_autocorr(pop, npar, popsize, nbits, start_indices, normalize)
    autocorr = zeros(popsize, npar, nbits);
    % Compute auto-correlation component
    for i = 1:npar
        start_index = start_indices(i);
        currSequences ... 
            = -1*(2*pop(:, start_index:(start_index+nbits - 1)) - 1);
        % Get correlation component, and add to objective function
        corr = real(ifft(fft( currSequences' ).*conj(fft( currSequences' ))));
        % Get average sqr auto-corr, ignoring 1st, and add this result
        autocorr(:,i,:) = reshape(corr',[popsize,1,nbits]); % across nbits for curr seq
    end
    if normalize == 1 % Normalize
        autocorr = autocorr/nbits;
    end
end