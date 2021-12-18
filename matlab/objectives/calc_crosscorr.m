function [crosscorr] = calc_crosscorr(pop, popsize, nbits, pairs, npairs, normalize)
    crosscorr = zeros(popsize, npairs, nbits); 

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
        corr = real(ifft(fft( firstSequences' ).*conj(fft( secondSequences' ))) );
        crosscorr(:,i,:) = reshape(corr',[popsize,1,nbits]);
    end
    if normalize == 1 % Normalize:
        crosscorr = crosscorr/nbits;
    end
end

