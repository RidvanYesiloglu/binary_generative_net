function bin_seq = rand_bernoulli(p, nSeq)
% INPUTS:
% p represents probability vector for which to select 1
%     must have a length which corresponds to the sequence length
%     Ex: even bias of length-5 sequences is p=[0.5, 0.5, 0.5, 0.5, 0.5]
% nSeq represents number of random sequences to generate with prob p
%
% OUTPUTS:
% bin_seq is nseq x length(p) collection of sequences, sampled from
%     Bernoulli distribution represnted by p vector
    bin_seq = rand( nSeq, length(p) );
    p_mat = repmat(p, [nSeq, 1]);
    bin_seq = (bin_seq < p_mat);

end