function leg_seq = getLegendreSequence(L)
% L = 11;
legendre_set = nan*ones(1,L-1);
for i = 1:(L-1)
    val = mod(i*i, L);
    legendre_set(i) = val;
end
legendre_set = unique(legendre_set);
% disp('legendre set')
% disp(legendre_set)
% disp(' ')

% First is defined as a 1
leg_seq = -1*ones(1, L);
leg_seq_flip_i = legendre_set+1;
leg_seq(leg_seq_flip_i) = 1;
leg_seq(1) = -1; 
% disp('legendre sequence')
% disp(leg_seq)
end
