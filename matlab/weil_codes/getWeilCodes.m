function weilCodes = getWeilCodes(L)
% Returns set of length-L Weil codes in +1/-1 notation
% NOTE: expecting L to be prime for Weil code generation (only exist for
% prime number lengths)

% get base sequence (called sequence t in Rushanan 2006)
leg_seq = getLegendreSequence(L);
nWeil = (L-1)/2;   % family size of Weil codes

% create empty set to store entire Weil code family
weilCodes = nan*ones(nWeil, L);

% for each Weil code, generate by circularly shifting base sequence by
% delay i (i goes from 1 to (L-1)/2) and multiplying (see Rushanan 2006)
for i = 1:nWeil
   weilCodes(i, :) = leg_seq .* circshift(leg_seq, [0,-1*i]);
end

end