function weilCodes = getWeilCodes(L)

leg_seq = getLegendreSequence(L);
nWeil = (L-1)/2;
weilCodes = nan*ones(nWeil, L);
for i = 1:nWeil
   weilCodes(i, :) = leg_seq .* circshift(leg_seq, [0,i]);
end

end