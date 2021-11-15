% Test 2 MLS for Gold codes
function code = goldCode(tap1, tap2, init_reg, delay)

% get maximum length sequences first
mls1 = return_LFSR_sequence(tap1, init_reg);
mls2 = return_LFSR_sequence(tap2, init_reg);

% shift first code by delay
mls2 = circshift(mls2, [0, delay]);
code = xor(mls1, mls2);
end