% Test 2 MLS for Gold codes
function code = goldCode(tap1, tap2, init_reg, delay)
% 31-length
%tap1 = [1, 0, 1, 0, 0];
%tap2 = [1, 0, 1, 1, 1];
%init_reg = [1, 0, 0, 0, 0];

% testing 63-length
%tap1 = [1, 0, 0, 0, 0, 1];
%tap2 = [1, 1, 0, 0, 1, 1];
%init_reg = [1, 1, 1, 1, 1, 1];
mls1 = return_LFSR_sequence(tap1, init_reg);
mls2 = return_LFSR_sequence(tap2, init_reg);

%delay = 14;
mls2 = circshift(mls2, [0, delay]);
code = xor(mls1, mls2);
end