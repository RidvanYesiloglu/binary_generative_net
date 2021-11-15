function fullSeq = return_LFSR_sequence(tapArray, reg)
    % assuming register (reg) initialized to go from right to left (i.e.
    % index 1 of reg will be 1st out of the register)
    %       i.e.                    reg = [1 1 1 0 0 0 0] <- 1, 0, 1, ...
    %               becomes    1 <- reg = [1 1 0 0 0 0 1] <- 0, 1, ...
    %               in the next time step
    %               and     1, 1 <- reg = [1 0 0 0 0 1 0] <- 1, ...
    %               in the next time step
    % tapArray corresponds directly with reg (1st index of tapArray
    % corresponds to the 1st index to leave the shift register)
    lenReg = length(tapArray);
    seqLength = 2^lenReg - 1;

    % save vector of full sequence (initialized with reg)
    fullSeq = [reg, zeros(1,seqLength-lenReg)];
    
    % determine remaining contents of shift register
    for i = (lenReg+1):seqLength
        
        % take xor product of current content of shift register, for bits
        % corresponding to tapArray
        fullSeq(i) = mod( dot(reg, tapArray), 2 );  
        
        % shift the register to left by 1, add in the xor product at end
        reg = [ reg(2:lenReg), fullSeq(i) ];
        
    end
    
    
