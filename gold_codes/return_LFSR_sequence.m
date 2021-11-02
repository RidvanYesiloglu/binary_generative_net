function fullSeq = return_LFSR_sequence(tapArray, reg)
    %prn = 26;

    lenReg = length(tapArray);
    seqLength = 2^lenReg - 1;

    %reg = ones(1,lenReg);
    %reg = [1, 0, 0, 0, 0];
    fullSeq = [reg, zeros(1,seqLength-lenReg)];
    
    for i = (lenReg+1):seqLength
        fullSeq(i) = mod( dot(reg, tapArray), 2 );
        %g2(i) = mod( dot(g2reg, tap2), 2 );
        reg = [ reg(2:lenReg), fullSeq(i) ];
        %g2reg = [ g2reg(2:10), g2(i) ];
    end

    %delays = [5,6,7,8,17,18,139,140,141,251,252,254,255,256,257,258, ...
    %          469,470,471,472,473,474,509,512,513,514,515,516,859,860, ...
    %          861,862,863,950,947,948,950];

    %g2 = circshift(g2, [0, delays(prn)]);
    
    %prn_sequence = xor(g1, g2);
    
    
