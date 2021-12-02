function leg_seq = getLegendreSequence(L)
% Returns base sequence to generate length-L Weil codes in +1/-1 notation
% NOTE: expecting L to be prime for Weil code generation (only exist for
% prime number lengths)

% Using definition of Legendre symbol used to define Weil codes 
% Check out the following reference:
%       1. Rushanan, "Weil Sequences: A Family of Binary Sequences with 
%           Good Correlation Properties," 2006 
%       2. https://en.wikipedia.org/wiki/Legendre_symbol)
% Thus, we first want to find what are the set of quadratic residues 
% (squares) modulo L

% Nice info about quadratic residues:
%       https://en.wikipedia.org/wiki/Quadratic_residue
%       https://mathworld.wolfram.com/QuadraticResidue.html
% Both links also have nice table of quadratic residues to check this code

% save set of quadratic residues found in search
legendre_set = nan*ones(1,L-1);

% go through i from 1 to L-1 (for all values up to L)
% note: could technically limit range of search for i <= floor(L/2),
% according to Wolfram link, but it's fine.. being extra sure, I guess
for i = 1:(L-1)
    val = mod(i*i, L);   % find quadratic residue corresponding to this i
    legendre_set(i) = val;   % save the quadratic residue
end
legendre_set = unique(legendre_set);   % reduce set to only unique values
% NOTE: can check the above legendre_set matches tables in links above for
% the given L

% initialize Legendre sequence as all -1 values (will set to 1 if it
% corresponds to a quadratic residue) (see Rushanan 2006)
leg_seq = -1*ones(1, L);

% convert Legendre set to indices to flip (if v is in legendre_set, it is a
% quadratic residue mod L, and this means the Legendre symbol s_v = 1,
% where s_v is actually the (v+1)th element of the sequence -- so if v is 
% in legendre_set, we want to set MATLAB index (v+1) to be +1)
% (see Rushanan 2006)
leg_seq_flip_i = legendre_set+1;

% flip the indices corresponding to quadratic residues to be +1
leg_seq(leg_seq_flip_i) = 1;

% first bit of Weil "base" sequence (called sequence t in Rushanan 2006)
% is defined as a -1 in +/-1 convention (see definition in Rushanan 2006)
leg_seq(1) = -1;   % (should already be -1, but again, extra sure maybe?)

end
