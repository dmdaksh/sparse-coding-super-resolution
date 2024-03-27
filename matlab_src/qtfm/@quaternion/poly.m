function c = poly(A)
% POLY Characteristic polynomial of a Hermitian quaternion matrix.

% Copyright © 2022 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

narginchk(1, 1), nargoutchk(0, 1)

if ~ismatrix(A)
    error('Argument must be a matrix')
end

[n, m] = size(A);

if n ~= m
    error('Argument must be a square matrix')
end

% Implementation note. This function does not compute the characteristic
% polynomial in the case of a non-Hermitian matrix, because it can be done
% by the MATLAB POLY function, with just the additional step of computing
% the adjoint matrix using QTFM's ADJOINT function. Because the adjoint
% matrix is of dimension 2n, the polynomial will be of order 2n. In the
% case of a Hermitian matrix we can compute a characteristic polynomial of
% one order less by working directly on the quaternion matrix. For some
% background on the adjoint, see:
%
% Fuzhen Zhang, 'Quaternions and matrices of quaternions', Linear Algebra
% and Applications, 251:21-57 (1997). DOI:10.1016/0024-3795(95)00543-9.
% See particularly: Theorem 8.1 and Corollary 5.1.

if ~ishermitian(A)
    disp('The POLY function is not implemented for non-Hermitian quaternion')
    disp('matrices. A characteristic polynomial may be computed from the')
    disp('adjoint matrix computed by ADJOINT(A) using MATLAB''s POLY function.')
    disp('See implementation note in source code.')
    error('Implementation restriction - see advice above')
end

% The Faddeev–LeVerrier algorithm is used here (see Wikipedia for a
% description and references). The only aspect of the algorithm below that
% is not considered in the Wikipedia article is how to obtain a
% real/complex value from the computed c coefficient, which would otherwise
% be a quaternion. In fact it turns out that the c coefficients come out
% with zero vector part to within rounding error, because the trace of a
% Hermitian matrix is scalar (in the quaternion sense, having zero vector
% part), so we take the scalar part below.

c = [1, zeros([1, n], 'like', A.x)]; % This will be the array of real or
                                     % complex coefficients. The first
                                     % coefficient is always 1.

M = zerosq(n, 'like', A.x); % Auxiliary quaternion matrices (we only need
                            % one of these at a time). This value is M_0,
                            % which is always a matrix of zeros.
I = eye(n, 'like', A.x);

AM = M; % This will become A * M in the loop below, but since M is
        % currently all zeros, we can take a short cut here and avoid a
        % matrix multiplication.

for k = 1:n
    % Indexing: the c row vector is stored with c_n at index 1. So the
    % second element is c_(n-1). Since this is a class method, we have to
    % use subsref and subsasgn here. See the file Implementation_notes.txt,
    % section 8 for an explanation.
    M = AM + subsref(c, substruct('()', {k})) .* I;
    AM = A * M; % Keep this for use on the line above in the next iteration.
    T = -trace(AM) ./ k;
    % c(k + 1) = scalar(T);
    c = subsasgn(c, substruct('()', {k + 1}), scalar(T));

    NVT = abs(normq(vector(T)));
    if NVT > 1e-6 % TODO This tolerance should depend on n.
        warning(['Vector part of coefficient ', num2str(k), ...
                 ' was greater than tolerance: ', num2str(NVT)])
    end
end

% $Id: poly.m 1171 2022-12-26 15:32:59Z sangwine $
