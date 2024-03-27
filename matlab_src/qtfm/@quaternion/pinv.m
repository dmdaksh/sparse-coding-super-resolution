function B = pinv(A, tol)
% PINV   Pseudoinvers
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2023 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% Prior to version 3.5 of QTFM this function was not overloaded for
% quaternions, because the Matlab function PINV worked for quaternions.
% Matlab release R2023b broke this in a way that could not be fixed,
% because the Matlab code used undocumented Matlab built-in functions.
% That is why this code was introduced.

narginchk(1, 2), nargoutchk(0, 1)

% TODO Add some sanity checks on the parameters here.

[U, S, V] = svd(A, "vector");

if nargin == 1

    % The default tolerance calculation is copied from GNU Octave, see:
    % https://octave.sourceforge.io/octave/function/pinv.html

    tol = max(size(A)) * max(S) * eps; 
end

L = S >= tol; % Logical index for the values in S that we retain.

S(~L) = 0; % Suppress the singular values below the tolerance.

S(L) = S(L).^-1; % Invert the non-zero values in S.

% Make a diagonal matrix T out of S, conformal with V and so that V * T is
% conformal with U'. This requires that the number of rows in T matches the
% number of columns in V, and that the number of columns in T matches the
% number of rows in U (technically U', but since U is square this is the
% same thing).

T = zeros(size(V, 2), size(U, 1), 'like', S);

for i = 1:length(S), T(i, i) = S(i); end

B = V * T * U';

end

% $Id: pinv.m 1176 2023-10-10 16:29:01Z sangwine $
