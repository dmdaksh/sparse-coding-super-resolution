function test_eig
% Test code for the quaternion eigenvalue decomposition and characteristic
% polynomial.

% Copyright © 2006, 2022 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

disp('Testing eigenvalue decomposition ...')

T = 1e-12;

A = quaternion(randn(10,10),randn(10,10),randn(10,10),randn(10,10));
A = A * A'; % The matrix must be Hermitian.

[V, D] = eig(A);
compare(A * V , V * D, T, 'quaternion/eig failed test 1A')

D = eig(A);
compare(A * V, ...
        V * diag(D), T,   'quaternion/eig failed test 1B')

disp('Passed')

disp('Testing characteristic polynomial ...')

A = randq(4);
A = A * A'; % The matrix must be Hermitian.

P = poly(A); % Compute the characteristic polynomial.

D = eig(A); % Compute the eigenvalues using the code tested above, each of
            % which should satisfy the characteristic polynomial.
T = 1e-10;

for k=1:length(D)
    check(abs(polyval(P, D(k))) < T, 'quaternion/poly failed test 2')
end

disp('Passed');

% $Id: test_eig.m 1171 2022-12-26 15:32:59Z sangwine $

