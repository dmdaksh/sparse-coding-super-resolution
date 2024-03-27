function test_power
% Test code for the quaternion and octonion power functions (.^), and the
% quaternion mpower (^) function.

% This also tests the exponential and log functions, plus sqrt/conj etc.

% Copyright © 2008, 2022 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

disp('Testing power function (.^) ...');

T = 1e-10;

q = quaternion(randn(100), randn(100), randn(100), randn(100));

% Scalar powers handled as special cases.

compare(q,       q.^1,   T, 'quaternion/power failed test 1.');
compare(q.*q,    q.^2,   T, 'quaternion/power failed test 2.');
compare(sqrt(q), q.^0.5, T, 'quaternion/power failed test 3.');
compare(q,  (q.^-1).^-1, T, 'quaternion/power failed test 4.');
compare(q,(q.^-0.5).^-2, T, 'quaternion/power failed test 5.');

% General scalar power. 3 is not handled as a special case, so we can
% compare it with cubing explicitly.

compare(q.*q.*q, q.^3, T,   'quaternion/power failed test 6.');

% Scalar raised to a vector power.

compare(qi.^[0 1 2 3], [quaternion( 1,  0, 0, 0), ...
                        quaternion( 0,  1, 0, 0), ...
                        quaternion(-1,  0, 0, 0), ...
                        quaternion( 0, -1, 0, 0)],...
                        T, 'quaternion/power failed test 7.');
                    
% Test the octonion power function.
                    
o = rando(100);

% Scalar powers handled as special cases.

compare(o,       o.^1,   T, 'octonion/power failed test 1.');
compare(o.*o,    o.^2,   T, 'octonion/power failed test 2.');
compare(sqrt(o), o.^0.5, T, 'octonion/power failed test 3.');
compare(o,  (o.^-1).^-1, T, 'octonion/power failed test 4.');
compare(o,(o.^-0.5).^-2, T, 'octonion/power failed test 5.');

p = o .* o;
for j = 3:9
    p = p .* o;
    compare(o .^ j, p, T, ['octonion/power failed test 6 with j = ' num2str(j)]);
    compare(o .^ -j, p.^-1, T, ['octonion/power failed test 7 with j = ' num2str(j)]);    
end

disp('Passed')

disp('Testing mpower function (^) ...')

q = randq(3);

compare(q^(-1), inv(q),            T, 'quaternion/mpower failed inverse test.')
compare(q^(-2), inv(q) * inv(q),   T, 'quaternion/mpower failed inverse square test.')
compare(q^0,    eyeq(size(q)),     T, 'quaternion/mpower failed test with power 0.')

compare(q^1,    q,                 T, 'quaternion/mpower failed test with power 1.')
compare(q^2,    q*q,               T, 'quaternion/mpower failed square test.')
compare(q^3,    q*q*q,             T, 'quaternion/mpower failed cube test.')
compare(q^4,    q*q*q*q,           T, 'quaternion/mpower failed test with power 4.')
compare(q^5,    q*q*q*q*q,         T, 'quaternion/mpower failed test with power 5.')
compare(q^6,    q*q*q*q*q*q,       T, 'quaternion/mpower failed test with power 6.')
compare(q^7,    q*q*q*q*q*q*q,     T, 'quaternion/mpower failed test with power 7.')
compare(q^8,    q*q*q*q*q*q*q*q,   T, 'quaternion/mpower failed test with power 8.')
compare(q^9,    q*q*q*q*q*q*q*q*q, T, 'quaternion/mpower failed test with power 9.')

% This test checks out the general case code in MPOWER using matrix
% logarithm and exponential. The test is only valid if SQRTM uses a
% different method, which is the case at time of writing this test
% (SQRTM uses the MATLAB SQRTM on an adjoint matrix).

compare(q^0.5,  sqrtm(q),          T, 'quaternion/mpower failed test with power 0.5')

% Now test the octonion code, which is much more limited in the powers it
% can handle, because of the lack of power associativity in the matrix
% power case.

p = rando(3);

compare(p^0, octonion(eyeq(size(p)), zerosq(size(p))), ...
                  T, 'octonion/mpower failed test with power 0.')
compare(p^1, p,   T, 'octonion/mpower failed test with power 1.')
compare(p^2, p*p, T, 'octonion/mpower failed test with power 2.')

disp('Passed')

% $Id: test_power.m 1173 2022-12-28 18:46:45Z sangwine $
