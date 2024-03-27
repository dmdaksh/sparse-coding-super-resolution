function Z = mpower(X, Y)
% ^   Matrix power.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2008, 2009, 2022 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

narginchk(2, 2), nargoutchk(0, 1)

% There are three cases that we can handle:
%
% 1. X and Y are both scalar. Handled by the .^ (power) function.
% 2. X is a square quaternion matrix and Y is a numeric scalar (negative
%    values included). This can be computed by repeated squaring if Y is an
%    integer, or using a matrix exponential/logarithm if not.
% 3. Neither of X or Y is scalar. This is an error, just as in the Matlab ^
%    function. There is no way to raise a (quaternion) matrix to a
%    (quaternion) matrix power.

if isscalar(X) && isscalar(Y) % Case 1.
    Z = power(X, Y); % The power function will check the sanity of X and Y.
    return;
end

if ~isscalar(X) && ~isscalar(Y) % Case 3.
   error('At least one operand must be scalar.') 
end

if isscalar(Y) && isnumeric(Y) % Possibly case 2 ...
    if ismatrix(X) && (size(X, 1) == size(X, 2)) % if X is square.
        % Case 2 confirmed.
        if round(Y) == Y && isreal(Y) % Test whether Y is a real integer.
            % Case 2, integer Y. We can handle positive integer values by
            % repeated multiplication.
            if Y < 0
                % Negative powers. Invert the matrix and call this function
                % recursively on the inverse with positive Y.
                Z = inv(X)^abs(Y); return
            elseif Y == 0
                Z = eyeq(size(X), 'like', X.x);
                return
            else
                % Y must be greater than zero. We use the method of
                % repeated squaring, composing the result as the product of
                % selected powers of 2, using the binary representation of
                % Y to select the powers, one bit at a time. See, for
                % example:
                % https://www.planetmath.org/computingpowersbyrepeatedsquaring

                B = flip(dec2bin(Y)); % Express Y as a sequence of bits in
                                      % binary, least significant bit on
                                      % the left at index 1. There will be
                L = length(B);        % no leading zeros.

                % We use a string of bits for ease of coding, rather than
                % working with a numeric representation. We assume that the
                % matrix squaring and multiplication will be much more time
                % consuming than the bit twiddling.

                % We work sequentially here, so that only one power of X
                % has to be stored at a time. This is an important
                % consideration because we might be handling a large power
                % of a large matrix.

                S = X; % First power, X^1.

                for j = 1:L % Find the first non-zero bit in B.
                    if B(j) == '1'
                        Z = S; % Set Z to the power of X stored in S.
                        break % ... and move onto the second loop.
                    else
                        S = S * S; % Compute the next power of 2 by
                                   % squaring, ready for j+1 in this loop.
                    end
                end

                % Notice that we switch from computing the square at the
                % end of the loop (above), to computing the square at the
                % start of the loop (below). This avoids us computing a
                % square at the end that we don't need, whereas above it
                % enabled us to enter the loop with X as the first power.

                for k = j+1:L
                    S = S * S;
                    if B(k) == '1'
                        Z = Z * S;
                    end
                end
                return
            end
        else
            % Case 2, non-integer Y. There are bound to be complexities to
            % this formula that we have not investigated, especially due to
            % the matrix logarithm. But at least providing the formula
            % makes it possible to compute a value, rather than raising an
            % error. See the POWER function for the inspiration for this
            % formula, which may seem obvious, but it wasn't until 2022!

            Z = expm(logm(X) * Y); return
        end
    else
        error('Matrix must be square to compute power.')
    end
else
    error(['Exponent must be a scalar, numeric value, given: ', ...
           class(Y), ' of dimension [', num2str(size(Y)), ']'])
end

end

% $Id: mpower.m 1173 2022-12-28 18:46:45Z sangwine $
