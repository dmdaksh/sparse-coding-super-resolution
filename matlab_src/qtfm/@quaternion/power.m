function Z = power(X, Y)
% .^   Array power.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005, 2006, 2008, 2016, 2022 Stephen J. Sangwine
%                                      and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

narginchk(2, 2), nargoutchk(0, 1)

% This function can handle left and right parameters of the same size, or
% cases where one or other is a scalar.  The general case of a matrix or
% vector raised to a matrix or vector power requires elementwise operations
% and is handled using a general formula using logarithms, even though some
% of the elements of the right argument may be special cases (discussed
% below).

% When the right operand is a scalar, some special cases are handled using
% specific formulae because of the greater accuracy or better speed
% available. E.g. for Y == -1, the elementwise inverse is used, for Y == 2,
% elementwise squaring is used.

% For a power of ± 1/2, the sqrt function is used, with or without a
% reciprocal.

% TODO Copy across code from Clifford, for the case of a vector of
% non-negative integer powers.

if isscalar(Y) && isnumeric(Y)
    if round(Y) == Y && isreal(Y)
        % Exponent is a real integer.
        if Y < 0
            % Negative powers. Invert the elements of X and call this
            % function recursively on the result with positive Y.
            Z = (conj(X) ./ normq(X)) .^ abs(Y);
        elseif Y == 0
            Z = onesq(size(X), 'like', X.x);
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

            S = X; % First power, X.^1.

            for j = 1:L % Find the first non-zero bit in B.
                if B(j) == '1'
                    Z = S; % Set Z to the power of X stored in S.
                    break % ... and move onto the second loop.
                else
                    S = S .* S; % Compute the next power of 2 by
                                % squaring, ready for j+1 in this loop.
                end
            end

            % Notice that we switch from computing the square at the
            % end of the loop (above), to computing the square at the
            % start of the loop (below). This avoids us computing a
            % square at the end that we don't need, whereas above it
            % enabled us to enter the loop with X as the first power.

            for k = j+1:L
                S = S .* S;
                if B(k) == '1'
                    Z = Z .* S;
                end
            end
        end
    else
        % Non-integer or non-real (complex) case. Deal with square root
        % cases separately in order to use the square root algorithm
        % (assumed better than logs).
        if Y == 1/2
            Z = sqrt(X);
        elseif Y == -1/2
            Z = sqrt(X .^ -1);
        else
            Z = general_case(X, Y);
        end
    end

elseif isscalar(X)

    % X is a scalar, but Y is not (otherwise it would have been handled
    % above). The general case code will handle this, since it will expand
    % X to the same size as Y before pointwise multiplication.
    
    Z = general_case(X, Y);

else
    
    % Neither X nor Y is a scalar, therefore we have to use the general
    % method. This will work only if the sizes are compatible. From Matlab
    % R2016b, 'compatible' has a looser interpretation based on implicit
    % singleton expansion. Rather than try to check for compatibility here,
    % we have removed the size check and we leave it to the Matlab code to
    % raise an error if the sizes are incompatible.
        
    Z = general_case(X, Y);

end

end

function Z = general_case(X, Y)
% The formula used here is taken from:
%
% A quaternion algebra tool set, Doug Sweetser,
%
% http://www.theworld.com/~sweetser/quaternions/intro/tools/tools.html
        
Z = exp(log(X) .* Y); % NB log(X) is the natural logarithm of X.
                      % (Matlab convention.)
                      
end

% $Id: power.m 1179 2023-10-10 16:41:06Z sangwine $
