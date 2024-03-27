function d = double(~) %#ok<STOUT>
% DOUBLE Convert octonion to double precision (obsolete).
% (Octonion overloading of standard Matlab function.)

% Copyright © 2006 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(['Conversion to double from octonion is not possible. ',...
       'Try cast(q, ''double'')'])

% Note: this function was replaced from version 0.9 with the convert
% function, because it is incorrect to provide a conversion function
% that returns an octonion result. The convert function provides the
% same functionality, but will not be called implicitly by Matlab to
% implement an assignment like X(1,2) = octonion(1,2,3,4) which gave
% erroneous results prior to version 0.9 and now raises an error.
% Convert was itself replaced with cast.

end

% $Id: double.m 1175 2023-07-14 11:21:32Z sangwine $

% Created automatically from the quaternion
% function of the same name on 17-Feb-2020.
