function h=display_network_nonsquare_subplots(A, numcols)
%  display_network -- displays the state of the network
%  A = basis function matrix
%
% [SCSR][rj] - use to display dictionary atoms, i.e., learned patches.
% Patches are stored in dict as column vecs; this reshapes them to square
% patches and displays all patches as adjacent square matrices.
%

warning off all

[L, M]=size(A);
if ~exist('numcols', 'var')
    numcols = ceil(sqrt(L));
    while mod(L, numcols), numcols= numcols+1; end
end
ysz = numcols;
xsz = ceil(L/ysz);

m=floor(sqrt(M*ysz/xsz));
n=ceil(M/m);

colormap(gray)

buf=1;
array=-ones(buf+m*(xsz+buf),buf+n*(ysz+buf));

k=1;
for i=1:m
    for j=1:n
        if k>M, continue; end
        clim=max(abs(A(:,k)));
        array(buf+(i-1)*(xsz+buf)+[1:xsz],buf+(j-1)*(ysz+buf)+[1:ysz])=...
            reshape(A(:,k),xsz,ysz)/clim;
        k=k+1;
    end
end
nexttile;
if isreal(array)
    h=imagesc(array,'EraseMode','none',[-1 1]);
else
    h=imagesc(20*log10(abs(array)),'EraseMode','none',[-1 1]);
end
axis image off

drawnow

warning on all
