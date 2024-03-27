function [  ] = display_patches( M, patchsize, nb )

nb = 100;
patchsize = 5;
M = H;

n = size(M,2);
if nb>n, nb=n; end

Mre = zeros(patchsize, nb*patchsize);

for nn=1:nb
    curr = reshape(M(:,nn),[patchsize,patchsize]);
    yy = (nn-1)*patchsize+1:nn*patchsize;
    Mre(:,yy) = curr;
end

Mres = reshape(Mre,[50,50]);
Mresp = zeros(size(Mres)+10);
for xx=1:10
%     cx = xx:xx+patchsize-1;
    cx = (xx-1)*patchsize+1:xx*patchsize;
    dx = cx+(xx-1);
    for yy=1:10
%         cy = yy:yy+patchsize-1;
        cy = (yy-1)*patchsize+1:yy*patchsize;
        dy = cy+(yy-1);
        Mresp(dx,dy) = Mres(cx,cy);
        Mresp(dx(end)+1,dy) = -10000;
        Mresp(dx(end)+1,dy(end)+1) = -10000;
        Mresp(dx,dy(end)+1) = -10000;
%         Mresp(dx(end)+1,dy(end)+1) = -10000;
    end
end


% nbs = sqrt(nb);
% Mres = zeros(nbs*patchsize, nbs*patchsize);
% for nn=1:nbs
%     curr = zeros(patchsize, nbs*patchsize);
%     for jj=1:nbs
%         yy = (nn-1)*patchsize+1:nn*patchsize;
%         curr = Mre(:,yy);



