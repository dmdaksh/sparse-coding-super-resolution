function r = dictresidual(A, B)

path1='/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/dicts_bkp/Dh_512_US3_L0.1_PS5.pkl';
path2='/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/dicts_bkp/Dh_1024_US3_L0.1_PS5.pkl';

% 'Dh_1024_US3_L0.1_PS5.pkl',...	
%     'Dh_2048_US3_L0.1_PS5.pkl'
B=loadDictionaryPkl(path1);
A=loadDictionaryPkl(path2);

[ax,ay]=size(A);
[bx,by]=size(B);

ny = min([ay,by]);
nx = min([ax,bx]);

[Ua,Sa,Va]=svd(A);
[Ub,Sb,Vb]=svd(B);

G = Ua(:,1:ny);
H = Ub(:,1:ny);

G = A(:,1:ny);
H = B(:,1:ny);

r = norm( (G*G' - eye)*H, 'fro')^2;

% G = Va(1:,1:nx);
% H = Vb(1:,1:nx);
% r = norm( (G*G' - eye)*H, 'fro')^2;



