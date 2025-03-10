function [x,losses,iter_count]=L1QP_FeatureSign_yang(lambda,A,b)
% [x,loss]=L1QP_FeatureSign_yang(lambda,A,b)

A = double(A);
b = double(b);

EPS = 1e-9;
x=zeros(size(A, 1), 1);           %coeff

grad=A*sparse(x)+b;
[ma, mi]=max(abs(grad).*(x==0));

cnt=0;
losses = [];
iter_count.inner_loop = [];
iter_count.outer_loop = [];
while true
    cnt=cnt+1;
%     if mod(cnt,10000)==0, fprintf('outcnt=%d\n',cnt); end
  
  if grad(mi)>lambda+EPS
    x(mi)=(lambda-grad(mi))/A(mi,mi);
  elseif grad(mi)<-lambda-EPS
    x(mi)=(-lambda-grad(mi))/A(mi,mi);            
  else
    if all(x==0)
      break;
    end
  end    
  
  incnt=0;
  while true
      incnt=incnt+1;
%       if mod(incnt,10000)==0, fprintf('incnt=%d\n',incnt); end

    a=x~=0;   %active set
    Aa=A(a,a);
    ba=b(a);
    xa=x(a);

    %new b based on unchanged sign
    vect = -lambda*sign(xa)-ba;
    x_new= Aa\vect;
    idx = find(x_new);
    o_new=(vect(idx)/2 + ba(idx))'*x_new(idx) + lambda*sum(abs(x_new(idx)));
    
    %cost based on changing sign
    s=find(xa.*x_new<=0);
    if isempty(s)
      x(a)=x_new;
      loss=o_new;
      losses(end+1)=loss;
      break;
    end
    x_min=x_new;
    o_min=o_new;
    d=x_new-xa;
    t=d./xa;
    for zd=s'
      x_s=xa-d/t(zd);
      x_s(zd)=0;  %make sure it's zero
%       o_s=L1QP_loss(net,Aa,ba,x_s);
      idx = find(x_s);
      o_s = (Aa(idx, idx)*x_s(idx)/2 + ba(idx))'*x_s(idx)+lambda*sum(abs(x_s(idx)));
      if o_s<o_min
        x_min=x_s;
        o_min=o_s;
      end
    end
    
    x(a)=x_min;
    loss=o_min;
    losses(end+1)=loss;
    iter_count.inner_loop(end+1)=incnt;
  end 
    
  grad=A*sparse(x)+b;
  
  [ma, mi]=max(abs(grad).*(x==0));
  if ma <= lambda+EPS
    break;
  end
  
end

iter_count.outer_loop = cnt;

%   ORIGINAL HEADER/HELP NOTES:
% %% L1QP_FeatureSign solves nonnegative quadradic programming 
% %% using Feature Sign. 
% %%
% %%    min  0.5*x'*A*x+b'*x+\lambda*|x|
% %%
% %% [net,control]=NNQP_FeatureSign(net,A,b,control)
% %%  
% %% 
% %%


