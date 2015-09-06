load('X.mat');
[row,col] = size(X);
T0=zeros(2000,10);
T0(:,1)=1;
T1=zeros(1979,10);
T1(:,2)=1;
T2=zeros(1999,10);
T2(:,3)=1;
T3=zeros(2000,10);
T3(:,4)=1;
T4=zeros(2000,10);
T4(:,5)=1;
T5=zeros(2000,10);
T5(:,6)=1;
T6=zeros(2000,10);
T6(:,7)=1;
T7=zeros(2000,10);
T7(:,8)=1;
T8=zeros(2000,10);
T8(:,9)=1;
T9=zeros(2000,10);
T9(:,10)=1;
Target = [T0;T1;T2;T3;T4;T5;T6;T7;T8;T9];
W= rand(513,10);
activation=X*W;
y=bsxfun(@rdivide,exp(activation),sum(exp(activation),2));
delE=((y-Target)'*X)';
Error=-(sum(sum(Target.*log(y))));

Errorold = inf;
Errornew = Error;
eta= 0.00003;
for i=1:1000
    W = W - (eta*delE);
    delEold=delE;
    activation=X*W;
    y=bsxfun(@rdivide,exp(activation),sum(exp(activation),2));
    Errorold = Errornew;
    delE=((y-Target)'*X)';
    Errornew=-(sum(sum(Target.*log(y))));
    if(Errornew<Errorold)
        Errorold=Errornew;
    else
        W = W+ (eta*delEold);
        delE = delEold;
        eta = 0.75*eta;
    end
    errorpl(i) = Errornew;
end

plot(1:1000,errorpl);
