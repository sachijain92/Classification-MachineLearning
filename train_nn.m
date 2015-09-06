load('X.mat');
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
No_of_rows = 19978;
D=513;
K=10;
M=20;
wji = rand(M,D);
wkj = rand(K,M+1);

oldE=inf;
oldwji = wji;
oldwkj = wkj;
eta=0.0003;

iterateminctr=0;
crossmin=0;





zjbias = ones(No_of_rows,M+1);

for epoch = 1:20
    aj=X*wji';
    
    zj=tanh(aj);
    zjbias(:,2:end) = zj;
    zjbias(:,1)=1;
    
    ak = zjbias*wkj';
    
    ak=exp(ak);
    acol=sum(ak,2);
    
    yk=ak;
    
    for i = 1:No_of_rows
        yk(i,:)=yk(i,:)/ acol(i);
    end
    
    Error = -sum(sum(Target.*log(yk)));
   
    iterateminctr = iterateminctr +1;
    
    if(oldE> Error)
        oldE = Error;
        oldwkj = wkj;
        oldwji = wji;
    end
    
    if(oldE<Error)
        if iterateminctr <= 10
            disp('Training over');
            break;
        end
        
        crossmin = crossmin +1;
        iterateminctr=0;
        eta = eta / (epoch ^ (1/3));
        wkj = oldwkj;
        wji=oldwji;
        disp(['MIn crossed after ', num2str(epoch)]);
        continue;
    end
    delk= yk - Target;
    delEbydelwkj = delk'*zjbias;
    
    
    
    deljbias = delk*wkj;
    delj = deljbias(:,2:end);
    delEbydelwji = (delj.*(ones(No_of_rows,M) - zj.^2))'*X;
    
    
    
    wkj = wkj - eta * delEbydelwkj;
    wji = wji - eta * delEbydelwji;
    
    disp(['error at epoch', num2str(epoch), ':', num2str(Error)]);
    
    eta = eta + epoch*(100/Error^3)*(1/(crossmin +1));
    
    
end


    aj = X * wji';
    zj = tanh(aj);
    zjbias(:,2:end) = zj;
    zjbias(:,1)=1;
    
    ak = zjbias*wkj';
    
    ak=exp(ak);
    acol=sum(ak,2);
    
    yk=ak;
    
    Predclass = zeros(No_of_rows,10);
    missclasspredperclass = zeros(10,1);
    misspred = 0;
    MRR = zeros(10,1);
    sclass = zeros(10,1);
    for i = 1:No_of_rows
        yk(i,:) = yk(i,:) / acol(i);
        [valuemaxy, indexy] = max(yk(i,:));
        [valuemaxt, indext] = max(Target(i,:));
        Predclass(i,indexy) = 1;
        booleanval = isequal(Predclass(i,:),Target(i,:));
        
        if(booleanval == false)
           misspred = misspred+1; 
           missclasspredperclass(indext) = missclasspredperclass(indext)+1;
        end
        
        probyk = yk(i,indext);
        tempyk = yk(i,:);
        tempyk = sort(tempyk, 'descend');
        rank = find(tempyk == probyk);
        
        reci = 1/rank;
        MRR(indext) = MRR(indext) + reci;
        sclass(indext) = sclass(indext) +1;
    
    end
    
    Classrate = ((No_of_rows- misspred)/No_of_rows)*100;
    disp(['correct classification of ', num2str(Classrate), '% after' num2str(epoch)]);
    
    MRR = MRR./sclass;
    disp({'Mean',num2str(MRR)});
