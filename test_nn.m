load('Y.mat');
load('X.mat');
M=20;
zjbias = ones(1500,M+1);
Target = zeros(1500,10);
Target(1:150,1) =1;
Target(151:300,2) =1;
Target(301:450,3) =1;
Target(451:600,4) =1;
Target(601:750,5) =1;
Target(751:900,6) =1;
Target(901:1050,7) =1;
Target(1051:1200,8) =1;
Target(1201:1350,9) =1;
Target(1351:1500,10) =1;

    aj = Y * wji';
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
    class = zeros(No_of_rows,1);
    
       for i = 1:1500
        yk(i,:) = yk(i,:) / acol(i);
        [valuemaxy, indexy] = max(yk(i,:));
        [valuemaxt, indext] = max(Target(i,:));
        Predclass(i,indexy) = 1;
        class(i) = indexy-1;
        
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
       
       MRR = MRR ./ sclass;
       dlmwrite('classes_nn.txt',class);
       Errortest = -sum(sum(Target.*log(yk)));
       disp(['Test error = ', num2str(Errortest)]);
       
       Classrate = ((No_of_rows- misspred)/No_of_rows)*100;
       disp(['correct classification of ', num2str(Classrate), '% after' num2str(i)]);
    
    