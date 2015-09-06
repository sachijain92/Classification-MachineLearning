load('Y.mat');
load('X.mat');
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

activation = Y*W;
ytest = bsxfun(@rdivide,exp(activation),sum(exp(activation),2));

file = fopen('classes_lr.txt','w');
errorctr = 0;
classpre = zeros(1500,1);

for i = 1:1500
    [maxvaluey classpre(i)] = max(ytest(i,:));
    fprintf(file,'%d\n',classpre(i));
    if(classpre(i) ~= Target(i));
        errorctr = errorctr+1;
    end
end
fclose(file);
errorrate = (errorctr/1500)*100;
fprintf('Test Error = %f\n', errorrate);

Recipror = zeros(10,1);
for loop = 1:10
    reciproctr=1;
    k = (loop-1)*150 +1;
    if (k<=1500);
    while(classpre(k) ~= loop)
        reciproctr = reciproctr+1;
        k=k+1;
    end
    end
    Reciprop(loop,1)=reciproctr;
end
Recirank=0;
for m= 1:10
    fprintf('REciprocal rank of class %d =',(m-1));
    
    fprintf('1/%d\n',Reciprop(m));
    Recirank = Recirank + 1/Recipror(m);
end

fprintf('MRR =');
f = Recirank/10;
fprintf('%d\n',f);