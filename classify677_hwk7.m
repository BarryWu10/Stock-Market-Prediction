function [confusionMatrix,accuracy,predictions] =  classify677_hwk7(X,y,options)
%function [confusionMatrix,accuracy] =  classify677_hwk7(Xdata,y,options)
% Input:
%      Xdata: A vector of feature, nxD, one set of attributes for each
%                sample (n samples x D dimensional data)
%      y: n x 1 ground truth labeling
%      options.method = {'LogisticRegression', 'KNN', 'SVM','ClassificationTree',
%                                 'BaggedTree', 'Boosting','Adaboost','nnet'}
%      options.numberOfFolds- the number of folds for k-fold cross validation
%      options.lambda - regularization lamda used in LogisticRegression
%      options.knn_k - number of nearest neighbors in KNN
%      options.adaboost_numFeatures- number of decision splits used in boosting
% Output:
%      confusionMatrix- an overall confusion matrix
%      accuracy- an overall accuracy value
%
%  CMPE-789, Machine Intelligence
%  R. Ptucha, 2014
%  Rochester Institute of Technology


[n,D] = size(X);
if length(y) ~= n
    error('X is nxD, and y is nx1');
end
numClasses = length(unique(y));
C = numClasses;

if (~exist('options','var'))
    options = [];
end

if  isfield(options,'method')
    method=options.method;
else
    method='KNN';
end

if strcmp(options.method,'LogisticRegression')
    Xdata = [ones(n, 1) X];
else
    Xdata = X;
end

if  isfield(options,'numberOfFolds')
    numberOfFolds=options.numberOfFolds;
else
    numberOfFolds =2;
end


rng(2000);  %random number generator seed so results are repeatable
%Generate a fold value for each training sample
CVindex = crossvalind('Kfold',y, numberOfFolds);
i=1;  %this is for easier debugging....

for i = 1:numberOfFolds
    
    if (numberOfFolds == 1)  %then test = train set- be careful here!
        %Get train and test index values
        TestIndex = find(CVindex == i);
        TrainIndex = find(CVindex == i);
    else
        %Get train and test index values
        TestIndex = find(CVindex == i);
        TrainIndex = find(CVindex ~= i);
    end
    
    %for train and test partitions
    TrainXdata = Xdata(TrainIndex,:);
    TrainGT = y(TrainIndex);
    TestXdata = Xdata(TestIndex,:);
    TestGT = y(TestIndex);
    
    %
    %build the model using TrainXdata and TrainGT
    %test the built model using TestXdata
    %
    switch method
        case 'LogisticRegression'
            
            if  isfield(options,'lambda')
                lambda = options.lambda;
            else
                lambda = 0;
            end
            
            % for Logistic Regression, we need to solve for theta
            % Initialize fitting parameters
            all_theta = zeros(numClasses, size(Xdata, 2));
            
            for c=1:numClasses
                % Set Initial theta
                initial_theta = zeros(size(Xdata, 2), 1);
                % Set options for fminunc
                opts = optimset('GradObj', 'on', 'MaxIter', 50);
                
                % Run fmincg to obtain the optimal theta
                % This function will return theta and the cost
                [theta] = ...
                    fmincg (@(t)(costFunctionLogisticRegression(t, TrainXdata, (TrainGT == c), lambda)), ...
                    initial_theta, opts);
                
                all_theta(c,:) = theta;
            end
            
            % Using TestDataCV, compute testing set prediction using
            % the model created
            % for Logistic Regression, the model is theta
            all_pred = sigmoid(TestXdata*all_theta');
            [maxVal,maxIndex] = max(all_pred,[],2);
            TestDataPred=maxIndex;
            
        case 'KNN'
            if  isfield(options,'knn_k')
                knn_k = options.knn_k;
            else
                knn_k = 1;
            end
            [idx, dist] = knnsearch(TrainXdata,TestXdata,'k',knn_k);
            nnList=[];
            for i=1:knn_k
                nnList = [nnList TrainDataGT(idx(:,i))];
            end
            TestDataPred=mode(nnList')';
            
        case 'SVM'
            %Note- this is libsvm not the built-in svm functions to matlab
            if  isfield(options,'svm_t')
                svm_t = options.svm_t;
            else
                svm_t = 0;
            end
            if  isfield(options,'svm_c')
                svm_C = options.svm_c;
            else
                svm_C = 1;
            end
            
            if  isfield(options,'svm_g')
                svm_g = options.svm_g;
            else
                svm_g = 1/n;
            end
            
            if  isfield(options,'svm_h')
                svm_h = options.svm_h;
            else
                svm_h = 1;
            end
            
            [TrainXdataNorm, mu, sigma] = featureNormalize(TrainXdata,0,0);
            eval(['model = svmtrain(TrainGT,TrainXdataNorm,''-t  ' num2str(svm_t) ' -h ' num2str(svm_h)  ' -c ' num2str(svm_C) ' -g ' num2str(svm_g) ''' );']);
            
            [TestXdataNorm, mu, sigma] = featureNormalize(TestXdata,mu, sigma);
            TestDataPred = svmpredict( TestGT, TestXdataNorm, model, '-q');
            
        case 'ClassificationTree'
            tree = ClassificationTree.fit(TrainXdata,TrainGT);
            TestDataPred = predict(tree,TestXdata);
            
        case 'BaggedTree'
            rng(2000);  %random number generator seed
            t = ClassificationTree.template('MinLeaf',1);
            bagtree = fitensemble(TrainXdata,TrainGT,'Bag',10,t,'type','classification');
            TestDataPred = predict(bagtree,TestXdata);  %really should test with a test set here
            
        case 'Adaboost'
            if  isfield(options,'adaboost_numFeatures')
                adaboost_numFeatures = options.adaboost_numFeatures;
            else
                adaboost_numFeatures = round(n/2);
            end
            
            if numClasses ~= 2
                error('Adaboost only works with two class data');
            end
            %change class labels to -1 and +1
            yList = unique(TrainGT);
            if yList(1) ~= -1
                TrainGT(TrainGT==yList(1))=-1;
                TrainGT(TrainGT==yList(2))= 1;
                TestGT(TestGT==yList(1))=-1;
                TestGT(TestGT==yList(2))= 1;
            end
            
            [classifiers, errors,pred] = myAdaBoost(TrainXdata,TrainGT,adaboost_numFeatures,TestXdata,TestGT);
            allTrain(i,:) = errors.train;
            allTest(i,:) = errors.test;
            allEB(i,:) = errors.eb;
            
            TestDataPred = pred.test;
            TestDataPred(TestDataPred==1)=yList(2);
            TestDataPred(TestDataPred==-1)=yList(1);
            
            
        case 'nnet'
            if  isfield(options,'nnet_hiddenLayerSize')
                hiddenLayerSize=options.nnet_hiddenLayerSize;
            else
                hiddenLayerSize =10;
            end
            
            %Convert X and y data into Matlab nnet format:
            inputs = TrainXdata';
            
            n = size(TrainXdata, 1); %number of samples = 5000, 500 from each class
            C = length(unique(TrainGT)); %number of classes, class values are 1...10, where 10 is a digit '0'
            %Convert to one-hot encoding ground truth values
            targets = zeros(C,n);
            
            for ii=1:n
                %using y data
                targets(TrainGT(ii),ii) = 1;
            end
            
            % Create a Pattern Recognition Network
            setdemorandstream(2014784333);   %seed for random number generator
            net = patternnet(hiddenLayerSize);
            
            % Set up Division of Data for Training, Validation, Testing
            net.divideParam.trainRatio = 0.8;  %note- splits are done in a random fashion
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0.0;
            
            % Train the Network
            [net,tr] = train(net,inputs,targets); %return neural net and a training record
            % plotperform(tr); %shows train, validation, and test per epoch
            
            %Convert X and y test data into Matlab nnet format:
            inputsTest = TestXdata';
            
            testY = net(inputsTest); %pass input test values into network
            TestDataPred=vec2ind(testY)'; %converts nnet float for each class into most likely class per sample
            
        otherwise
            error('Unknown classification method')
    end
    
    predictionLabels(TestIndex,:) =double(TestDataPred);
end

predictions = predictionLabels;
confusionMatrix = confusionmat(y,predictionLabels);
accuracy = sum(diag(confusionMatrix))/sum(sum(confusionMatrix));


%fprintf(sprintf('%s: Accuracy = %6.2f%%%% \n',method,accuracy*100));

%fprintf('Confusion Matrix:\n');
%[r c] = size(confusionMatrix);
%for i=1:r
%    for j=1:r
%        fprintf('%6d ',confusionMatrix(i,j));
%    end
%    fprintf('\n');
%end

if strcmp(options.method,'Adaboost')
    meanTest = mean(allTest);
    meanTrain = mean(allTrain);
    meanEB  = mean(allEB);
    figure
    hold on
    x = 1:1:adaboost_numFeatures;
    plot(x, meanEB,'k:',x,meanTest,'r-',x,meanTrain,'b--','LineWidth',2);
    legend('ErrorBound','TestErr','TrainErr','Location','Best');
    xlabel 'iteration (number of Classifiers)'
    ylabel 'error rates (50 trials)'
    title 'AdaBoost Performance on Bupa'
    %print -dpng hwk5_11.png
end