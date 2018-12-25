% closes everything
close all ; clear all; clc;

% setup paths and directories
cd C:\Users\bxw8904\Desktop\stock_market_prediction\stock_market_prediction
addpath C:\Users\bxw8904\Desktop\stock_market_prediction\stock_market_prediction\libsvm-3.18\libsvm-3.18\windows
addpath C:\Users\bxw8904\Desktop\stock_market_prediction\stock_market_prediction\DOWJONES_CSV_train
addpath C:\Users\bxw8904\Desktop\stock_market_prediction\stock_market_prediction\DOWJONES_CSV_test
addpath C:\Users\bxw8904\Desktop\stock_market_prediction\stock_market_prediction\DOWJONES_CSV_data

fprintf("Adding paths to predict stock prices for the Dow Jones\n");
% gets the companies in the dow jones industrial average
dowTxtFile = fopen('dow.txt','r');
dowDataCell = textscan(dowTxtFile,'%s');
dow = string(dowDataCell{1,1});
fclose(dowTxtFile);

fprintf("Added the companies in the Dow Jones\n");

% for each company in the dow jones industrial average
for dowIndex = 1:numel(dow)
    
    fprintf("Making the model for " + dow(dowIndex) + '\n');
    %eveything = load('C:\Users\zxasq\Desktop\stock_market_prediction\DOWJONES_Industrial_Average_Stocks\' + dow(dowIndex)+".csv");
    train = load(dow(dowIndex)+"_train.csv");
    test = load(dow(dowIndex)+"_test.csv");
    
    
    trainX = train(:,1:end -1);
    trainGT = train(:,end);
    testX = test(:,1:end -1);
    testGT = test(:,end);
    
    test_matrix_size = size(testGT);
    m = test_matrix_size(1);
    
    x_axis = linspace(0,m,m);
    figure
    plot(x_axis, testGT);
    title(sprintf('Ground Truth: '+dow(dowIndex)));
    xlabel("Days Ahead Predicted");
    ylabel("Closing Price");
    print("C:\Users\bxw8904\Desktop\stock_market_prediction\stock_market_prediction\DOWJONES_Prediction_GT\"+dow(dowIndex),'-dpng','-r0')
    
    
    %% SVM
    t = 2;
    %b = 1;
    %s = 3;
    c = 2;
    g = 1;
    %model = svmtrain( trainGT, trainX,  sprintf('-t %d -c %d -g %d -b %d  -s %d', t, c, g, b, s) );
    model_svm = svmtrain( trainGT, trainX,  sprintf('-t %d -c %d -g %d ', t, c, g) );
    %model_svm = fitcecoc(trainX,trainGT);
    
    %predict_svm = svmpredict(testGT,testX,model, '-b 1');
    predict_svm = svmpredict(testGT,testX,model_svm);
    
    svm_mse = mse(predict_svm,testGT);
    
    fprintf("SVM finished...\n");
    
    %% Ridge Regression
    model_ridge = ridge(trainGT,trainX,1:1:m);
    test_ridge = testX .* model_ridge';
    sum_test_rige = sum(test_ridge,2);
    sum_weights = sum(model_ridge);
    predict_ridge = sum_test_rige ./ sum_weights';
    ridge_mse = mse(predict_ridge,testGT);
    fprintf("Ridge finished...\n");

    %% LS Boost
    model_ls = fitrensemble(trainX,trainGT,'Method','LSBoost','NumLearningCycles',100, 'LearnRate', 0.1);
    predict_ls = predict(model_ls,testX);
    
    ls_mse = mse(predict_ls,testGT);
    
    fprintf("LS finished...\n");
    
    %% Bagged Tree
    model_bagged_tree = fitrensemble(trainX,trainGT,'Method','bag', 'NumLearningCycles',100);
    predict_bt = predict(model_bagged_tree,testX);
    bagged_mse = mse(predict_bt, testGT);
    fprintf("BT finished...\n");
    
    %% AdaBoost
    model_ada = fitcensemble(trainX,trainGT,'Method','AdaBoostM2', 'NumLearningCycles',10);
    predict_ada = predict(model_ada,testX);
    ada_mse = mse(predict_ada,testGT);
    fprintf("ADA finished...\n");
    
    %% Weighted average
    numberOfVotes = 0;
    weighted_prediction = zeros(21,1);
    
    if(svm_mse < 100)
        weighted_prediction = weighted_prediction + predict_svm;
        numberOfVotes = numberOfVotes + 1;       
    end
    
    if(ridge_mse < 100)
        weighted_prediction = weighted_prediction + predict_ridge;
        numberOfVotes = numberOfVotes + 1;       
    end
    
    
    if(ls_mse < 100)
        weighted_prediction = weighted_prediction + predict_ls;
        numberOfVotes = numberOfVotes + 1;       
    end
    
    if(bagged_mse < 100)
        weighted_prediction = weighted_prediction + predict_bt;
        numberOfVotes = numberOfVotes + 1;       
    end
    
    if(ada_mse < 100)
        weighted_prediction = weighted_prediction + predict_ada;
        numberOfVotes = numberOfVotes + 1;       
    end
    %weighted_prediction = predict_svm + predict_ridge + predict_ls + predict_bt + predict_ridge + predict_ls + predict_bt + predict_ridge + predict_ls + predict_bt;
    %weighted_prediction = weighted_prediction ./ 10;
    
    
    weighted_prediction = weighted_prediction ./ numberOfVotes;
    
    weighted_mse = mse(weighted_prediction,testGT);
    weighted_percent_error = abs((weighted_prediction-testGT)./testGT).* 100;
    weighted_percent = mean(weighted_percent_error);
    
    fprintf("Weights finished...\n");
    
    
    %% Pretty pictures
    
    %disp(svm_mse);
    %disp(ridge_mse);
    %disp(weighted_mse);
    %disp(weighted_percent);
    %print("C:\Users\zxasq\Desktop\stock_market_prediction\DOWJONES_Prediction\"+dow(dowIndex),'-dpng','-r0')
    
    figure
    hold off;
    plot(x_axis, testGT,'LineWidth',2);
    hold on;
    plot(x_axis, predict_svm,'LineWidth',2);
    hold on;
    plot(x_axis, predict_ridge,'LineWidth',2);
    hold on;
    plot(x_axis, predict_ls,'LineWidth',2);
    hold on;
    plot(x_axis, predict_bt,'LineWidth',2);
    hold on;
    plot(x_axis, predict_ada,'LineWidth',2);
    grid on;
    title(sprintf('Ground Truth vs Machine Learning Model: '+dow(dowIndex)));
    xlabel("Days Ahead Predicted");
    ylabel("Closing Price");    
    legend('GT','SVM', 'Ridge', 'LS Boost', 'Bagged Tree', 'AdaBoost','location','best');
    
    print("C:\Users\bxw8904\Desktop\stock_market_prediction\stock_market_prediction\DOWJONES_Prediction_models\"+dow(dowIndex),'-dpng','-r0')
    
    figure
    hold off;
    plot(x_axis, testGT,'LineWidth',2);
    hold on;
    plot(x_axis, weighted_prediction, 'r','LineWidth',2);
    grid on;
    title(sprintf('Ground Truth vs Weighted Model: '+dow(dowIndex)));
    xlabel("Days Ahead Predicted");
    ylabel("Closing Price");
    legend('GT','Weighted Prediction', 'location','best');
    
    print("C:\Users\bxw8904\Desktop\stock_market_prediction\stock_market_prediction\DOWJONES_Prediction_results\"+dow(dowIndex),'-dpng','-r0')
    
    disp("SVM: " + num2str(svm_mse));
    disp("ridge_mse: " + num2str(ridge_mse));
    disp("ls_mse: " + num2str(ls_mse));
    disp("bagged_mse: " + num2str(bagged_mse));
    disp("ada_mse: " + num2str(ada_mse));
    disp("weighted_mse: " + num2str(weighted_mse));
    disp("weighted_percent: " + num2str(weighted_percent));
    
    %break;
    pause;
    close all;
end