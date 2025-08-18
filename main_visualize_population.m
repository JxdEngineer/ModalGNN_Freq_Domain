%% plot the distributions of frequencies
clc
clear
close all


% load population 1, training set
load dataset1_train_valid_PSD.mat
N_train_valid_p1 = length(node_out);
Freq_train_valid_p1 = zeros(N_train_valid_p1,4);
for i = 1:N_train_valid_p1
    Freq_train_valid_p1(i,:) = frequency_out{i}(1:4);
end

% load population 1, testing set
load dataset1_test_PSD.mat
N_test_p1 = length(node_out);
Freq_test_p1 = zeros(N_test_p1,4);
for i = 1:N_test_p1
    Freq_test_p1(i,:) = frequency_out{i}(1:4);
end

% load population 2, testing set
load dataset2_test_PSD.mat
N_test_p2 = length(node_out);
Freq_test_p2 = zeros(N_test_p2,4);
for i = 1:N_test_p2
    Freq_test_p2(i,:) = frequency_out{i}(1:4);
end


fontsize = 10;
BW = 0.25; % bin width for histogram
% plot histograms of population 1
figure
hold on
% plot train and validation
histogram(Freq_train_valid_p1(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.25);
histogram(Freq_train_valid_p1(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.25);
histogram(Freq_train_valid_p1(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.25);
histogram(Freq_train_valid_p1(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.25);
% plot test
histogram(Freq_test_p1(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.5,'LineStyle',':');
histogram(Freq_test_p1(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.5,'LineStyle',':');
histogram(Freq_test_p1(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.5,'LineStyle',':');
histogram(Freq_test_p1(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.5,'LineStyle',':');
xlabel('Frequency (Hz)')
ylabel('Probability')
legend('Mode1-train-pop1', 'Mode2-train-pop1', 'Mode3-train-pop1', 'Mode4-train-pop1', ...
    'Mode1-test-pop1', 'Mode2-test-pop1', 'Mode3-test-pop1', 'Mode4-test-pop1')
set(gcf,...
    'Unit', 'Centimeter', ...
    'Position', [2, 2, 7, 7])
set(gca,...
    'FontName', 'Times New Roman', ...
    'FontSize', fontsize, ...
    'YDir','normal',...
    'Box', 'On', ...
    'Xgrid', 'On', ...
    'Ygrid', 'On', ...
    'Xlim', [0,25],...
    'Ylim', [0,0.6],...
    'TickDir', 'In', ...
    'TickLength', [0.01 0.01])

% plot histograms of population 2
figure
hold on
% plot train and validation
histogram(Freq_train_valid_p1(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.25);
histogram(Freq_train_valid_p1(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.25);
histogram(Freq_train_valid_p1(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.25);
histogram(Freq_train_valid_p1(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.25);
% plot test
histogram(Freq_test_p2(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.5,'LineStyle',':');
histogram(Freq_test_p2(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.5,'LineStyle',':');
histogram(Freq_test_p2(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.5,'LineStyle',':');
histogram(Freq_test_p2(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.5,'LineStyle',':');
xlabel('Frequency (Hz)')
ylabel('Probability')
legend('Mode1-train-pop1', 'Mode2-train-pop1', 'Mode3-train-pop1', 'Mode4-train-pop1', ...
    'Mode1-test-pop2', 'Mode2-test-pop2', 'Mode3-test-pop2', 'Mode4-test-pop2')
set(gcf,...
    'Unit', 'Centimeter', ...
    'Position', [2, 2, 7, 7])
set(gca,...
    'FontName', 'Times New Roman', ...
    'FontSize', fontsize, ...
    'YDir','normal',...
    'Box', 'On', ...
    'Xgrid', 'On', ...
    'Ygrid', 'On', ...
    'Xlim', [0,25],...
    'Ylim', [0,0.6],...
    'TickDir', 'In', ...
    'TickLength', [0.01 0.01])
%% plot distributions of damping ratios
clc
clear
close all


% load population 1, training set
load dataset1_train_valid_PSD.mat
N_train_valid_p1 = length(node_out);
Zeta_train_valid_p1 = zeros(N_train_valid_p1,4);
for i = 1:N_train_valid_p1
    Zeta_train_valid_p1(i,:) = damping_out{i}(1:4)*100;
end

% load population 1, testing set
load dataset1_test_PSD.mat
N_test_p1 = length(node_out);
Zeta_test_p1 = zeros(N_test_p1,4);
for i = 1:N_test_p1
    Zeta_test_p1(i,:) = damping_out{i}(1:4)*100;
end

% load population 2, testing set
load dataset2_test_PSD.mat
N_test_p2 = length(node_out);
Zeta_test_p2 = zeros(N_test_p2,4);
for i = 1:N_test_p2
    Zeta_test_p2(i,:) = damping_out{i}(1:4)*100;
end


fontsize = 10;
BW = 0.01; % bin width for histogram
% plot histograms of population 1
figure
hold on
% plot train and validation
histogram(Zeta_train_valid_p1(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.25);
histogram(Zeta_train_valid_p1(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.25);
histogram(Zeta_train_valid_p1(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.25);
histogram(Zeta_train_valid_p1(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.25);
% plot test
histogram(Zeta_test_p1(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.5,'LineStyle',':');
histogram(Zeta_test_p1(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.5,'LineStyle',':');
histogram(Zeta_test_p1(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.5,'LineStyle',':');
histogram(Zeta_test_p1(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.5,'LineStyle',':');
xlabel('Damping ratio (%)')
ylabel('Probability')
legend('Mode1-train-pop1', 'Mode2-train-pop1', 'Mode3-train-pop1', 'Mode4-train-pop1', ...
    'Mode1-test-pop1', 'Mode2-test-pop1', 'Mode3-test-pop1', 'Mode4-test-pop1')
set(gcf,...
    'Unit', 'Centimeter', ...
    'Position', [2, 2, 7, 7])
set(gca,...
    'FontName', 'Times New Roman', ...
    'FontSize', fontsize, ...
    'YDir','normal',...
    'Box', 'On', ...
    'Xgrid', 'On', ...
    'Ygrid', 'On', ...
    'Xlim', [0.2,1.1],...
    'TickDir', 'In', ...
    'TickLength', [0.01 0.01])

% plot histograms of population 2
figure
hold on
% plot train and validation
histogram(Zeta_train_valid_p1(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.25);
histogram(Zeta_train_valid_p1(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.25);
histogram(Zeta_train_valid_p1(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.25);
histogram(Zeta_train_valid_p1(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.25);
% plot test
histogram(Zeta_test_p2(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.5,'LineStyle',':');
histogram(Zeta_test_p2(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.5,'LineStyle',':');
histogram(Zeta_test_p2(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.5,'LineStyle',':');
histogram(Zeta_test_p2(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.5,'LineStyle',':');
xlabel('Damping ratio (%)')
ylabel('Probability')
legend('Mode1-train-pop1', 'Mode2-train-pop1', 'Mode3-train-pop1', 'Mode4-train-pop1', ...
    'Mode1-test-pop2', 'Mode2-test-pop2', 'Mode3-test-pop2', 'Mode4-test-pop2')
set(gcf,...
    'Unit', 'Centimeter', ...
    'Position', [2, 2, 7, 7])
set(gca,...
    'FontName', 'Times New Roman', ...
    'FontSize', fontsize, ...
    'YDir','normal',...
    'Box', 'On', ...
    'Xgrid', 'On', ...
    'Ygrid', 'On', ...
    'Xlim', [0.2,1.1],...
    'TickDir', 'In', ...
    'TickLength', [0.01 0.01])
%% plot distributions of mode shapes
clc
clear


% load population 1, training set
load dataset1_train_valid_PSD.mat
N_train_valid_p1 = length(node_out);
Phi_train_valid_p1 = zeros(N_train_valid_p1,4);
modeshape_out_train_valid_p1 = modeshape_out;
for i = 1:N_train_valid_p1
    % Phi_train_valid_p1(i,:) = sum(diff(modeshape_out{i}(node_out{i}(:,2)==0,1:4),2,1).^2);
    Phi_train_valid_p1(i,:) = sum(modeshape_out{i}(:,1:4).^2); % energy of mode shapes
    % Phi_train_valid_p1(i,:) = std(abs(modeshape_out{i}(:,1:4))); % std of mode shapes
    for j = 1:4
        % Phi_train_valid_p1(i,j) = max(xcorr(modeshape_out{i}(:,j),modeshape_out_train_valid_p1{1}(:,j))); % use cross-correlation to measure similarity
        % Phi_train_valid_p1(i,j) = dtw(modeshape_out{i}(:,j),modeshape_out_train_valid_p1{1}(:,j)); % use dtw to measure similarity
    end
end

% load population 1, testing set
load dataset1_test_PSD.mat
N_test_p1 = length(node_out);
Phi_test_p1 = zeros(N_test_p1,4);
for i = 1:N_test_p1
    % Phi_test_p1(i,:) = sum(diff(modeshape_out{i}(node_out{i}(:,2)==0,1:4),2,1).^2);
    Phi_test_p1(i,:) = sum(modeshape_out{i}(:,1:4).^2);
    % Phi_test_p1(i,:) = std(abs(modeshape_out{i}(:,1:4))); % std of mode shapes
    for j = 1:4
        % Phi_test_p1(i,j) = max(xcorr(modeshape_out{i}(:,j),modeshape_out_train_valid_p1{1}(:,j)));
        % Phi_test_p1(i,j) = dtw(modeshape_out{i}(:,j),modeshape_out_train_valid_p1{1}(:,j)); % use dtw to measure similarity
    end
end

% load population 2, testing set
load dataset2_test_PSD.mat
N_test_p2 = length(node_out);
Phi_test_p2 = zeros(N_test_p2,4);
for i = 1:N_test_p2
    % Phi_test_p2(i,:) = sum(diff(modeshape_out{i}(node_out{i}(:,2)==0,1:4),2,1).^2);
    Phi_test_p2(i,:) = sum(modeshape_out{i}(:,1:4).^2);
    % Phi_test_p2(i,:) = std(abs(modeshape_out{i}(:,1:4))); % std of mode shapes
    for j = 1:4
        % Phi_test_p2(i,j) = max(xcorr(modeshape_out{i}(:,j),modeshape_out_train_valid_p1{1}(:,j)));
        % Phi_test_p2(i,j) = dtw(modeshape_out{i}(:,j),modeshape_out_train_valid_p1{1}(:,j)); % use dtw to measure similarity
    end
end

close all
fontsize = 9;
BW = 1; % bin width for histogram
% plot histograms of population 1
figure
hold on
% plot train and validation
histogram(Phi_train_valid_p1(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.25);
histogram(Phi_train_valid_p1(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.25);
histogram(Phi_train_valid_p1(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.25);
histogram(Phi_train_valid_p1(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.25);
% plot test
histogram(Phi_test_p1(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.5,'LineStyle',':');
histogram(Phi_test_p1(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.5,'LineStyle',':');
histogram(Phi_test_p1(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.5,'LineStyle',':');
histogram(Phi_test_p1(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.5,'LineStyle',':');
xlabel('sum(|mode shape|^2)')
ylabel('Probability')
legend('Mode1-train-pop1', 'Mode2-train-pop1', 'Mode3-train-pop1', 'Mode4-train-pop1', ...
    'Mode1-test-pop1', 'Mode2-test-pop1', 'Mode3-test-pop1', 'Mode4-test-pop1')
set(gcf,...
    'Unit', 'Centimeter', ...
    'Position', [2, 2, 7, 7])
set(gca,...
    'FontName', 'Times New Roman', ...
    'FontSize', fontsize, ...
    'YDir','normal',...
    'Box', 'On', ...
    'Xgrid', 'On', ...
    'Ygrid', 'On', ...
    'Xlim', [0,50],...
    'Ylim', [0,0.3],...
    'TickDir', 'In', ...
    'TickLength', [0.01 0.01])

% plot histograms of population 2
figure
hold on
% plot train and validation
histogram(Phi_train_valid_p1(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.25);
histogram(Phi_train_valid_p1(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.25);
histogram(Phi_train_valid_p1(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.25);
histogram(Phi_train_valid_p1(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.25);
% plot test
histogram(Phi_test_p2(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.5,'LineStyle',':');
histogram(Phi_test_p2(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.5,'LineStyle',':');
histogram(Phi_test_p2(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.5,'LineStyle',':');
histogram(Phi_test_p2(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.5,'LineStyle',':');
xlabel('sum(|mode shape|^2)')
ylabel('Probability')
legend('Mode1-train-pop1', 'Mode2-train-pop1', 'Mode3-train-pop1', 'Mode4-train-pop1', ...
    'Mode1-test-pop2', 'Mode2-test-pop2', 'Mode3-test-pop2', 'Mode4-test-pop2')
set(gcf,...
    'Unit', 'Centimeter', ...
    'Position', [2, 2, 7, 7])
set(gca,...
    'FontName', 'Times New Roman', ...
    'FontSize', fontsize, ...
    'YDir','normal',...
    'Box', 'On', ...
    'Xgrid', 'On', ...
    'Ygrid', 'On', ...
    'Xlim', [0,50],...
    'Ylim', [0,0.3],...
    'TickDir', 'In', ...
    'TickLength', [0.01 0.01])
%% plot mode shapes
clc
clear
close all

% load population 1, training set
load dataset1_train_valid_PSD.mat
N_train_valid_p1 = length(node_out);
Phi_train_valid_p1 = cell(N_train_valid_p1,1);
for i = 1:N_train_valid_p1
    Phi_train_valid_p1{i} = abs(modeshape_out{i}(:,1:4)); 
end

% load population 1, testing set
load dataset1_test_PSD.mat
N_test_p1 = length(node_out);
Phi_test_p1 = cell(N_test_p1,1);
for i = 1:N_test_p1
    Phi_test_p1{i} = abs(modeshape_out{i}(:,1:4)); 
end

% load population 2, testing set
load dataset2_test_PSD.mat
N_test_p2 = length(node_out);
Phi_test_p2 = cell(N_test_p2,1);
for i = 1:N_test_p2
    Phi_test_p2{i} = abs(modeshape_out{i}(:,1:4)); 
end

nfft = 1024;
order = 20;
pxx1 = pburg(Phi_train_valid_p1{1}(:,1),order,nfft);
pxx2 = pburg(Phi_train_valid_p1{1}(:,2),order,nfft);
pxx3 = pburg(Phi_train_valid_p1{1}(:,3),order,nfft);
pxx4 = pburg(Phi_train_valid_p1{1}(:,4),order,nfft);
figure
hold on
plot(db(pxx1))
plot(db(pxx2))
plot(db(pxx3))
plot(db(pxx4))
legend('1','2','3','4')

fontsize = 9;
BW = 1; % bin width for histogram
figure
hold on
for i = 1:N_train_valid_p1
    plot(Phi_train_valid_p1{i}(:,1),'color','#FF1F5B')
end
for i = 1:N_test_p1
    plot(Phi_test_p1{i}(:,1),'color','#00CD6C')
end


figure
hold on
for i = 1:N_train_valid_p1
    plot(Phi_train_valid_p1{i}(:,1),'color','#FF1F5B')
end
for i = 1:N_test_p2
    plot(Phi_test_p2{i}(:,1),'color','#00CD6C')
end



% plot train and validation
histogram(Phi_train_valid_p1(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.25);
histogram(Phi_train_valid_p1(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.25);
histogram(Phi_train_valid_p1(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.25);
histogram(Phi_train_valid_p1(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.25);
% plot test
histogram(Phi_test_p1(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.5,'LineStyle',':');
histogram(Phi_test_p1(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.5,'LineStyle',':');
histogram(Phi_test_p1(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.5,'LineStyle',':');
histogram(Phi_test_p1(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.5,'LineStyle',':');
xlabel('sum(|mode shape|^2)')
ylabel('Probability')
legend('Mode1-train-pop1', 'Mode2-train-pop1', 'Mode3-train-pop1', 'Mode4-train-pop1', ...
    'Mode1-test-pop1', 'Mode2-test-pop1', 'Mode3-test-pop1', 'Mode4-test-pop1')
set(gcf,...
    'Unit', 'Centimeter', ...
    'Position', [2, 2, 7, 7])
set(gca,...
    'FontName', 'Times New Roman', ...
    'FontSize', fontsize, ...
    'YDir','normal',...
    'Box', 'On', ...
    'Xgrid', 'On', ...
    'Ygrid', 'On', ...
    'Xlim', [0,50],...
    'Ylim', [0,0.4],...
    'TickDir', 'In', ...
    'TickLength', [0.01 0.01])

% plot histograms of population 2
figure
hold on
% plot train and validation
histogram(Phi_train_valid_p1(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.25);
histogram(Phi_train_valid_p1(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.25);
histogram(Phi_train_valid_p1(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.25);
histogram(Phi_train_valid_p1(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.25);
% plot test
histogram(Phi_test_p2(:,1),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FF1F5B','LineWidth',1.5,'LineStyle',':');
histogram(Phi_test_p2(:,2),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#00CD6C','LineWidth',1.5,'LineStyle',':');
histogram(Phi_test_p2(:,3),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#FFC61E','LineWidth',1.5,'LineStyle',':');
histogram(Phi_test_p2(:,4),"BinWidth",BW,'Normalization','probability','DisplayStyle','stairs','EdgeColor','#AF58BA','LineWidth',1.5,'LineStyle',':');
xlabel('sum(|mode shape|^2)')
ylabel('Probability')
legend('Mode1-train-pop1', 'Mode2-train-pop1', 'Mode3-train-pop1', 'Mode4-train-pop1', ...
    'Mode1-test-pop2', 'Mode2-test-pop2', 'Mode3-test-pop2', 'Mode4-test-pop2')
set(gcf,...
    'Unit', 'Centimeter', ...
    'Position', [2, 2, 7, 7])
set(gca,...
    'FontName', 'Times New Roman', ...
    'FontSize', fontsize, ...
    'YDir','normal',...
    'Box', 'On', ...
    'Xgrid', 'On', ...
    'Ygrid', 'On', ...
    'Xlim', [0,50],...
    'Ylim', [0,0.4],...
    'TickDir', 'In', ...
    'TickLength', [0.01 0.01])