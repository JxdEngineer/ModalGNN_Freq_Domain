%% modal identification with FDD - complete measurements
clc
clear
close all
load dataset1_test_time.mat

Phi_MAC = zeros(100,4);
Phi_MAE = zeros(100,4);
Freq_RE = zeros(100,4);
Zeta_RE = zeros(100,4);

fs = 200;
t = 1/fs*[0:length(acceleration_time_out{1}(1,:))-1];

tic
for i = 1:length(damping_out)
    disp(['i=',num2str(i)])
    acceleration_time_out_lowpass = lowpass(acceleration_time_out{i}',30,fs,ImpulseResponse="iir",Steepness=0.95)';  % narrow down the frequency bandwidth
    [phi,freq,zeta] = AFDD(acceleration_time_out_lowpass,t,15,'PickingMethod','auto'); % identify more modes for redundancy

    Phi_id = phi(1:4,:)';
    Zeta_id = zeta(1:4);
    Freq_id = freq(1:4);

    Freq_true = frequency_out{i}(1:4);
    Zeta_true = damping_out{i}(1:4);
    Phi_true = modeshape_out{i}(:,1:4);

    for j = 1:4
        Phi_MAC(i,j) = MAC(abs(real(Phi_id(:,j))),abs(Phi_true(:,j)));
        Phi_MAE(i,j) = mean(abs(abs(real(Phi_id(:,j)))-abs(Phi_true(:,j))));
        Zeta_RE(i,j) = (Zeta_id(j)-Zeta_true(j))/Zeta_true(j)*100;
        Freq_RE(i,j) = (Freq_id(j)-Freq_true(j))/Freq_true(j)*100;
    end
end
toc

clear statistics
for i = 1:4
    statistics(i,:) = [mean(Phi_MAC(:,i)),std(Phi_MAC(:,i)),min(Phi_MAC(:,i)),...
        mean(Phi_MAE(:,i)),std(Phi_MAE(:,i)),max(Phi_MAE(:,i)),...
        mean(Zeta_RE(:,i)),std(Zeta_RE(:,i)),max(Zeta_RE(:,i)),...
        mean(Freq_RE(:,i)),std(Freq_RE(:,i)),max(Freq_RE(:,i))];
end
statistics = statistics;
%% modal identification with FDD - incomplete measurements
clc
clear
close all
load dataset1_test_time.mat

Phi_MAC = zeros(100,4);
Phi_MAE = zeros(100,4);
Freq_RE = zeros(100,4);
Zeta_RE = zeros(100,4);

fs = 200;
t = 1/fs*[0:length(acceleration_time_out{1}(1,:))-1];

tic
for i = 1:length(damping_out)
    disp(['i=',num2str(i)])

    node_N = length(acceleration_time_out{i}(:,1));
    % ignore some node acceleration, only 18% remains
    node_mask = ones(node_N,1);
    missing_indices = 1:2:node_N;
    node_mask(missing_indices) = 0;
    missing_indices = 1:3:node_N;
    node_mask(missing_indices) = 0;
    missing_indices = 2:3:node_N;
    node_mask(missing_indices) = 0;
    node_mask = logical(node_mask);
    acceleration_incomplete = acceleration_time_out{i}(node_mask,:);

    acceleration_time_out_lowpass = lowpass(acceleration_incomplete',30,fs,ImpulseResponse="iir",Steepness=0.95)';  % narrow down the frequency bandwidth
    [phi,freq,zeta] = AFDD(acceleration_time_out_lowpass,t,15,'PickingMethod','auto'); % identify more modes for redundancy
    Phi_id_incomplete = phi(1:4,:)';
    Zeta_id = zeta(1:4);
    Freq_id = freq(1:4);

    % use inter/extrapolation to recover complete mode shapes on every node
    Phi_id = zeros(node_N,4);
    node_incomplete = node_out{i}(node_mask,:);
    for j = 1:4
        F = scatteredInterpolant(node_incomplete(:,1),...
            node_incomplete(:,2),Phi_id_incomplete(:,j),'natural','linear');
        Phi_id(:,j) = F(node_out{i}(:,1),node_out{i}(:,2));
    end
    
    Phi_true = modeshape_out{i}(:,1:4);
    Zeta_true = damping_out{i}(1:4);
    Freq_true = frequency_out{i}(1:4);

    for j = 1:4
        Phi_MAC(i,j) = MAC(abs(real(Phi_id(:,j))),abs(Phi_true(:,j)));
        Phi_MAE(i,j) = mean(abs(abs(real(Phi_id(:,j)))-abs(Phi_true(:,j))));
        Zeta_RE(i,j) = (Zeta_id(j)-Zeta_true(j))/Zeta_true(j)*100;
        Freq_RE(i,j) = (Freq_id(j)-Freq_true(j))/Freq_true(j)*100;
    end
end
toc

clear statistics
for i = 1:4
    statistics(i,:) = [mean(Phi_MAC(:,i)),std(Phi_MAC(:,i)),min(Phi_MAC(:,i)),...
        mean(Phi_MAE(:,i)),std(Phi_MAE(:,i)),max(Phi_MAE(:,i)),...
        mean(Zeta_RE(:,i)),std(Zeta_RE(:,i)),max(Zeta_RE(:,i)),...
        mean(Freq_RE(:,i)),std(Freq_RE(:,i)),max(Freq_RE(:,i))];
end
statistics = statistics;
%% modal identification with FDD - complete measurements + 10% noise
clc
clear
close all
load dataset1_test_time.mat

Phi_MAC = zeros(100,4);
Phi_MAE = zeros(100,4);
Freq_RE = zeros(100,4);
Zeta_RE = zeros(100,4);

fs = 200;
t = 1/fs*[0:length(acceleration_time_out{1}(1,:))-1];

tic
for i = 1:length(damping_out)
    disp(['i=',num2str(i)])
    acceleration_time_out_lowpass = lowpass(acceleration_time_out{i}',30,fs,ImpulseResponse="iir",Steepness=0.95)';  % narrow down the frequency bandwidth

    for j = 1:length(node_out{i})
        % add noises
        signal_power = mean(acceleration_time_out_lowpass(j,:).^2);
        noise_power = 0.1 * signal_power; % Power of the noise (% of signal power)
        noise_std = sqrt(noise_power);    % Standard deviation of the noise
        noise = noise_std * randn(size(acceleration_time_out_lowpass(j,:)));
        acceleration_time_out_lowpass(j,:) = acceleration_time_out_lowpass(j,:) + noise;
    end

    [phi,freq,zeta] = AFDD(acceleration_time_out_lowpass,t,15,'PickingMethod','auto'); % identify more modes for redundancy

    Phi_id = phi(1:4,:)';
    Zeta_id = zeta(1:4);
    Freq_id = freq(1:4);

    Freq_true = frequency_out{i}(1:4);
    Zeta_true = damping_out{i}(1:4);
    Phi_true = modeshape_out{i}(:,1:4);

    for j = 1:4
        Phi_MAC(i,j) = MAC(abs(real(Phi_id(:,j))),abs(Phi_true(:,j)));
        Phi_MAE(i,j) = mean(abs(abs(real(Phi_id(:,j)))-abs(Phi_true(:,j))));
        Freq_RE(i,j) = (Freq_id(j)-Freq_true(j))/Freq_true(j)*100;
        Zeta_RE(i,j) = (Zeta_id(j)-Zeta_true(j))/Zeta_true(j)*100;
    end
end
toc

clear statistics
for i = 1:4
    statistics(i,:) = [mean(Phi_MAC(:,i)),std(Phi_MAC(:,i)),min(Phi_MAC(:,i)),...
        mean(Phi_MAE(:,i)),std(Phi_MAE(:,i)),max(Phi_MAE(:,i)),...
        mean(Zeta_RE(:,i)),std(Zeta_RE(:,i)),max(Zeta_RE(:,i)),...
        mean(Freq_RE(:,i)),std(Freq_RE(:,i)),max(Freq_RE(:,i))];
end
statistics = statistics;