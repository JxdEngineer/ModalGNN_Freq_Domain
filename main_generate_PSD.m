%% load data
clc
clear
load dataset1_test_time.mat
dt = 1/200;
%% generate PSD features with pwelch
tic
nfft = 1024*2^1;
window = hamming(nfft/4);
acceleration_pwelch = cell(length(acceleration_time_out),1);
for i = 1:length(acceleration_time_out)
    acceleration_pwelch{i} = zeros(length(node_out{i}),nfft/2+1);
    for j = 1:length(node_out{i})
        % add noises
        signal_power = mean(acceleration_time_out{i}(j,:).^2);
        noise_power = 0.1 * signal_power; % Power of the noise (% of signal power)
        noise_std = sqrt(noise_power);    % Standard deviation of the noise
        noise = noise_std * randn(size(acceleration_time_out{i}(j,:)));
        acceleration_time_out{i}(j,:) = acceleration_time_out{i}(j,:) + noise;

        [psd,f] = pwelch(acceleration_time_out{i}(j,:),window,[],nfft,1/dt);
        acceleration_pwelch{i}(j,:) = psd;
    end
    acceleration_pwelch{i}(isnan(acceleration_pwelch{i})) = 0;
    acceleration_pwelch{i} = acceleration_pwelch{i}/max(max(acceleration_pwelch{i})); % normalization
    disp(['i=',num2str(i)])
end

save dataset node_out frequency_out modeshape_out acceleration_pwelch element_out damping_out f % output time-history acceleration
toc