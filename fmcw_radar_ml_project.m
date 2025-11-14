%% ========================================================================
% FMCW Radar + ML Target Classification (Software-Only Project)
% Author: Brian Rono
%
% This script simulates an FMCW radar front-end in baseband, generates a
% range–Doppler map, and then builds a simple ML classifier that
% distinguishes between three target types (human, car, drone) using
% simulated Doppler/velocity profiles.
%
% Dependencies:
%   - Base MATLAB
%   - Recommended: Signal Processing Toolbox
%   - Recommended: Statistics & Machine Learning Toolbox
% ========================================================================

clear; close all; clc;

%% ------------------------ 1. Radar Parameters ---------------------------
c  = 3e8;         % speed of light (m/s)
fc = 77e9;        % carrier frequency (Hz) used in automotive radar
lambda = c/fc;    % wavelength

% FMCW waveform parameters
B      = 150e6;       % sweep bandwidth (Hz)
Tchirp = 40e-6;       % chirp duration (s)
slope  = B / Tchirp;  % chirp slope (Hz/s)

% Sampling / frame parameters
Fs      = 2e6;                      % baseband sampling frequency (Hz)
Ns      = round(Tchirp * Fs);       % samples per chirp   (fast-time)
Nchirps = 128;                      % number of chirps    (slow-time)

t_fast = (0:Ns-1) / Fs;             % fast-time index within one chirp
t_slow = (0:Nchirps-1) * Tchirp;    % slow-time index across chirps

fprintf('Fast-time samples per chirp: %d\n', Ns);
fprintf('Number of chirps per frame: %d\n', Nchirps);

%% ------------------------ 2. Target Definition --------------------------
% Each row: [range (m), radial velocity (m/s), amplitude]
% Example scenario: one stationary target and two moving targets.
targets = [ ...
    40,   0,    1.0;   % target 1: stationary at 40 m
    60,   15,   0.8;   % target 2: receding at +15 m/s
    80,  -30,   0.6];  % target 3: approaching at -30 m/s

Nt = size(targets,1);

%% ------------------------ 3. Transmit Signal Generation -----------------
% Build the full frame of transmitted FMCW chirps (Nchirps x Ns)
tx = zeros(Nchirps, Ns);

for k = 1:Nchirps
    t = t_fast + (k-1)*Tchirp;   % absolute time for chirp k
    tx(k,:) = exp(1j*2*pi*(fc*t + 0.5*slope*t.^2));
end

%% ------------------------ 4. Received Signal Simulation -----------------
% Sum of delayed and Doppler-shifted replicas for all targets.
rx = zeros(Nchirps, Ns);

for ti = 1:Nt
    R0   = targets(ti,1);   % initial range
    vel  = targets(ti,2);   % radial velocity
    amp  = targets(ti,3);   % complex amplitude

    % Constant round-trip delay for this range
    tau  = 2*R0 / c;

    % Doppler frequency due to radial velocity
    fd   = 2*vel / lambda;  % Hz

    for k = 1:Nchirps
        t = t_fast + (k-1)*Tchirp;

        % Time shift due to range and phase ramp due to Doppler
        t_delayed = t - tau;

        % Keep only physically valid samples
        valid = t_delayed >= 0;

        % Baseband received FMCW from this target
        s_rx = zeros(size(t));
        s_rx(valid) = amp * exp(1j*2*pi*( ...
            fc*t_delayed(valid) + 0.5*slope*t_delayed(valid).^2 + fd*t(valid)));

        rx(k,:) = rx(k,:) + s_rx;
    end
end

% Add white Gaussian noise for a chosen SNR
SNR_dB = 20;
rx = awgn(rx, SNR_dB, 'measured');

%% ------------------------ 5. De-chirp (Mixing) -------------------------
% Beat signal obtained by mixing received signal with transmitted chirp.
mix = tx .* conj(rx);

%% ------------------------ 6. Range FFT (fast-time) ---------------------
Nfft_range = 2^nextpow2(Ns);
% FFT across fast-time dimension (columns)
range_fft = fft(mix, Nfft_range, 2);

% Keep only non-negative beat frequencies
range_fft = range_fft(:, 1:Nfft_range/2);

% Range resolution and axis
R_res      = c / (2 * B);
range_axis = (0:(Nfft_range/2-1)) * R_res;

%% ------------------------ 7. Doppler FFT (slow-time) -------------------
Nfft_dopp = 2^nextpow2(Nchirps);
% FFT across slow-time dimension (rows) and shift to center zero Doppler
rdm = fftshift(fft(range_fft, Nfft_dopp, 1), 1);

% Doppler and velocity axes (approximate)
PRF     = 1 / Tchirp;
fd_axis = (-Nfft_dopp/2 : Nfft_dopp/2-1) * (PRF / Nfft_dopp);
vel_axis = (fd_axis * lambda) / 2;

%% ------------------------ 8. Range–Doppler Map Plot --------------------
figure;
imagesc(range_axis, vel_axis, 20*log10(abs(rdm)));
axis xy;
xlabel('Range (m)');
ylabel('Velocity (m/s)');
title('Range–Doppler Map (FMCW Radar)');
colorbar;

%% ------------------------ 9. Micro-Doppler-Type Signature --------------
% Select a strong range bin and examine Doppler behaviour over slow-time.
[~, max_idx] = max(sum(abs(range_fft).^2,1));
selected_range = range_axis(max_idx);
fprintf('Selected range bin around %.2f m for micro-Doppler view\n', selected_range);

% Slow-time signal at the chosen range bin
s_slow = squeeze(range_fft(:, max_idx));  % Nchirps x 1

% STFT / spectrogram over slow-time
figure;
spectrogram(s_slow, 32, 24, 128, PRF, 'yaxis');
title(sprintf('Micro-Doppler Signature at Range ~ %.1f m', selected_range));
ylabel('Frequency (Hz) ~ Doppler');
xlabel('Time (s)');

%% ========================================================================
%  PART B: RF + ML CLASSIFICATION (Human vs Car vs Drone)
%
%  Each class is represented by a different velocity profile as a function
%  of slow-time. Features derived from the velocity profile are used to
%  train a simple multi-class classifier.
% ========================================================================

%% ------------------------ 10. Class Setup ------------------------------
classes = {'Human', 'Car', 'Drone'};
Nc = numel(classes);

Nexamples_per_class = 40;             % number of synthetic examples per class
Nexamples = Nexamples_per_class * Nc;

% Feature matrix and label vector
% Features: [mean_vel, std_vel, max_vel, mean_abs_derivative]
X = zeros(Nexamples, 4);
Y = strings(Nexamples,1);

example_idx = 1;

%% ------------------------ 11. Doppler/Velocity Simulation --------------
% Wrapper for generating a velocity profile for a given class name.
simulate_velocity_profile = @(className, t) ...
    simulate_velocity_pattern(className, t);

% For ML, it is sufficient to model Doppler using velocity over slow-time
% instead of re-running the full FMCW simulation for every example.
for ci = 1:Nc
    cname = classes{ci};

    for n = 1:Nexamples_per_class

        % Observation interval based on the number of chirps
        Tobs = Nchirps * Tchirp;
        t_md = linspace(0, Tobs, Nchirps);

        % Velocity and corresponding Doppler profiles
        vel_profile = simulate_velocity_profile(cname, t_md);   % m/s
        fd_profile  = 2 * vel_profile / lambda;                 % Hz

        % Complex Doppler signal across slow-time (phase accumulation model)
        s_md = exp(1j*2*pi*cumsum(fd_profile)/PRF);

        %#ok<NASGU> % s_md kept for potential future feature design

        % Feature extraction from velocity profile
        mean_v = mean(vel_profile);
        std_v  = std(vel_profile);
        max_v  = max(abs(vel_profile));
        mean_dv = mean(abs(diff(vel_profile))) / (Tobs/Nchirps);

        X(example_idx,:) = [mean_v, std_v, max_v, mean_dv];
        Y(example_idx)   = string(cname);

        example_idx = example_idx + 1;
    end
end

%% ------------------------ 12. Train/Test Split -------------------------
rng(1);  % for reproducible shuffling
idx         = randperm(Nexamples);
train_ratio = 0.7;
Ntrain      = round(train_ratio * Nexamples);

Xtrain = X(idx(1:Ntrain), :);
Ytrain = Y(idx(1:Ntrain), :);
Xtest  = X(idx(Ntrain+1:end), :);
Ytest  = Y(idx(Ntrain+1:end), :);

%% ------------------------ 13. Classifier Training ----------------------
% Multi-class SVM using fitcecoc (requires Statistics & ML Toolbox)
Mdl = fitcecoc(Xtrain, Ytrain);

Ypred = predict(Mdl, Xtest);
acc = mean(Ypred == Ytest);

fprintf('ML classification accuracy: %.2f %%\n', acc*100);

%% ------------------------ 14. Feature-Space Visualisation --------------
figure;
gscatter(X(:,1), X(:,3), Y);
xlabel('Mean velocity (m/s)');
ylabel('Max velocity (m/s)');
title('Feature Space: Mean vs. Max Velocity (Human / Car / Drone)');
grid on;

%% ------------------------ Helper Function Definition -------------------
% Helper function must appear at the end of the script in MATLAB.

function v = simulate_velocity_pattern(className, t)
%SIMULATE_VELOCITY_PATTERN  Generate a time-varying velocity profile (m/s)
% for different target types: 'Human', 'Car', and 'Drone'.
%
% Input:
%   className : string or char array specifying the target type
%   t         : time vector (s)
%
% Output:
%   v         : velocity as a function of time (same size as t)

    t = t(:)';                     % ensure row vector
    T = t(end) - t(1);
    if T <= 0
        v = zeros(size(t));
        return;
    end

    switch lower(className)
        case 'human'
            % Human walking: low average speed with significant micro-motion
            v_mean    = 1 + 0.5*randn;        % around 1 m/s
            step_freq = 1.5 + 0.3*randn;      % step frequency (Hz)
            micro_amp = 0.5 + 0.2*randn;      % leg motion amplitude
            v = v_mean + micro_amp .* sin(2*pi*step_freq*t) ...
                + 0.2*randn(size(t));         % random perturbations

        case 'car'
            % Car: higher mean speed with relatively smooth profile
            v_mean = 15 + 5*randn;            % around 15 m/s
            accel  = (randn * 1);             % small acceleration
            v = v_mean + accel*(t - t(1)) + 0.5*randn(size(t));

            % Avoid negative speeds for this simple model
            v(v < 0) = 0;

        case 'drone'
            % Drone: moderate speed with oscillatory "hovering" motion
            v_mean     = 5 + 2*randn;         % around 5 m/s
            hover_freq = 3 + rand;            % hover oscillation (Hz)
            hover_amp  = 2 + rand;            % oscillation amplitude
            v = v_mean + hover_amp .* sin(2*pi*hover_freq*t) ...
                + 1.0*randn(size(t));

        otherwise
            % Default case: zero velocity
            v = zeros(size(t));
    end
end
