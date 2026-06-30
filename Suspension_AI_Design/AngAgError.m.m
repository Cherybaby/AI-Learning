% 位置传感器测角误差谐波分析（数值方法，避免符号积分卡死）
% 参数：gs, gc - sin/cos 增益误差; os, oc - 偏置; phi - 正交误差

clear; clc;

% 设定参数数值（可修改）
gs_val = 0.05;      % sin 增益误差 5%
gc_val = 0.03;      % cos 增益误差 3%
os_val = 0.02;      % sin 偏置
oc_val = 0.01;      % cos 偏置
phi_val = 0.05;     % 正交误差 0.05 rad (~2.9°)

% theta 从 0 到 2π，采样 10000 点
theta = linspace(0, 2*pi, 10000)';

% 信号模型
s = (1 + gs_val)*sin(theta) + os_val;
c = (1 + gc_val)*cos(theta + phi_val) + oc_val;

% 测得角
phimeas = atan2(s, c);

% 真实角
phitrue = theta;

% 误差
err = phimeas - phitrue;

% 用 FFT 计算谐波系数（直流分量和前 N 个谐波）
N = 5;  % 计算前 5 阶谐波
fft_err = fft(err);
fft_len = length(err);

% 计算傅里叶系数（实部和虚部对应 cos 和 sin）
A0 = 2*real(fft_err(1))/fft_len;  % 直流项
An = zeros(N, 1);
Bn = zeros(N, 1);
for n = 1:N
    An(n) = 2*real(fft_err(n+1))/fft_len;
    Bn(n) = -2*imag(fft_err(n+1))/fft_len;
end

% 输出结果
fprintf('=== 位置传感器测角误差谐波分析 ===\n');
fprintf('参数设置：gs=%.4f, gc=%.4f, os=%.4f, oc=%.4f, phi=%.4f\n\n', ...
    gs_val, gc_val, os_val, oc_val, phi_val);

fprintf('直流项 A0 = %.6e\n\n', A0);

fprintf('谐波系数表：\n');
fprintf('n\t An (cos)\t Bn (sin)\t 幅值\t 相位(rad)\n');
fprintf('--\t --------\t --------\t -----\t ---------\n');
for n = 1:N
    mag = sqrt(An(n)^2 + Bn(n)^2);
    phase = atan2(Bn(n), An(n));
    fprintf('%d\t %.6e\t %.6e\t %.6e\t %.6e\n', n, An(n), Bn(n), mag, phase);
end

% 判定主要谐波
fprintf('\n=== 主要谐波分量 ===\n');
mag_threshold = max(sqrt(An.^2 + Bn.^2)) * 0.01;  % 取最大幅值的 1% 作为阈值
for n = 1:N
    mag = sqrt(An(n)^2 + Bn(n)^2);
    if mag > mag_threshold
        fprintf('第 %d 阶谐波存在（幅值=%.6e）\n', n, mag);
    end
end

% 绘制误差波形和频谱
figure('Position', [100 100 1200 400]);

subplot(1,2,1);
plot(theta, err*1e3, 'b', 'LineWidth', 1.5);
xlabel('角度 θ (rad)'); ylabel('测角误差 (mrad)');
title('测角误差波形');
grid on;

subplot(1,2,2);
harmonics = 1:N;
mag_vals = sqrt(An.^2 + Bn.^2);
bar(harmonics, mag_vals*1e6);
xlabel('谐波阶次 n'); ylabel('幅值 (μrad)');
title('谐波频谱');
grid on;

fprintf('\n已绘制误差波形和频谱。\n');