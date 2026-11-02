%% Задание 4: полёт шара
clear; clc; close all;
cd('C:\Users\reino\Documents\MATLAB')

alpha_deg = 30;                      % угол в градусах
v0 = 10;

alpha = pi*alpha_deg/180;
vx0 = v0*cos(alpha);
vy0 = v0*sin(alpha);

% В модель IC интеграторов должны быть заданы через переменные vx0, vy0!
% (см. пункт 7)
assignin('base', 'vx0', vx0);
assignin('base', 'vy0', vy0);

simOut = sim('fourth_lab', 'StopTime', '1');

t  = simOut.tout;
x  = get_sig(simOut, 'x');
y  = get_sig(simOut, 'y');

% --- График дальности ---
figure('Color','w')
plot(t, x, 'LineWidth', 2); grid on
xlabel('t, с'); ylabel('x(t), м')
title(sprintf('Дальность x(t), \\alpha = %d°', alpha_deg))

% --- График высоты ---
figure('Color','w')
plot(t, y, 'LineWidth', 2); grid on
xlabel('t, с'); ylabel('y(t), м')
title(sprintf('Высота y(t), \\alpha = %d°', alpha_deg))

% --- Траектория ---
figure('Color','w')
plot(x, y, 'LineWidth', 2); grid on
xlabel('x, м'); ylabel('y, м')
title(sprintf('Траектория, \\alpha = %d°', alpha_deg))
axis equal

%% вспомогательная функция
function v = get_sig(simOut, name)
    if any(strcmp(simOut.who, name))
        v = simOut.get(name);
    else
        v = evalin('base', name);
    end
    if isa(v,'timeseries'), v = v.Data; end
    v = v(:);
end