%% Задание 5: 3 модели маятника
clear; clc; close all;

model = 'fifth_lab';       
T = 10;                

% Начальные условия: отклонение pi/2, начальная скорость 0
x0  = pi/2;
dx0 = 0;

assignin('base', 'x0',  x0);
assignin('base', 'dx0', dx0);

assignin('base', 'x0_lin',  x0);  assignin('base', 'dx0_lin',  dx0);
assignin('base', 'x0_nl',   x0);  assignin('base', 'dx0_nl',   dx0);
assignin('base', 'x0_damp', x0);  assignin('base', 'dx0_damp', dx0);

simOut = sim(model, 'StopTime', num2str(T));

t = simOut.tout;

% Линейная модель: x'' + 4*pi^2*x = 0
x_lin  = get_sig(simOut, 'x');
dx_lin = get_sig(simOut, 'dx');

% Нелинейная модель: x'' + 4*pi^2*sin(x) = 0
x_nl   = get_sig(simOut, 'x1');
dx_nl  = get_sig(simOut, 'dx1');

% Модель с сопротивлением воздуха:
% x'' + 0.03*dx*|dx| + 4*pi^2*sin(x) = 0
x_damp  = get_sig(simOut, 'x2');
dx_damp = get_sig(simOut, 'dx2');

%% 1. Временные характеристики: сравнение x(t)
figure('Color','w')
plot(t, x_lin,  'LineWidth', 2); hold on; grid on
plot(t, x_nl,   '--', 'LineWidth', 2)
plot(t, x_damp, '-.', 'LineWidth', 2)
xlabel('t, с'); ylabel('x, рад')
title('Сравнение отклонений x(t) для трёх моделей маятника')
legend('линейная', 'нелинейная', 'с сопротивлением воздуха', ...
       'Location', 'best')

%% 2. Временные характеристики: сравнение dx(t)
figure('Color','w')
plot(t, dx_lin,  'LineWidth', 2); hold on; grid on
plot(t, dx_nl,   '--', 'LineWidth', 2)
plot(t, dx_damp, '-.', 'LineWidth', 2)
xlabel('t, с'); ylabel('dx/dt, рад/с')
title('Сравнение скоростей dx/dt для трёх моделей маятника')
legend('линейная', 'нелинейная', 'с сопротивлением воздуха', ...
       'Location', 'best')

%% 3. Фазовые портреты: сравнение в одном окне
figure('Color','w')
plot(x_lin, dx_lin, 'LineWidth', 2); hold on; grid on
plot(x_nl, dx_nl, '--', 'LineWidth', 2)
plot(x_damp, dx_damp, '-.', 'LineWidth', 2)
xlabel('x, рад'); ylabel('dx/dt, рад/с')
title('Фазовые портреты: сравнение трёх моделей')
legend('линейная', 'нелинейная', 'с сопротивлением воздуха', ...
       'Location', 'best')

%% 4. Фазовые портреты отдельно по подграфикам
figure('Color','w')

subplot(1,3,1)
plot(x_lin, dx_lin, 'LineWidth', 1.5); grid on
xlabel('x'); ylabel('dx/dt')
title('Линейная')

subplot(1,3,2)
plot(x_nl, dx_nl, 'LineWidth', 1.5); grid on
xlabel('x'); ylabel('dx/dt')
title('Нелинейная')

subplot(1,3,3)
plot(x_damp, dx_damp, 'LineWidth', 1.5); grid on
xlabel('x'); ylabel('dx/dt')
title('С сопротивлением')

%% Вспомогательная функция получения сигнала
function v = get_sig(simOut, name)
    if any(strcmp(simOut.who, name))
        v = simOut.get(name);
    else
        v = evalin('base', name);
    end

    if isa(v, 'timeseries')
        v = v.Data;
    end

    v = v(:);
end