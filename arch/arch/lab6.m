%% Задание 6: Две емкости (Tank System)
clear; clc; close all;

model = 'six_lab';        
T = 100;                    

% Параметры системы
A1 = 10;
A2 = 7;
Q1 = 3;
C1 = 0.64;
C2 = 0.64;
g = 9.81;
x1_0 = 2;
x2_0 = 3;

assignin('base', 'A1', A1); assignin('base', 'A2', A2);
assignin('base', 'Q1', Q1); assignin('base', 'C1', C1); assignin('base', 'C2', C2);
assignin('base', 'g', g);
assignin('base', 'x1_0', x1_0); assignin('base', 'x2_0', x2_0);

% Запуск симуляции
simOut = sim(model, 'StopTime', num2str(T));

t = simOut.tout;

% Извлечение сигналов из To Workspace
x1 = get_sig(simOut, 'x1');
x2 = get_sig(simOut, 'x2');

%% 1. График уровня в емкости 1
figure('Color','w')
plot(t, x1, 'k-', 'LineWidth', 2); grid on
xlabel('t, с'); ylabel('x_1, м')
title('Уровень воды в емкости 1')
legend('x1', 'Location', 'best')

%% 2. График уровня в емкости 2
figure('Color','w')
plot(t, x2, 'k--', 'LineWidth', 2); grid on
xlabel('t, с'); ylabel('x_2, м')
title('Уровень воды в емкости 2')
legend('x2', 'Location', 'best')

%% 3. Совмещенный график
figure('Color','w')
plot(t, x1, 'k-', 'LineWidth', 2); hold on; grid on
plot(t, x2, 'k--', 'LineWidth', 2)
xlabel('t, с'); ylabel('Уровень x_1, x_2, м')
title('Сравнение уровней в емкостях')
legend('x1', 'x2', 'Location', 'best')

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