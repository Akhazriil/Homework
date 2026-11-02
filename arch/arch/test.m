%% Скрипт-диагностика для Задания 6 (Две емкости)
clear; clc; close all;

model = 'six_lab'; % Имя вашей Simulink-модели

% 1. ЗАДАЕМ ЭТАЛОННЫЕ ПАРАМЕТРЫ ИЗ МЕТОДИЧКИ
A1 = 10; A2 = 7; Q1 = 3; C1 = 0.64; C2 = 0.64; g = 9.81;
x1_0 = 2; x2_0 = 3;

fprintf('=== МАТЕМАТИЧЕСКАЯ ПРОВЕРКА ===\n');
% Считаем начальные расходы
Q_in2_0 = C1 * sqrt(g * x1_0); % Поток из 1-го во 2-й бак
Q_out2_0 = C2 * sqrt(g * x2_0); % Поток из 2-го бака наружу

% Считаем начальные производные (скорости изменения уровней)
dx1_dt_0 = (Q1 - Q_in2_0) / A1;
dx2_dt_0 = (Q_in2_0 - Q_out2_0) / A2;

fprintf('Начальный поток во 2-й бак: %.3f\n', Q_in2_0);
fprintf('Начальный поток из 2-го бака: %.3f\n', Q_out2_0);
fprintf('Ожидаемая скорость изменения x1 в t=0: %.4f (должна быть > 0, бак 1 наполняется)\n', dx1_dt_0);
fprintf('Ожидаемая скорость изменения x2 в t=0: %.4f (должна быть < 0, бак 2 ОПУСКАЕТСЯ!)\n\n', dx2_dt_0);

% Считаем установившийся режим (когда Q1 = Q_out1 = Q_out2)
x1_ss = (Q1 / C1)^2 / g;
x2_ss = (Q1 / C2)^2 / g;
fprintf('Ожидаемый установившийся уровень x1: %.4f\n', x1_ss);
fprintf('Ожидаемый установившийся уровень x2: %.4f\n\n', x2_ss);

% 2. ЗАПУСК ВАШЕЙ МОДЕЛИ
fprintf('=== ЗАПУСК SIMULINK-МОДЕЛИ ===\n');
try
    simOut = sim(model, 'StopTime', '10'); % 10 секунд достаточно для проверки
catch ME
    error('Не удалось запустить модель. Проверьте имя файла и ошибки в Simulink.\n%s', ME.message);
end

t = simOut.tout;
X1_sim = get_sig(simOut, 'y1');
X2_sim = get_sig(simOut, 'y2');

% 3. АНАЛИЗ РЕЗУЛЬТАТОВ
fprintf('=== ДИАГНОСТИКА ОШИБОК ===\n');

% Проверка начальных условий
if abs(X1_sim(1) - x1_0) > 0.1
    fprintf('❌ ОШИБКА: Начальное условие X1 не равно 2 (сейчас %g). Проверьте Integrator в Tank1.\n', X1_sim(1));
end
if abs(X2_sim(1) - x2_0) > 0.1
    fprintf('❌ ОШИБКА: Начальное условие X2 не равно 3 (сейчас %g). Проверьте Integrator в Tank2.\n', X2_sim(1));
end

% Проверка направления движения X2
% Вычисляем производную по первым двум точкам
dx2_sim = (X2_sim(2) - X2_sim(1)) / (t(2) - t(1));

if dx2_sim > 0
    fprintf('❌ ВНИМАНИЕ: График X2 идет ВВЕРХ (dx2/dt = %.4f > 0)!\n', dx2_sim);
    fprintf('   Причины:\n');
    fprintf('   1. В блоке Sum (Tank2) перепутаны знаки. Должно быть + для In3 и - для Product1.\n');
    fprintf('   2. Начальное условие Integrator1 (Tank2) стоит 2, а не 3.\n');
    fprintf('   3. В блоке Gain2 (Tank2) стоит 7, а не 1/7.\n');
    fprintf('   4. Связи: на вход In3 (Tank2) подается не Q2, а константа Q1.\n');
else
    fprintf('✅ X2 движется ВНИЗ, как и должно быть.\n');
end

% Проверка установившегося режима (на конце симуляции)
if length(t) > 50
    fprintf('Уровень X2 в конце симуляции (t=%.1f): %.4f (ожидается ~2.24)\n', t(end), X2_sim(end));
    if X2_sim(end) > 2.4
        fprintf('❌ ОШИБКА: Система не пришла к нужному уровню. Проверьте значения C1, C2, Q1 и коэффициенты Gain (площади баков).\n');
    end
end

% 4. ВИЗУАЛИЗАЦИЯ ДЛЯ СРАВНЕНИЯ
figure('Color','w', 'Name', 'Сравнение с эталоном');
plot(t, X1_sim, 'b-', 'LineWidth', 2); hold on; grid on;
plot(t, X2_sim, 'r-', 'LineWidth', 2);

% Рисуем ожидаемые касательные в t=0
t_tangent = linspace(0, t(end), 100);
x1_tangent = x1_0 + dx1_dt_0 * t_tangent;
x2_tangent = x2_0 + dx2_dt_0 * t_tangent;

plot(t_tangent, x1_tangent, 'b--', 'LineWidth', 1.5);
plot(t_tangent, x2_tangent, 'r--', 'LineWidth', 1.5);

% Рисуем линии установившегося режима
yline(x1_ss, 'b:', 'LineWidth', 1.5);
yline(x2_ss, 'r:', 'LineWidth', 1.5);

xlabel('Время, с'); ylabel('Уровень, м');
title('Проверка модели: сплошная - Simulink, пунктир - эталон');
legend('X1 (ваша)', 'X2 (ваша)', 'X1 касательная', 'X2 касательная', 'Уст. режим X1/X2', 'Location', 'best');

%% Вспомогательная функция
function v = get_sig(simOut, name)
    if any(strcmp(simOut.who, name))
        v = simOut.get(name);
    else
        try
            v = evalin('base', name);
        catch
            error(['Сигнал "', name, '" не найден! Проверьте имена блоков To Workspace.']);
        end
    end
    if isa(v, 'timeseries'), v = v.Data; end
    v = v(:);
end