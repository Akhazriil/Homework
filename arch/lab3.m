%% Задание 3: распространение инфекционного заболевания
clear; clc; close all;
cd('C:\Users\reino\Desktop\Учеба\Homework\arch')

% Параметры и начальные условия (если они используются в модели)
p1 = 0.001;
p2 = 0.072;
p3 = 0.05;
x1_0 = 880;
x2_0 = 20;
x3_0 = 100;

assignin('base', 'p1', p1);
assignin('base', 'p2', p2);
assignin('base', 'p3', p3);
assignin('base', 'x1_0', x1_0);
assignin('base', 'x2_0', x2_0);
assignin('base', 'x3_0', x3_0);

simOut = sim('third_lab', 'StopTime', '100');

t  = simOut.tout;
x1 = get_sig(simOut, 'x1');
x2 = get_sig(simOut, 'x2');
x3 = get_sig(simOut, 'x3');

t1  = simOut.tout;
x4 = get_sig(simOut, 'x4');
x5 = get_sig(simOut, 'x5');
x6 = get_sig(simOut, 'x6');
x7 = get_sig(simOut, 'x7');

figure('Color','w')
plot(t, x1, '--', t, x2, ':', t, x3, '-', 'LineWidth', 2)
grid on
xlabel('t, с')
ylabel('Число особей')
title('Динамика распространения инфекции')
legend('x1 — не инфицированные', 'x2 — больные', 'x3 — вылечившиеся', ...
       'Location', 'best')

figure('Color','w')
plot(t, x4, '--', t, x5, ':', t, x6, '-', t, x7, '-.', 'LineWidth', 2)
grid on
xlabel('t, с')
ylabel('Число особей')
title('Динамика распространения инфекции(смерт)')
legend('x4 - не инфицированные', 'x5 - больные', 'x6 - вылечившиеся', ...
       "x7 - умершие", 'Location', 'best')

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