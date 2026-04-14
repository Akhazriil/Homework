# lab1_4.py
# Error State Kalman Filter (ESKF) для навигации мобильного объекта

import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

# Вспомогательные математические функции

def R(q):
    """
    Матрица поворота из кватерниона.
    q = [q0, [q1, q2, q3]] (скалярная часть, векторная часть)
    """
    q0 = q[0]
    q1 = q[1][0]
    q2 = q[1][1]
    q3 = q[1][2]
    return np.array([[2*q0*q0 - 1 + 2*q1*q1, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
                     [2*q1*q2 + 2*q0*q3, 2*q0*q0 - 1 + 2*q2*q2, 2*q2*q3 - 2*q0*q1],
                     [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 2*q0*q0 - 1 + 2*q3*q3]])

def q_from_theta(theta):
    """Кватернион поворота на угол |theta| вокруг оси theta/|theta|."""
    norm = np.linalg.norm(theta)
    if norm == 0:
        return [1, np.array([0, 0, 0])]
    return [np.cos(norm/2), np.array(theta / norm * np.sin(norm/2))]

def mult(q1, q2):
    """Умножение кватернионов: q1 ⊗ q2."""
    return [q1[0]*q2[0] - np.dot(q1[1], q2[1]),
            q1[0]*q2[1] + q2[0]*q1[1] + np.cross(q1[1], q2[1])]

def cross_matrix(a):
    """Кососимметрическая матрица для вектора a (3x1)."""
    a = a.flatten()
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])

def n(var_acc, var_gyro, dt):
    """Ковариация шума процесса."""
    return dt*dt * np.block([[var_acc * np.eye(3), np.zeros((3,3))],
                             [np.zeros((3,3)), var_gyro * np.eye(3)]])

def L():
    """Матрица влияния шума."""
    return np.block([[np.zeros((3,3)), np.zeros((3,3))],
                     [np.eye(3), np.zeros((3,3))],
                     [np.zeros((3,3)), np.eye(3)]])

def F(dt, q, f):
    """Линеаризованная матрица перехода."""
    Rq = R(q)
    a_body = f.flatten()
    return np.block([[np.eye(3), np.eye(3)*dt, np.zeros((3,3))],
                     [np.zeros((3,3)), np.eye(3), -dt * cross_matrix(Rq @ a_body)],
                     [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)]])

def prediction(var_acc, var_gyro, P, dt, q, f):
    """Прогноз ковариации ошибки."""
    Fx = F(dt, q, f)
    return Fx @ P @ Fx.T + L() @ n(var_acc, var_gyro, dt) @ L().T

def kalman_update(md, P, dt, q, f, p, var_acc, var_gyro, measurements, variances):
    """Обновление ESKF для произвольного набора доступных измерений."""
    # Шаг прогноза ковариации ошибки
    P = prediction(var_acc, var_gyro, P, dt, q, f)
    
    # Если измерений нет, просто возвращаем спрогнозированную ковариацию
    if not measurements:
        return md, P

    n = len(measurements)
    
    # Единый вектор измерений (размерность 3*N)
    y_all = np.concatenate([m.flatten() for m in measurements])
    
    # матрица H (размерность 3*N x 9)
    H_all = np.zeros((3 * n, 9))
    for i in range(n):
        H_all[3*i : 3*(i+1), :3] = np.eye(3)
        
    # матрица ковариации шума измерений
    Q_all = np.zeros((3 * n, 3 * n))
    for i, var in enumerate(variances):
        Q_all[3*i : 3*(i+1), 3*i : 3*(i+1)] = var * np.eye(3)
    
    # Предсказанные измерения
    y_pred = np.tile(p.flatten(), n)
    
    S = H_all @ P @ H_all.T + Q_all
    # Численная стабилизация
    S += np.eye(3 * n) * 1e-8
    K = P @ H_all.T @ np.linalg.inv(S)
    
    # Обновление вектора ошибки
    innovation = y_all - y_pred
    md = (K @ innovation).flatten()
    
    # Обновление ковариации
    P = P - K @ S @ K.T
    
    return md, P

def plus(mx, md):
    """Внесение ошибки в номинальное состояние и сброс ошибки."""
    # Обновление позиции
    mx[0] = mx[0] + np.array([[md[0]], [md[1]], [md[2]]])
    # Обновление скорости
    mx[1] = mx[1] + np.array([[md[3]], [md[4]], [md[5]]])
    # Обновление ориентации
    mx[2] = mult(q_from_theta(np.array([md[6], md[7], md[8]])), mx[2])
    return mx

def upd(state, dt, f, w):
    """Прогноз номинального состояния."""
    g = np.array([0, 0, -9.81])
    a_body = f.flatten()
    Rq = R(state[2])
    a_world = Rq @ a_body + g
    state[0] = state[0] + dt * state[1] + 0.5 * dt*dt * a_world.reshape(-1, 1)
    state[1] = state[1] + dt * a_world.reshape(-1, 1)
    state[2] = mult(state[2], q_from_theta(w * dt))
    return state

# ==============================================================================
# Основная функция запуска фильтра
# ==============================================================================

def run_eskf(data_path='data/data.pkl', var_acc=0.01, var_gyro=0.01, 
             var_gnss=1.0, var_lidar=1.0, P0_scale=1.0, eps=0.06, plot=True):
    """Запуск ESKF и визуализация результатов."""
    
    # 1. Загрузка данных
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    gt = data['gt']
    imu_f = data['imu_f']
    imu_w = data['imu_w']
    gnss = data['gnss']
    lidar = data['lidar']
    
    # Матрица трансформации лидара
    C = np.array([[0.99376, -0.09722, 0.05466],
                  [0.09971, 0.99401, -0.04475],
                  [-0.04998, 0.04992, 0.9975]])
    tt = np.array([[0.5, 0.1, 0.5]])

    # Подготовка управляющих воздействий
    control = []  # [время, ускорение, угловая скорость]
    for i in range(1, len(imu_f.data)):
        control.append([imu_f.t[i], imu_f.data[i-1], imu_w.data[i-1]])

    # Преобразование измерений лидара
    obs_lidar = []
    for i in range(len(lidar.t)):
        z = (C @ lidar.data[i]).T + tt
        obs_lidar.append([lidar.t[i], z.flatten()])

    # Измерения GNSS
    obs_gnss = []
    for i in range(len(gnss.t)):
        obs_gnss.append([gnss.t[i], gnss.data[i].flatten()])

    # Инициализация фильтра
    mx = [np.array(gt.p[0]).reshape(-1, 1),   # положение (3,1)
          np.array(gt.v[0]).reshape(-1, 1),   # скорость (3,1)
          [1, np.array([0, 0, 0])]]           # единичный кватернион
    md = np.zeros(9)
    P = np.eye(9) * P0_scale

    lstx = imu_f.t[0]
    est_pos = []
    est_vel = []

    print(f"Всего шагов: {len(control)}")
    
    # Основной цикл фильтрации
    for idx, (t, f, w) in enumerate(control):
        # Поиск синхронизированных измерений
        idx_gnss = -1
        idx_lidar = -1
        
        for i, (tt_gnss, _) in enumerate(obs_gnss):
            if abs(t - tt_gnss) < eps:
                idx_gnss = i
                break
                
        for i, (tt_lidar, _) in enumerate(obs_lidar):
            if abs(t - tt_lidar) < eps:
                idx_lidar = i
                break

        # Сохраняем кватернион до прогноза
        q_before = copy.deepcopy(mx[2])
        
        # Прогноз номинального состояния
        mx = upd(mx, t - lstx, f, w)
        
        # Положение после прогноза для матрицы наблюдения
        p_after = copy.deepcopy(mx[0])

        # Формируем список измерений
        measurements = []
        variances = []
        
        if idx_gnss != -1:
            measurements.append(obs_gnss[idx_gnss][1])
            variances.append(var_gnss)
            
        if idx_lidar != -1:
            measurements.append(obs_lidar[idx_lidar][1])
            variances.append(var_lidar)
        
        # Шаг коррекции
        md, P = kalman_update(md, P, t - lstx, q_before, f, p_after, 
                             var_acc, var_gyro, measurements, variances)
        
        # Внесение ошибки в номинальное состояние
        if np.any(np.abs(md) > 1e-10):
            mx = plus(mx, md)
            md = np.zeros(9)

        # Сохраняем результаты
        est_pos.append(mx[0].flatten())
        est_vel.append(mx[1].flatten())
        lstx = t
        
        # Прогресс
        if (idx + 1) % 500 == 0:
            print(f"Обработано {idx + 1}/{len(control)} шагов")

    est_positions = np.array(est_pos)

    # Оценка точности
    min_len = min(len(est_positions), len(gt.p))
    pos_error = np.linalg.norm(est_positions[:min_len] - gt.p[:min_len], axis=1)
    rmse = np.sqrt(np.mean(pos_error**2))
    
    print(f"\n=== Результаты ===")
    print(f"RMSE положения: {rmse:.4f} м")
    print(f"Максимальная ошибка: {np.max(pos_error):.4f} м")
    print(f"Средняя ошибка: {np.mean(pos_error):.4f} м")
    print(f"Стандартное отклонение: {np.std(pos_error):.4f} м")

    # Визуализация (изменена только здесь)
    if plot:
        # Создаем фигуру с 3 подграфиками
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 3D траектория (изменены цвета и стили)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(gt.p[:min_len, 0], gt.p[:min_len, 1], gt.p[:min_len, 2],
                'green', linewidth=2, label='Истинная траектория', alpha=0.8)
        ax1.plot(est_positions[:min_len, 0], est_positions[:min_len, 1], est_positions[:min_len, 2],
                'red', linewidth=1.5, label='Оценка ESKF', alpha=0.9)
        ax1.scatter(est_positions[0, 0], est_positions[0, 1], est_positions[0, 2], 
                   color='blue', s=50, label='Старт')
        ax1.set_xlabel('X, м', fontsize=10)
        ax1.set_ylabel('Y, м', fontsize=10)
        ax1.set_zlabel('Z, м', fontsize=10)
        ax1.set_title('3D траектория движения', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Ошибка положения (изменен стиль графика)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(pos_error, 'b-', linewidth=1.5, label='Ошибка ESKF', alpha=0.7)
        ax2.fill_between(range(len(pos_error)), 0, pos_error, alpha=0.3, color='blue')
        ax2.axhline(y=rmse, color='r', linestyle='--', linewidth=2, label=f'RMSE = {rmse:.3f} м')
        ax2.set_xlabel('Номер шага', fontsize=10)
        ax2.set_ylabel('Ошибка, м', fontsize=10)
        ax2.set_title('Ошибка положения', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle=':')
        
        # 3. Ошибки по осям X, Y, Z (новый график)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(pos_error, 'k-', linewidth=1, label='Общая', alpha=0.5)
        ax3.plot(np.abs(est_positions[:min_len, 0] - gt.p[:min_len, 0]), 'r-', linewidth=1, label='Ошибка X', alpha=0.7)
        ax3.plot(np.abs(est_positions[:min_len, 1] - gt.p[:min_len, 1]), 'g-', linewidth=1, label='Ошибка Y', alpha=0.7)
        ax3.plot(np.abs(est_positions[:min_len, 2] - gt.p[:min_len, 2]), 'b-', linewidth=1, label='Ошибка Z', alpha=0.7)
        ax3.set_xlabel('Номер шага', fontsize=10)
        ax3.set_ylabel('Ошибка, м', fontsize=10)
        ax3.set_title('Ошибки по осям координат', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3, linestyle=':')
        
        # 4. Гистограмма ошибок (новый график)
        ax4 = fig.add_subplot(2, 2, 4)
        n_bins = 30
        ax4.hist(pos_error, bins=n_bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        ax4.axvline(x=rmse, color='r', linestyle='--', linewidth=2, label=f'RMSE = {rmse:.3f} м')
        ax4.axvline(x=np.mean(pos_error), color='orange', linestyle='--', linewidth=2, label=f'Среднее = {np.mean(pos_error):.3f} м')
        ax4.set_xlabel('Ошибка, м', fontsize=10)
        ax4.set_ylabel('Плотность', fontsize=10)
        ax4.set_title('Распределение ошибок', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Анализ работы ESKF фильтра', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

    return est_positions, rmse


if __name__ == "__main__":
    print("Запуск ESKF фильтра...")
    est_pos, rmse = run_eskf(
        data_path='data/data.pkl', 
        var_acc=0.01,
        var_gyro=0.01,
        var_gnss=1.0,
        var_lidar=1.0,
        P0_scale=1.0,
        eps=0.06,
        plot=True
    )
    print(f'\nСреднеквадратическая ошибка (RMSE) положения: {rmse:.4f} м')