# lab1_4.py
# Error State Kalman Filter (ESKF) для навигации мобильного объекта
# Исправленная версия, аналогичная рабочему коду

import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

# Вспомогательные математические функции

def skew(v):
    """Кососимметрическая матрица для вектора v ∈ R^3."""
    v = v.flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

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

def quat_to_rot(q):
    """Альтернативное преобразование кватерниона в матрицу поворота."""
    return R(q)

def rotation_vector_to_quaternion(theta):
    """Кватернион поворота на угол |theta| вокруг оси theta/|theta|."""
    norm = np.linalg.norm(theta)
    if norm == 0:
        return [1, np.array([0, 0, 0])]
    return [np.cos(norm/2), np.array(theta / norm * np.sin(norm/2))]

def q_from_theta(theta):
    """Кватернион поворота на угол |theta| вокруг оси theta/|theta|."""
    return rotation_vector_to_quaternion(theta)

def quat_multiply(q1, q2):
    """Умножение кватернионов: q1 ⊗ q2."""
    return [q1[0]*q2[0] - np.dot(q1[1], q2[1]),
            q1[0]*q2[1] + q2[0]*q1[1] + np.cross(q1[1], q2[1])]

def mult(q1, q2):
    """Умножение кватернионов: q1 ⊗ q2."""
    return quat_multiply(q1, q2)

def normalize_quat(q):
    """Нормализация кватерниона."""
    norm = np.sqrt(q[0]**2 + np.dot(q[1], q[1]))
    if norm == 0:
        return [1, np.array([0, 0, 0])]
    return [q[0]/norm, q[1]/norm]

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

def compute_F_matrix(dt, q, acc):
    """Линеаризованная матрица перехода (альтернативная версия)."""
    return F(dt, q, acc)

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
    y_all = np.concatenate(measurements, axis=0)
    
    # матрица H (размерность 3*N x 9)
    # Для GNSS и лидара матрица наблюдения одинакова: H = [I_3 | 0_{3x6}]
    H_all = np.zeros((3 * n, 9))
    for i in range(n):
        H_all[3*i : 3*(i+1), :3] = np.eye(3)
        
    # матрица ковариации шума измерений Q (размерность 3*N x 3*N)
    Q_all = np.zeros((3 * n, 3 * n))
    for i, var in enumerate(variances):
        Q_all[3*i : 3*(i+1), 3*i : 3*(i+1)] = var * np.eye(3)
        
    S = H_all @ P @ H_all.T + Q_all
    # Численная стабилизация
    S += np.eye(3 * n) * 1e-8
    K = P @ H_all.T @ np.linalg.inv(S)
    
    y_pred = np.tile(p.flatten(), n)
    
    # Обновление вектора ошибки и ковариации
    md = (K @ (y_all - y_pred)).flatten()
    P = P - K @ S @ K.T
    
    return md, P

def plus(mx, md):
    """Внесение ошибки в номинальное состояние и сброс ошибки."""
    mx[0] = mx[0] + np.array([[md[0], md[1], md[2]]])
    mx[1] = mx[1] + np.array([[md[3], md[4], md[5]]])
    mx[2] = mult(q_from_theta(np.array([md[6], md[7], md[8]])), mx[2])
    mx[2] = normalize_quat(mx[2])
    return mx

def state_update(mx, md):
    """Внесение ошибки в номинальное состояние (альтернативная версия)."""
    return plus(mx, md)

def upd(state, dt, f, w):
    """Прогноз номинального состояния."""
    g = np.array([0, 0, -9.81])
    a_body = f.flatten()
    Rq = R(state[2])
    a_world = Rq @ a_body + g
    state[0] = state[0] + dt * state[1] + 0.5 * dt*dt * a_world.reshape(-1, 1)
    state[1] = state[1] + dt * a_world.reshape(-1, 1)
    state[2] = mult(state[2], q_from_theta(w * dt))
    state[2] = normalize_quat(state[2])
    return state

def predict_nominal(mx, dt, acc, gyro):
    """Прогноз номинального состояния (альтернативная версия)."""
    return upd(mx, dt, acc, gyro)

# ==============================================================================
# Основная функция запуска фильтра
# ==============================================================================

def run_eskf(data_path='data/data.pkl', var_acc=0.02, var_gyro=0.02, 
             var_gnss=1.0, var_lidar=1.0, P0_scale=1.0, eps=0.05, plot=True):
    """Запуск ESKF и визуализация результатов."""
    
    # 1. Загрузка данных
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    gt = data['gt']
    imu_f = data['imu_f']
    imu_w = data['imu_w']
    gnss = data.get('gnss', None)
    lidar = data.get('lidar', None)
    
    # Преобразование лидара (матрица C и вектор tt)
    C = np.array([[0.99376, -0.09722, 0.05466],
                  [0.09971, 0.99401, -0.04475],
                  [-0.04998, 0.04992, 0.9975]])
    tt = np.array([[0.5, 0.1, 0.5]])

    # Подготовка управляющих воздействий и наблюдений
    control = []  # [время, ускорение, угловая скорость]
    for i in range(1, len(imu_f.data)):
        control.append([imu_f.t[i], imu_f.data[i-1], imu_w.data[i-1]])

    # Преобразование лидара
    obs_lidar = []
    for i in range(len(lidar.t)):
        z = (C @ lidar.data[i]).T + tt
        obs_lidar.append([lidar.t[i], z.flatten()])

    obs_gnss = []
    for i in range(len(gnss.t)):
        obs_gnss.append([gnss.t[i], gnss.data[i].flatten()])

    # 2. Инициализация фильтра
    mx = [np.array(gt.p[0]).reshape(-1, 1),   # положение (3,1)
          np.array(gt.v[0]).reshape(-1, 1),   # скорость (3,1)
          [1, np.array([0, 0, 0])]]           # единичный кватернион
    md = np.zeros(9)
    P = np.eye(9) * P0_scale

    lstx = imu_f.t[0]
    est_pos = []   # список оценок положения
    est_vel = []   # список оценок скорости

    print(f"Всего шагов: {len(control)}")
    
    # 3. Цикл фильтрации
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

        q_before = copy.deepcopy(mx[2])    # кватернион до прогноза
        mx = upd(mx, t - lstx, f, w)       # прогноз номинального состояния
        p_after = copy.deepcopy(mx[0])     # положение после прогноза

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
        if np.any(md != 0):
            mx = plus(mx, md)
            md = np.zeros(9)  # Сброс ошибки

        # Сохраняем результаты
        est_pos.append(mx[0].flatten())
        est_vel.append(mx[1].flatten())
        lstx = t
        
        # Отображение прогресса
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

    # Визуализация
    if plot:
        fig = plt.figure(figsize=(18, 6))
        
        # Левый подграфик - 3D траектории
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot(est_positions[:min_len, 0], est_positions[:min_len, 1], est_positions[:min_len, 2],
                'b-', linewidth=2, label='ESKF')
        ax1.plot(gt.p[:min_len, 0], gt.p[:min_len, 1], gt.p[:min_len, 2],
                'k-', linewidth=2, label='Ground truth')
        ax1.set_xlabel('x [м]')
        ax1.set_ylabel('y [м]')
        ax1.set_zlabel('z [м]')
        ax1.set_title('Сравнение траекторий')
        ax1.set_zlim(-0.75, 1)
        ax1.legend()
        
        # Правый подграфик - ошибка положения во времени
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(pos_error, 'g-', linewidth=1.5)
        ax2.set_xlabel('Шаг')
        ax2.set_ylabel('Ошибка положения [м]')
        ax2.set_title('Ошибка положения во времени')
        ax2.grid(alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        # Добавляем RMSE на график
        ax2.axhline(y=rmse, color='r', linestyle='--', label=f'RMSE: {rmse:.4f} м')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('eskf_results.png', dpi=150)
        plt.show()

    return est_positions, rmse


if __name__ == "__main__":
    print("Запуск ESKF фильтра...")
    print("\n=== Запуск с параметрами по умолчанию ===")
    est_pos, rmse = run_eskf(
        data_path='data/data.pkl', 
        var_acc=0.01,        # Уменьшенный шум акселерометра
        var_gyro=0.01,       # Уменьшенный шум гироскопа
        var_gnss=1.0, 
        var_lidar=1.0, 
        P0_scale=1.0, 
        eps=0.06,            # Увеличенный допуск для синхронизации
        plot=True
    )
    print(f'\nСреднеквадратическая ошибка (RMSE) положения: {rmse:.4f} м')