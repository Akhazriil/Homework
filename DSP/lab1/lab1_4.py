import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy


def R(q):
    """
    Матрица поворота из кватерниона.
    Нужна, чтобы перевести вектор ускорения из системы координат тела (где его мерит IMU)
    в глобальную систему координат. Берётся прямо из формулы для R(q) в задании 4.
    """
    q0 = q[0]
    q1 = q[1][0]
    q2 = q[1][1]
    q3 = q[1][2]
    return np.array([[2*q0*q0 - 1 + 2*q1*q1, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
                     [2*q1*q2 + 2*q0*q3, 2*q0*q0 - 1 + 2*q2*q2, 2*q2*q3 - 2*q0*q1],
                     [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 2*q0*q0 - 1 + 2*q3*q3]])

def q_from_theta(theta):
    """
    Строим кватернион малого поворота из вектора ошибки угла delta_theta.
    В ESKF ошибка ориентации всегда маленькая, поэтому sin(theta/2) ~ theta/2, cos(theta/2) ~ 1.
    Это соответствует аппроксимации q(delta_theta) ≈ [1, 0.5*delta_theta] из лекций.
    """
    norm = np.linalg.norm(theta)
    if norm == 0:
        return [1, np.array([0, 0, 0])]
    return [np.cos(norm/2), np.array(theta / norm * np.sin(norm/2))]

def mult(q1, q2):
    """
    Перемножение кватернионов (q1 ⊗ q2).
    Используется для обновления ориентации при шаге прогноза и при внесении ошибки в состояние.
    """
    return [q1[0]*q2[0] - np.dot(q1[1], q2[1]),
            q1[0]*q2[1] + q2[0]*q1[1] + np.cross(q1[1], q2[1])]

def cross_matrix(a):
    """
    Кососимметрическая матрица [a]x для вектора a.
    Нужна, чтобы записать векторное произведение в матричном виде.
    Появляется в матрице Якоби F при линеаризации модели вращения.
    """
    a = a.flatten()
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])

def n(var_acc, var_gyro, dt):
    """
    Ковариация шума процесса Q_n.
    Строится из дисперсий акселерометра и гироскопа, умножается на dt^2, как в задании.
    """
    return dt*dt * np.block([[var_acc * np.eye(3), np.zeros((3,3))],
                             [np.zeros((3,3)), var_gyro * np.eye(3)]])

def L():
    """
    Матрица влияния шума L.
    Показывает, какие компоненты вектора ошибки (скорость и угол) напрямую зависят 
    от шума акселерометра и гироскопа согласно формуле из lab1.
    """
    return np.block([[np.zeros((3,3)), np.zeros((3,3))],
                     [np.eye(3), np.zeros((3,3))],
                     [np.zeros((3,3)), np.eye(3)]])

def F(dt, q, f):
    """
    Линеаризованная матрица перехода F для вектора ошибки.
    Получаем, беря частные производные уравнений движения по ошибке состояния.
    Блок -dt * [R(q)f]x отвечает за то, как ошибка ориентации превращается в ошибку ускорения.
    """
    Rq = R(q)
    a_body = f.flatten()
    return np.block([[np.eye(3), np.eye(3)*dt, np.zeros((3,3))],
                     [np.zeros((3,3)), np.eye(3), -dt * cross_matrix(Rq @ a_body)],
                     [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)]])

def prediction(var_acc, var_gyro, P, dt, q, f):
    """
    Прогноз ковариации ошибки: P_k|k-1 = F * P_{k-1} * F^T + L * Q_n * L^T.
    Ошибка растёт на каждом шаге из-за неточности датчиков и неидеальной модели.
    """
    Fx = F(dt, q, f)
    return Fx @ P @ Fx.T + L() @ n(var_acc, var_gyro, dt) @ L().T

def kalman_update(md, P, dt, q, f, p, var_acc, var_gyro, measurements, variances):
    """
    Шаг коррекции ESKF. Фильтр работает не с полным состоянием, а только с вектором ошибки md.
    """
    # Сначала прогнозируем ковариацию ошибки на текущий шаг
    P = prediction(var_acc, var_gyro, P, dt, q, f)
    
    # Если в этот момент нет внешних измерений (GNSS/лидар), просто возвращаем P
    if not measurements:
        return md, P

    n = len(measurements)
    
    # Склеиваем все доступные измерения в один большой вектор
    y_all = np.concatenate([m.flatten() for m in measurements])
    
    # Матрица наблюдения H. Поскольку GNSS и лидар меряют только положение,
    # H просто выбирает первые 3 компоненты вектора ошибки (ошибка позиции)
    H_all = np.zeros((3 * n, 9))
    for i in range(n):
        H_all[3*i : 3*(i+1), :3] = np.eye(3)
        
    # Собираем общую ковариацию шума измерений R (в коде обозначена Q_all)
    Q_all = np.zeros((3 * n, 3 * n))
    for i, var in enumerate(variances):
        Q_all[3*i : 3*(i+1), 3*i : 3*(i+1)] = var * np.eye(3)
    
    # Предсказанные измерения: просто текущее спрогнозированное положение p
    y_pred = np.tile(p.flatten(), n)
    
    # Ковариация инноваций S = H*P*H^T + R
    S = H_all @ P @ H_all.T + Q_all
    # Численная стабилизация, чтобы обратная матрица не ломалась при малых ошибках
    S += np.eye(3 * n) * 1e-8
    # Коэффициент усиления Калмана K = P * H^T * S^{-1}
    K = P @ H_all.T @ np.linalg.inv(S)
    
    # Инновация: разница между реальным измерением и предсказанием
    innovation = y_all - y_pred
    # Обновляем оценку ошибки: md_k = md_{k-1} + K * innovation (md_{k-1} тут всегда 0 после сброса)
    md = (K @ innovation).flatten()
    
    # Обновляем ковариацию ошибки по формуле P = (I - K*H)*P (или P - K*S*K^T)
    P = P - K @ S @ K.T
    
    return md, P

def plus(mx, md):
    """
    Внесение найденной ошибки обратно в номинальное состояние и сброс ошибки.
    Это ключевой шаг ESKF: ошибка оценивается линейно, а потом применяется к нелинейной модели.
    Позицию и скорость просто прибавляем. Ориентацию корректируем левым умножением
    на кватернион малого поворота q(delta_theta), как указано в формулах задания.
    После внесения ошибку обнуляем, так как она уже учтена в mx.
    """
    # Обновление позиции
    mx[0] = mx[0] + np.array([[md[0]], [md[1]], [md[2]]])
    # Обновление скорости
    mx[1] = mx[1] + np.array([[md[3]], [md[4]], [md[5]]])
    # Обновление ориентации
    mx[2] = mult(q_from_theta(np.array([md[6], md[7], md[8]])), mx[2])
    return mx

def upd(state, dt, f, w):
    """
    Прогноз номинального состояния (без учёта ошибки).
    Интегрируем уравнения движения из задания: позиция, скорость, ориентация.
    Ускорение из тела поворачиваем в мир через R(q), добавляем гравитацию g.
    Угловую скорость интегрируем в кватернион поворота q(w*dt).
    """
    g = np.array([0, 0, -9.81])
    a_body = f.flatten()
    Rq = R(state[2])
    a_world = Rq @ a_body + g
    state[0] = state[0] + dt * state[1] + 0.5 * dt*dt * a_world.reshape(-1, 1)
    state[1] = state[1] + dt * a_world.reshape(-1, 1)
    state[2] = mult(state[2], q_from_theta(w * dt))
    return state


# Основная функция запуска фильтра


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
    
    # Основной цикл фильтрации
    for idx, (t, f, w) in enumerate(control):
        # Поиск синхронизированных измерений (допускаем небольшую рассинхронизацию eps)
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

        # Сохраняем кватернион до прогноза, так как матрица F линеаризуется вокруг предыдущего состояния
        q_before = copy.deepcopy(mx[2])
        
        # Прогноз номинального состояния (неоцениваемая часть ESKF)
        mx = upd(mx, t - lstx, f, w)
        
        # Положение после прогноза для матрицы наблюдения
        p_after = copy.deepcopy(mx[0])

        # Формируем список измерений, которые доступны в данный момент времени
        measurements = []
        variances = []
        
        if idx_gnss != -1:
            measurements.append(obs_gnss[idx_gnss][1])
            variances.append(var_gnss)
            
        if idx_lidar != -1:
            measurements.append(obs_lidar[idx_lidar][1])
            variances.append(var_lidar)
        
        # Шаг коррекции: считаем ошибку md и обновляем её ковариацию P
        md, P = kalman_update(md, P, t - lstx, q_before, f, p_after, 
                             var_acc, var_gyro, measurements, variances)
        
        # Внесение ошибки в номинальное состояние и сброс md в ноль
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

    # Визуализация
    if plot:
        fig = plt.figure(figsize=(16, 10))
        
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
        
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(pos_error, 'b-', linewidth=1.5, label='Ошибка ESKF', alpha=0.7)
        ax2.fill_between(range(len(pos_error)), 0, pos_error, alpha=0.3, color='blue')
        ax2.axhline(y=rmse, color='r', linestyle='--', linewidth=2, label=f'RMSE = {rmse:.3f} м')
        ax2.set_xlabel('Номер шага', fontsize=10)
        ax2.set_ylabel('Ошибка, м', fontsize=10)
        ax2.set_title('Ошибка положения', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle=':')
        
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
        var_lidar=3.0,
        P0_scale=1.0,
        eps=0.06,
        plot=True
    )
    print(f'\nСреднеквадратическая ошибка (RMSE) положения: {rmse:.4f} м')