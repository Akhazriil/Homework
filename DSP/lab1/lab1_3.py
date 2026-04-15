import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

class RBPF:
    """
    Исправленная версия RBPF с правильным масштабированием
    """
    def __init__(self, n_particles, sensors, mu0=4*np.pi*1e-7,
                 lambda_pos=1.0, delta_mom=0.25, snr_db=20):
        self.N = n_particles
        self.sensors = np.asarray(sensors, dtype=np.float64)
        self.L = len(sensors)
        self.mu0 = mu0
        
        # Параметры случайного блуждания из задания 3: x_{k+1} = x_k + w_{k+1}
        # Q_p и Q_q - матрицы ковариаций шума процесса для позиции и момента диполя
        self.Q_p = (lambda_pos**2) * np.eye(2, dtype=np.float64)
        self.Q_q = (delta_mom**2) * np.eye(2, dtype=np.float64)
        
        # SNR-based шум: задаём уровень доверия к измерениям заранее
        self.snr_db = snr_db
        self.R = None
        self.inv_R = None
        
        # Хранилище частиц: позиция берётся в фильтр частиц, а момент оценивается аналитически (теорема Рао-Блэкуэлла)
        self.particles_p = None
        self.q_mean = None
        self.q_cov = None
        self.weights = None

    def compute_measurements(self, p, q):
        """Вычисление магнитного поля (векторизовано)"""
        # Разности координат: вектор от диполя до каждого сенсора
        dx = self.sensors[np.newaxis, :, 0] - p[:, 0:1]
        dy = self.sensors[np.newaxis, :, 1] - p[:, 1:2]
        dz = self.sensors[np.newaxis, :, 2]
        
        # Расстояния до куба для знаменателя
        r2 = dx**2 + dy**2 + dz**2
        r3 = np.power(r2, 1.5) + 1e-12
        
        # Формула Био-Савара для перпендикулярной компоненты поля.
        # Модель линейна относительно q, что позволяет разделить оценку позиции и момента
        factor = self.mu0 / (4.0 * np.pi)
        B = factor * (q[:, 0:1] * dy - q[:, 1:2] * dx) / r3
        
        return B.squeeze()

    def initialize_with_snr(self, p_mean, p_cov, q_mean, q_cov, y_first):
        """
        Инициализация с автоматическим определением шума на основе SNR
        """
        # Генерируем начальный набор частиц позиции из априорного распределения
        self.particles_p = np.random.multivariate_normal(p_mean, p_cov, self.N)
        # Для каждой частицы заводим отдельные параметры фильтра Калмана для момента
        self.q_mean = np.tile(q_mean, (self.N, 1))
        self.q_cov = np.tile(q_cov[np.newaxis, :, :], (self.N, 1, 1))
        self.weights = np.ones(self.N, dtype=np.float64) / self.N
        
        # Оценка уровня сигнала для определения шума
        # Используем первую частицу для примерной оценки
        B_est = self.compute_measurements(self.particles_p[:1], self.q_mean[:1])
        signal_power = np.mean(B_est**2)
        
        # Устанавливаем ковариацию шума наблюдения R исходя из заданного SNR
        noise_power = signal_power / (10**(self.snr_db/10))
        noise_std = np.sqrt(noise_power)
        
        self.R = noise_std**2 * np.eye(self.L, dtype=np.float64)
        self.inv_R = 1.0 / noise_std**2
        
        return self.R

    def predict(self):
        """Шаг предсказания (разнесение частиц и прогноз Калмана)"""
        # Случайное блуждание позиции: добавляем шум процесса к каждой частице
        self.particles_p += np.random.multivariate_normal(
            np.zeros(2), self.Q_p, self.N
        )
        # Случайное блуждание момента: сдвигаем среднее значение Калмана
        self.q_mean += np.random.multivariate_normal(
            np.zeros(2), self.Q_q, self.N
        )
        # Прогноз ковариации момента по формуле Калмана: P_{k|k-1} = P_{k-1|k-1} + Q_q
        self.q_cov += self.Q_q[np.newaxis, :, :]

    def update(self, y):
        """Шаг коррекции с использованием информационного фильтра"""
        y = np.asarray(y, dtype=np.float64).flatten()
        p = self.particles_p
        
        # 1. Вычисление матрицы чувствительности G (lead field matrix)
        # Показывает, как изменение момента q влияет на показания сенсоров при фиксированной позиции
        dx = self.sensors[np.newaxis, :, 0] - p[:, 0:1]
        dy = self.sensors[np.newaxis, :, 1] - p[:, 1:2]
        dz = self.sensors[np.newaxis, :, 2]
        r2 = dx**2 + dy**2 + dz**2
        r3 = np.power(r2, 1.5) + 1e-12
        factor = self.mu0 / (4.0 * np.pi)
        
        G = np.empty((self.N, self.L, 2), dtype=np.float64)
        G[:, :, 0] = factor * dy / r3  # ∂B/∂q_x
        G[:, :, 1] = -factor * dx / r3  # ∂B/∂q_y
        
        # 2. Предсказанные измерения и вектор инноваций (разница между реальным и предсказанным)
        y_pred = np.einsum('nij,nj->ni', G, self.q_mean)
        innov = y[np.newaxis, :] - y_pred
        
        # 3. Информационная форма Калмана для q (обновляем обратную ковариацию, это численно стабильнее)
        GtG = np.einsum('nli,nlj->nij', G, G)
        
        # Апостериорная ковариация q: J_new = J_prior + G^T R^{-1} G
        inv_prior = np.linalg.inv(self.q_cov)
        A = inv_prior + self.inv_R * GtG
        P_new = np.linalg.inv(A)
        
        # 4. Вычисление правдоподобия для весов частиц
        # Аналитически исключаем q из апостериорного распределения (суть теоремы Рао-Блэкуэлла)
        # Считаем логарифм правдоподобия, чтобы избежать переполнения при перемножении вероятностей
        v = innov * self.inv_R
        vG = np.einsum('nl,nli->ni', v, G)
        
        # Квадратичная форма в экспоненте гауссовского распределения
        quad = np.einsum('ni,nij,nj->n', vG, P_new, vG)
        
        # Логарифмический определитель ковариации инноваций через матричное тождество
        det_term = np.eye(2)[np.newaxis, :, :] + self.inv_R * (self.q_cov @ GtG)
        sign, logdet_det_term = np.linalg.slogdet(det_term)
        logdet_S = self.L * np.log(1.0/self.inv_R) + logdet_det_term
        
        # Итоговый логарифм правдоподобия
        log_likelihood = -0.5 * (self.inv_R * np.sum(innov**2, axis=1) - quad + logdet_S)
        
        # Обновление весов по формуле Байеса и нормировка
        log_weights = log_likelihood + np.log(self.weights + 1e-300)
        log_weights -= np.max(log_weights)
        self.weights = np.exp(log_weights)
        self.weights /= np.sum(self.weights + 1e-300)
        
        # Обновление моментов q с учётом полученных измерений (шаг коррекции Калмана)
        self.q_mean += np.einsum('nij,nj->ni', P_new, vG)
        self.q_cov = P_new

    def resample(self, threshold_ratio=0.5):
        """Ресэмплинг при малом ESS (борьба с вырождением частиц)"""
        # Считаем эффективное количество частиц
        ess = 1.0 / np.sum(self.weights**2)
        if ess < threshold_ratio * self.N:
            # Если частиц осталось мало, делаем систематический ресэмплинг
            indices = self._systematic_resample()
            self.particles_p = self.particles_p[indices]
            self.q_mean = self.q_mean[indices]
            self.q_cov = self.q_cov[indices]
            self.weights.fill(1.0 / self.N)
        return ess

    def _systematic_resample(self):
        """Систематический ресэмплинг"""
        u = (np.arange(self.N) + np.random.random()) / self.N
        cs = np.cumsum(self.weights)
        indices = np.searchsorted(cs, u)
        return np.clip(indices, 0, self.N - 1)

    def get_estimate(self):
        """Получение взвешенной оценки состояния"""
        # Условное матожидание апостериорного распределения: сумма частиц, умноженных на их веса
        p_est = np.average(self.particles_p, weights=self.weights, axis=0)
        q_est = np.average(self.q_mean, weights=self.weights, axis=0)
        return p_est, q_est
    

def generate_realistic_data(T, sensors, true_trajectory, noise_std_scale=0.01):
    """
    Генерация синтетических данных для проверки работы фильтра
    """
    L = len(sensors)
    mu0 = 4 * np.pi * 1e-7
    factor = mu0 / (4 * np.pi)
    
    p_true = np.zeros((T, 2))
    q_true = np.zeros((T, 2))
    y_clean = np.zeros((T, L))
    
    # Истинная траектория (движение по кругу) и вращение момента
    for k in range(T):
        t = k / T * 2 * np.pi
        p_true[k] = true_trajectory['center'] + true_trajectory['radius'] * np.array([np.cos(t), np.sin(t)])
        
        # Дипольный момент (вращается)
        q_true[k] = true_trajectory['q_amplitude'] * np.array([np.cos(t), np.sin(t)])
        
        # Вычисление чистого магнитного поля по закону Био-Савара без шума
        for i, (px, py) in enumerate([p_true[k]]):
            dx = sensors[:, 0] - px
            dy = sensors[:, 1] - py
            dz = sensors[:, 2]
            r3 = (dx**2 + dy**2 + dz**2)**1.5 + 1e-12
            y_clean[k] = factor * (q_true[k, 0] * dy - q_true[k, 1] * dx) / r3
    
    # Оценка уровня шума на основе амплитуды сигнала
    signal_std = np.std(y_clean)
    noise_std = noise_std_scale * signal_std
    
    # Добавление аддитивного белого гауссовского шума
    y_noisy = y_clean + np.random.normal(0, noise_std, (T, L))
    
    return p_true, q_true, y_clean, y_noisy, noise_std


T = 100  # Временных шагов
L = 25   # Сенсоров (5x5 сетка)

# Создание сенсоров на высоте 3 единицы
x = np.linspace(-10, 10, 5)
y = np.linspace(-10, 10, 5)
xx, yy = np.meshgrid(x, y)
sensors = np.column_stack([xx.ravel(), yy.ravel(), np.ones(25) * 3])

# Истинная траектория
true_trajectory = {
    'center': np.array([0.0, 0.0]),
    'radius': 5.0,
    'q_amplitude': 1000.0  # Дипольный момент (в единицах A·м)
}

p_true, q_true, y_clean, y_noisy, noise_std = generate_realistic_data(
    T, sensors, true_trajectory, noise_std_scale=0.05
)

# Инициализация фильтра
rbpf = RBPF(
    n_particles=2000,
    sensors=sensors,
    lambda_pos=0.5,
    delta_mom=50.0,  # Увеличен для соответствия амплитуде
    snr_db=20
)

# Инициализация с первым измерением для оценки SNR и расстановки частиц
rbpf.initialize_with_snr(
    p_mean=np.array([0.0, 0.0]),
    p_cov=np.diag([100.0, 100.0]),
    q_mean=np.array([500.0, 0.0]),
    q_cov=np.diag([10000.0, 10000.0]),
    y_first=y_noisy[0]
)

# Фильтрация
p_est = []
q_est = []
ess_history = []

for k in range(T):
    if k > 0:
        rbpf.predict()
    rbpf.update(y_noisy[k])
    ess = rbpf.resample()
    p_k, q_k = rbpf.get_estimate()
    p_est.append(p_k)
    q_est.append(q_k)
    ess_history.append(ess)

p_est = np.array(p_est)
q_est = np.array(q_est)


# 1. График траектории
plt.figure(figsize=(10, 8))

# Основной график траектории
plt.subplot(2, 2, 1)
plt.plot(p_true[:, 0], p_true[:, 1], 'g-', linewidth=2, label='Истинная траектория')
plt.plot(p_est[:, 0], p_est[:, 1], 'r--', linewidth=2, label='Оценка RBPF')
plt.scatter(sensors[:, 0], sensors[:, 1], c='blue', marker='s', s=30, alpha=0.5, label='Сенсоры')
plt.scatter(p_true[0, 0], p_true[0, 1], c='green', marker='o', s=100, label='Старт')
plt.scatter(p_true[-1, 0], p_true[-1, 1], c='red', marker='*', s=150, label='Финиш')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.title('Траектория движения диполя')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 2. Ошибка позиции
plt.subplot(2, 2, 2)
position_error = np.sqrt(np.sum((p_true - p_est)**2, axis=1))
plt.plot(position_error, 'b-', linewidth=2)
plt.fill_between(range(len(position_error)), 0, position_error, alpha=0.3)
plt.xlabel('Временной шаг')
plt.ylabel('Ошибка позиции')
plt.title(f'Ошибка оценки позиции (ср.: {np.mean(position_error):.3f})')
plt.grid(True, alpha=0.3)

# 3. Дипольный момент
plt.subplot(2, 2, 3)
plt.plot(q_true[:, 0], 'g-', label='Истинный q_x', linewidth=2)
plt.plot(q_est[:, 0], 'r--', label='Оценка q_x', linewidth=2, alpha=0.7)
plt.plot(q_true[:, 1], 'b-', label='Истинный q_y', linewidth=2)
plt.plot(q_est[:, 1], 'orange', linestyle='--', label='Оценка q_y', linewidth=2, alpha=0.7)
plt.xlabel('Временной шаг')
plt.ylabel('Дипольный момент')
plt.title('Компоненты дипольного момента')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Ошибка дипольного момента
plt.subplot(2, 2, 4)
moment_error = np.sqrt(np.sum((q_true - q_est)**2, axis=1))
plt.plot(moment_error, 'r-', linewidth=2)
plt.fill_between(range(len(moment_error)), 0, moment_error, alpha=0.3)
plt.xlabel('Временной шаг')
plt.ylabel('Ошибка момента')
plt.title(f'Ошибка дипольного момента (ср.: {np.mean(moment_error):.2f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()