"""
YZR502u05a01 - ÖDEV0501
Mobil Robot için Sensör Füzyonu ile Lokalizasyon Simülasyonu
Genişletilmiş Kalman Filtresi (EKF) Tabanlı Sensör Füzyonu
Odometri + GPS Sensör Füzyonu
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.size'] = 11

# 1. SİMÜLASYON PARAMETRELERİ
dt = 0.1              # Zaman adımı (s)
T_total = 60.0        # Toplam simülasyon süresi (s)
N = int(T_total / dt) # Adım sayısı

# Gürültü parametreleri
sigma_v = 0.5         # Odometri lineer hız gürültüsü (m/s)
sigma_omega = 0.1     # Odometri açısal hız gürültüsü (rad/s)
sigma_gps_x = 2.0     # GPS x-pozisyon ölçüm gürültüsü (m)
sigma_gps_y = 2.0     # GPS y-pozisyon ölçüm gürültüsü (m)

# 2. MOBİL ROBOT MODELİ (Diferansiyel Tahrikli Robot)
# Durum vektörü: x = [x, y, theta]^T
# Kontrol girişi: u = [v, omega]^T (lineer hız, açısal hız)

def motion_model(state, u, dt):
    """
    Doğrusal olmayan hareket modeli (kinematik model).
    state: [x, y, theta]
    u: [v, omega]
    """
    x, y, theta = state
    v, omega = u

    if abs(omega) > 1e-6:
        x_new = x + (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        y_new = y + (v / omega) * (np.cos(theta) - np.cos(theta + omega * dt))
    else:
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt

    theta_new = theta + omega * dt
    # Açıyı [-pi, pi] aralığında tut
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

    return np.array([x_new, y_new, theta_new])


def jacobian_F(state, u, dt):
    """
    Hareket modelinin durum Jacobian'ı (F matrisi).
    """
    x, y, theta = state
    v, omega = u

    F = np.eye(3)
    if abs(omega) > 1e-6:
        F[0, 2] = (v / omega) * (np.cos(theta + omega * dt) - np.cos(theta))
        F[1, 2] = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
    else:
        F[0, 2] = -v * np.sin(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt

    return F


# 3. SENSÖR MODELLERİ

def odometry_model(state_true, u_true, dt):
    """
    Odometri sensörü: Gürültülü hız ölçümlerinden konum tahmini.
    Tekerlek enkoderlerini simüle eder.
    """
    v_noisy = u_true[0] + np.random.normal(0, sigma_v)
    omega_noisy = u_true[1] + np.random.normal(0, sigma_omega)
    return np.array([v_noisy, omega_noisy])


def gps_model(state_true):
    """
    GPS sensörü: Gürültülü konum ölçümü.
    Yalnızca x ve y konumunu ölçer (theta ölçmez).
    """
    z_x = state_true[0] + np.random.normal(0, sigma_gps_x)
    z_y = state_true[1] + np.random.normal(0, sigma_gps_y)
    return np.array([z_x, z_y])


# GPS ölçüm modeli: z = H * x + v
H_gps = np.array([
    [1, 0, 0],
    [0, 1, 0]
])

# 4. GENİŞLETİLMİŞ KALMAN FİLTRESİ (EKF)

class ExtendedKalmanFilter:
    """
    Genişletilmiş Kalman Filtresi (EKF) sınıfı.
    Odometri (tahmin) ve GPS (güncelleme) sensör füzyonu.
    """
    def __init__(self, x0, P0, Q, R):
        """
        x0: Başlangıç durum tahmini [x, y, theta]
        P0: Başlangıç kovaryans matrisi (3x3)
        Q:  Süreç gürültüsü kovaryans matrisi (3x3)
        R:  Ölçüm gürültüsü kovaryans matrisi (2x2)
        """
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R = R.copy()

    def predict(self, u, dt):
        """
        Tahmin (Prediction) adımı:
        1. Durumu hareket modeli ile ilerlet
        2. Kovaryansı Jacobian ile güncelle
        """
        # Durum tahmini: x_pred = f(x, u)
        self.x = motion_model(self.x, u, dt)

        # Jacobian hesapla
        F = jacobian_F(self.x, u, dt)

        # Kovaryans tahmini: P_pred = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

        return self.x.copy(), self.P.copy()

    def update(self, z, H):
        """
        Güncelleme (Update) adımı:
        1. İnovasyon (ölçüm artığı) hesapla
        2. Kalman kazancını hesapla
        3. Durum ve kovaryansı güncelle
        """
        # İnovasyon: y = z - H * x_pred
        y = z - H @ self.x

        # İnovasyon kovaryansı: S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R

        # Kalman kazancı: K = P * H^T * S^(-1)
        K = self.P @ H.T @ np.linalg.inv(S)

        # Durum güncelleme: x = x_pred + K * y
        self.x = self.x + K @ y

        # Açıyı normalize et
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi

        # Kovaryans güncelleme: P = (I - K * H) * P
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P

        return self.x.copy(), self.P.copy()


# 5. YÖRÜNGE OLUŞTURMA (Kare Yol)

def generate_square_trajectory(side_length=20.0, speed=2.0, dt=0.1):
    """
    Kare yörünge için kontrol girişleri oluşturur.
    Robot kare bir yolu takip eder.
    """
    # Her kenar için gereken süre
    t_side = side_length / speed
    # Dönüş için gereken süre (90 derece dönüş)
    omega_turn = np.pi / 4  # rad/s
    t_turn = (np.pi / 2) / omega_turn  # saniye

    controls = []
    times = []
    t = 0

    for i in range(8):  # 8 kenar (2 tur)
        # Düz git
        n_straight = int(t_side / dt)
        for _ in range(n_straight):
            controls.append([speed, 0.0])
            times.append(t)
            t += dt

        # 90 derece dön (sola)
        n_turn = int(t_turn / dt)
        for _ in range(n_turn):
            controls.append([speed * 0.3, omega_turn])
            times.append(t)
            t += dt

    return np.array(controls), np.array(times)


# 6. ANA SİMÜLASYON

def run_simulation():
    """Ana simülasyon fonksiyonu."""

    print("=" * 60)
    print("Mobil Robot Sensör Füzyonu Simülasyonu")
    print("Genişletilmiş Kalman Filtresi (EKF)")
    print("Odometri + GPS Füzyonu")
    print("=" * 60)

    # Kontrol girişlerini oluştur
    controls, times = generate_square_trajectory(side_length=20.0, speed=2.0, dt=dt)
    N_sim = len(controls)

    # Başlangıç durumu
    x0_true = np.array([0.0, 0.0, 0.0])  # [x, y, theta]

    # ---- Veri depolama dizileri ----
    true_states = np.zeros((N_sim + 1, 3))
    odom_states = np.zeros((N_sim + 1, 3))
    gps_measurements = np.zeros((N_sim, 2))
    ekf_states = np.zeros((N_sim + 1, 3))
    ekf_covariances = np.zeros((N_sim + 1, 3, 3))

    # Başlangıç değerleri
    true_states[0] = x0_true
    odom_states[0] = x0_true.copy()

    # ---- EKF Başlatma ----
    P0 = np.diag([0.1, 0.1, 0.01])  # Başlangıç kovaryans matrisi

    # Süreç gürültüsü kovaryans matrisi (Q)
    Q = np.diag([0.05, 0.05, 0.005])

    # Ölçüm gürültüsü kovaryans matrisi (R) - GPS için
    R = np.diag([sigma_gps_x**2, sigma_gps_y**2])

    ekf = ExtendedKalmanFilter(x0_true.copy(), P0, Q, R)
    ekf_states[0] = x0_true.copy()
    ekf_covariances[0] = P0.copy()

    # GPS ölçüm periyodu (her 3 adımda bir GPS güncellemesi)
    gps_period = 3

    # ---- Simülasyon Döngüsü ----
    for k in range(N_sim):
        u_true = controls[k]

        # 1) Gerçek durum güncelleme (gürültüsüz)
        true_states[k + 1] = motion_model(true_states[k], u_true, dt)

        # 2) Odometri tahmini (yalnızca odometri ile lokalizasyon)
        u_odom = odometry_model(true_states[k], u_true, dt)
        odom_states[k + 1] = motion_model(odom_states[k], u_odom, dt)

        # 3) EKF tahmin adımı (odometri verisi ile)
        ekf.predict(u_odom, dt)

        # 4) GPS ölçümü ve EKF güncelleme
        if k % gps_period == 0:
            z_gps = gps_model(true_states[k + 1])
            gps_measurements[k] = z_gps
            ekf.update(z_gps, H_gps)
        else:
            gps_measurements[k] = [np.nan, np.nan]

        ekf_states[k + 1] = ekf.x.copy()
        ekf_covariances[k + 1] = ekf.P.copy()

    # ---- Hata Hesaplama ----
    time_axis = np.arange(N_sim + 1) * dt

    # Öklid mesafesi ile lokalizasyon hatası
    error_odom = np.sqrt((true_states[:, 0] - odom_states[:, 0])**2 +
                         (true_states[:, 1] - odom_states[:, 1])**2)

    # GPS hatası (sadece ölçüm olan noktalarda)
    gps_valid = ~np.isnan(gps_measurements[:, 0])
    error_gps = np.full(N_sim + 1, np.nan)
    for k in range(N_sim):
        if gps_valid[k]:
            error_gps[k + 1] = np.sqrt(
                (true_states[k + 1, 0] - gps_measurements[k, 0])**2 +
                (true_states[k + 1, 1] - gps_measurements[k, 1])**2)

    error_ekf = np.sqrt((true_states[:, 0] - ekf_states[:, 0])**2 +
                        (true_states[:, 1] - ekf_states[:, 1])**2)

    # ---- İstatistiksel Sonuçlar ----
    rmse_odom = np.sqrt(np.mean(error_odom**2))
    rmse_gps = np.sqrt(np.nanmean(error_gps[~np.isnan(error_gps)]**2))
    rmse_ekf = np.sqrt(np.mean(error_ekf**2))

    print(f"\n{'Performans Metrikleri':^50}")
    print("-" * 50)
    print(f"{'Metrik':<30} {'Odometri':>8} {'GPS':>8} {'EKF':>8}")
    print("-" * 50)
    print(f"{'Ortalama Hata (m)':<30} {np.mean(error_odom):>8.3f} "
          f"{np.nanmean(error_gps):>8.3f} {np.mean(error_ekf):>8.3f}")
    print(f"{'Maks. Hata (m)':<30} {np.max(error_odom):>8.3f} "
          f"{np.nanmax(error_gps):>8.3f} {np.max(error_ekf):>8.3f}")
    print(f"{'Standart Sapma (m)':<30} {np.std(error_odom):>8.3f} "
          f"{np.nanstd(error_gps):>8.3f} {np.std(error_ekf):>8.3f}")
    print(f"{'RMSE (m)':<30} {rmse_odom:>8.3f} {rmse_gps:>8.3f} {rmse_ekf:>8.3f}")
    print("-" * 50)
    print(f"EKF iyileştirme (Odometriye gore): %{((rmse_odom - rmse_ekf) / rmse_odom * 100):.1f}")
    print(f"EKF iyileştirme (GPS'e gore):      %{((rmse_gps - rmse_ekf) / rmse_gps * 100):.1f}")

        # 7. GRAFİKLER
    
    # --- Şekil 1: Yörünge Karşılaştırma ---
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 7))
    ax1.plot(true_states[:, 0], true_states[:, 1], 'k-', linewidth=2.5,
             label='Gercek Yorunge', zorder=5)
    ax1.plot(odom_states[:, 0], odom_states[:, 1], 'b--', linewidth=1.2,
             alpha=0.7, label='Yalnizca Odometri')

    # GPS noktalarını çiz
    gps_x_vals = gps_measurements[gps_valid, 0]
    gps_y_vals = gps_measurements[gps_valid, 1]
    ax1.scatter(gps_x_vals, gps_y_vals, c='green', s=15, alpha=0.5,
                label='GPS Olcumleri', zorder=3)

    ax1.plot(ekf_states[:, 0], ekf_states[:, 1], 'r-', linewidth=1.5,
             alpha=0.9, label='EKF Sensor Fuzyonu', zorder=4)

    ax1.set_xlabel('X Konumu (m)', fontsize=12)
    ax1.set_ylabel('Y Konumu (m)', fontsize=12)
    ax1.set_title('Mobil Robot Yorunge Karsilastirmasi', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig('sekil1_yorunge_karsilastirma.png', dpi=200, bbox_inches='tight')
    print("\nSekil 1 kaydedildi: sekil1_yorunge_karsilastirma.png")

    # --- Şekil 2: Lokalizasyon Hatası (Zamana Bağlı) ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    ax2.plot(time_axis, error_odom, 'b-', linewidth=1.0, alpha=0.7,
             label='Yalnizca Odometri')
    ax2.plot(time_axis[~np.isnan(error_gps)], error_gps[~np.isnan(error_gps)],
             'g.', markersize=4, alpha=0.6, label='Yalnizca GPS')
    ax2.plot(time_axis, error_ekf, 'r-', linewidth=1.2, alpha=0.9,
             label='EKF Sensor Fuzyonu')

    ax2.set_xlabel('Zaman (s)', fontsize=12)
    ax2.set_ylabel('Lokalizasyon Hatasi (m)', fontsize=12)
    ax2.set_title('Zamana Bagli Lokalizasyon Hatasi Karsilastirmasi', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig('sekil2_hata_karsilastirma.png', dpi=200, bbox_inches='tight')
    print("Sekil 2 kaydedildi: sekil2_hata_karsilastirma.png")

    # --- Şekil 3: EKF Kovaryans (Belirsizlik) ---
    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ['X Belirsizligi (m)', 'Y Belirsizligi (m)', 'theta Belirsizligi (rad)']
    for i in range(3):
        sigma_vals = np.sqrt(ekf_covariances[:, i, i])
        axes3[i].plot(time_axis, sigma_vals, 'r-', linewidth=1.2)
        axes3[i].fill_between(time_axis, 0, sigma_vals, alpha=0.2, color='red')
        axes3[i].set_ylabel(labels[i], fontsize=10)
        axes3[i].grid(True, alpha=0.3)

    axes3[2].set_xlabel('Zaman (s)', fontsize=12)
    axes3[0].set_title('EKF Durum Tahmin Belirsizligi', fontsize=14, fontweight='bold')
    fig3.tight_layout()
    fig3.savefig('sekil3_ekf_belirsizlik.png', dpi=200, bbox_inches='tight')
    print("Sekil 3 kaydedildi: sekil3_ekf_belirsizlik.png")

    # --- Şekil 4: X ve Y Hata Bileşenleri ---
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax4a.plot(time_axis, true_states[:, 0] - odom_states[:, 0], 'b-',
              alpha=0.6, label='Odometri')
    ax4a.plot(time_axis, true_states[:, 0] - ekf_states[:, 0], 'r-',
              alpha=0.8, label='EKF')
    ax4a.set_ylabel('X Hatasi (m)', fontsize=11)
    ax4a.legend(fontsize=10)
    ax4a.grid(True, alpha=0.3)
    ax4a.set_title('X ve Y Konum Hatasi Bilesenleri', fontsize=14, fontweight='bold')

    ax4b.plot(time_axis, true_states[:, 1] - odom_states[:, 1], 'b-',
              alpha=0.6, label='Odometri')
    ax4b.plot(time_axis, true_states[:, 1] - ekf_states[:, 1], 'r-',
              alpha=0.8, label='EKF')
    ax4b.set_xlabel('Zaman (s)', fontsize=12)
    ax4b.set_ylabel('Y Hatasi (m)', fontsize=11)
    ax4b.legend(fontsize=10)
    ax4b.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4.savefig('sekil4_xy_hata_bilesenleri.png', dpi=200, bbox_inches='tight')
    print("Sekil 4 kaydedildi: sekil4_xy_hata_bilesenleri.png")

    # --- Şekil 5: RMSE Bar Grafiği ---
    fig5, ax5 = plt.subplots(1, 1, figsize=(7, 5))
    methods = ['Yalnizca\nOdometri', 'Yalnizca\nGPS', 'EKF\nSensor Fuzyonu']
    rmse_vals = [rmse_odom, rmse_gps, rmse_ekf]
    colors = ['#4472C4', '#70AD47', '#ED7D31']
    bars = ax5.bar(methods, rmse_vals, color=colors, width=0.5, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, rmse_vals):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{val:.3f} m', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax5.set_ylabel('RMSE (m)', fontsize=12)
    ax5.set_title('Lokalizasyon Yontemlerinin RMSE Karsilastirmasi', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    fig5.tight_layout()
    fig5.savefig('sekil5_rmse_karsilastirma.png', dpi=200, bbox_inches='tight')
    print("Sekil 5 kaydedildi: sekil5_rmse_karsilastirma.png")

    plt.close('all')

    return {
        'rmse_odom': rmse_odom,
        'rmse_gps': rmse_gps,
        'rmse_ekf': rmse_ekf,
        'mean_odom': np.mean(error_odom),
        'mean_gps': np.nanmean(error_gps),
        'mean_ekf': np.mean(error_ekf),
        'max_odom': np.max(error_odom),
        'max_gps': np.nanmax(error_gps),
        'max_ekf': np.max(error_ekf),
        'std_odom': np.std(error_odom),
        'std_gps': np.nanstd(error_gps),
        'std_ekf': np.std(error_ekf),
    }


# 8. ÇALIŞTIR
if __name__ == "__main__":
    np.random.seed(42)  # Tekrarlanabilirlik için
    results = run_simulation()
