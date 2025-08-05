import numpy as np
from scipy.fft import fft2, ifft2
from scipy.optimize import basinhopping
from scipy.interpolate import interp1d

# --- 1. MATLABから呼び出されるメイン関数 ---
#    引数に x_vec と y_vec を追加
def run_optimisation(config, Phi_grid, x_vec, y_vec):
    """
    MATLABから渡された設定、ポテンシャル、グリッドベクトルを使い、最適化を実行。
    """
    print("--- Python: Optimisation task started ---")
    
    # x_vec, y_vec を使ってソルバーを初期化
    solver = _ThomasFermiSolver(config, Phi_grid, x_vec, y_vec)
    
    solver.optimise()
    
    print("--- Python: Optimisation task finished ---")
    
    return solver.nu_smoothed


# --- 2. 内部ソルバークラス ---
class _ThomasFermiSolver:
    # コンストラクタの引数を変更
    def __init__(self, cfg_dict, Phi, x, y):
        print("Python Solver: Initializing...")
        self.cfg = cfg_dict
        self.Phi = np.array(Phi)
        
        # MATLABのmeshgridから渡される2D配列を処理
        x_grid = np.array(x)
        y_grid = np.array(y)

        # 1Dベクトルを抽出
        self.x = x_grid[0, :]
        self.y = y_grid[:, 0]

        # グリッドパラメータをベクトルから計算
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.Lx = (self.x[-1] - self.x[0])
        self.Ly = (self.y[-1] - self.y[0])
        self.dx = self.Lx / (self.Nx - 1) if self.Nx > 1 else self.Lx
        self.dy = self.Ly / (self.Ny - 1) if self.Ny > 1 else self.Ly
        self.dA = self.dx * self.dy

        print(f"Python Solver: Grid is {self.Nx}x{self.Ny}, Lx={self.Lx:.2e}, Ly={self.Ly:.2e}")
        
        self._prepare_kernels()
        self._prepare_exc_table()
        self._init_density()

    # _prepare_kernels, energy, optimise など、他のメソッドは変更不要です。
    # (内部のロジックは既に dx, dy など計算済みのプロパティを使っているため)
    
    def _prepare_kernels(self):
        # (変更なし)
        # 物理定数
        e = self.cfg['e']
        B = self.cfg['B']
        h = self.cfg['h']
        
        self.D = e * B / h
        self.ell_B = np.sqrt(h / (2 * np.pi * e * B))
        
        # フーリエ空間の波数ベクトル
        kx = 2 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        q = np.sqrt(KX**2 + KY**2)
        q[0, 0] = 1e-20

        self.G_q = np.exp(-0.5 * (self.ell_B * q)**2)

        # クーロンカーネル Vq
        epsilon_0 = self.cfg['epsilon_0']
        epsilon_perp = self.cfg['epsilon_perp']
        epsilon_parallel = self.cfg['epsilon_parallel']
        dt = self.cfg['dt']
        db = self.cfg['db']
        
        eps_hBN = np.sqrt(epsilon_perp * epsilon_parallel)
        beta = np.sqrt(epsilon_parallel / epsilon_perp)
        
        # With Screening Effect
        #numerator = 4 * np.pi * np.sinh(beta * dt * q) * np.sinh(beta * db * q)
        #denominator = np.sinh(beta * (dt + db) * q) * q
        #self.Vq = (e**2 / (4 * np.pi * epsilon_0 * eps_hBN)) * (numerator / denominator)

        # Without Gate Screening Effect
        self.Vq = (e**2) / (2 * eps_hBN * epsilon_0 * q)

    def _prepare_exc_table(self):
        # (変更なし)
        exc_data = np.loadtxt(self.cfg['excFile'], delimiter=",", skiprows=1)
        self.exc_interp = interp1d(exc_data[:, 0], exc_data[:, 1], kind="linear", bounds_error=False, fill_value="extrapolate")

    def _init_density(self):
        # (変更なし)
        nu0 = 0.5 - 1.0 * (self.Phi - np.median(self.Phi))
        self.nu0 = np.clip(nu0, 0.0, 1.0).flatten()
        
    def gaussian_convolve(self, arr):
        # (変更なし)
        return np.real(ifft2(fft2(arr) * self.G_q))

    def energy(self, nu_flat):
        # (変更なし)
        nu = nu_flat.reshape((self.Nx, self.Ny))
        nu_eff = self.gaussian_convolve(nu)
        n_eff = nu_eff * self.D

        n_q = fft2(n_eff) * self.dA
        E_C = 0.5 / (self.Lx * self.Ly) * np.sum(self.Vq * np.abs(n_q)**2)

        E_phi = np.sum(-self.cfg['e'] * self.Phi * n_eff) * self.dA
        
        E_xc = np.sum(self.exc_interp(nu_eff)) * self.dA * self.D

        total_J = E_phi + E_xc + E_C
        return total_J * (1.0 / 1.602e-22) # meVに変換

    def optimise(self):
        # (変更なし)
        bounds = [(0.0, 1.0)] * (self.Nx * self.Ny)
        
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "options": {
                "maxiter": self.cfg['lbfgs_maxiter'],
                "maxfun": self.cfg['lbfgs_maxfun'],
                "ftol": 1e-6,
                "eps": 1e-8,
            },
        }
        
        result = basinhopping(
            self.energy,
            self.nu0,
            niter=self.cfg['niter'],
            stepsize=self.cfg['step_size'],
            minimizer_kwargs=minimizer_kwargs,
            disp=True,
        )
        
        self.nu_opt = result.x.reshape((self.Nx, self.Ny))
        self.nu_smoothed = self.gaussian_convolve(self.nu_opt)
