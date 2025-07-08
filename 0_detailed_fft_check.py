#!/usr/bin/env python3
"""
Detailed Fourier transform coefficient verification script
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from solver1 import SimulationConfig, ThomasFermiSolver

def check_fft_coefficients():
    """Detailed FFT coefficient check"""
    print("=== Detailed FFT Coefficient Check ===")
    
    # Configuration
    Nx, Ny = 16, 16
    dx, dy = 1e-9, 1e-9  # 1 nm
    
    # Wave vector calculation
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    print(f"Grid size: {Nx} x {Ny}")
    print(f"Spatial step: dx = {dx:.2e} m, dy = {dy:.2e} m")
    print(f"Wave number step: dkx = {kx[1] - kx[0]:.2e} m⁻¹, dky = {ky[1] - ky[0]:.2e} m⁻¹")
    print(f"Wave number range: kx ∈ [{kx.min():.2e}, {kx.max():.2e}] m⁻¹")
    print(f"Wave number range: ky ∈ [{ky.min():.2e}, {ky.max():.2e}] m⁻¹")
    
    # Check normalization coefficients
    # FFT normalization: FFT→IFFT restores the original function
    test_func = np.random.random((Nx, Ny))
    
    # FFT
    fft_result = np.fft.fft2(test_func)
    
    # Inverse FFT (no normalization)
    ifft_result = np.fft.ifft2(fft_result)
    
    # Normalization factor
    norm_factor = Nx * Ny
    
    print(f"\nNormalization check:")
    print(f"Original function sum: {np.sum(test_func):.6f}")
    print(f"FFT→IFFT sum: {np.sum(ifft_result):.6f}")
    print(f"Normalization factor (Nx*Ny): {norm_factor}")
    print(f"Maximum error: {np.max(np.abs(test_func - ifft_result)):.2e}")
    
    return True

def check_gaussian_kernel():
    """Detailed Gaussian kernel check"""
    print("\n=== Detailed Gaussian Kernel Check ===")
    
    cfg = SimulationConfig(
        Nx=32, Ny=32,
        B=13.0,
        potential_data=(np.linspace(0, 100, 100), np.linspace(0, 100, 100), np.zeros(100))
    )
    
    solver = ThomasFermiSolver(cfg)
    
    # Magnetic length calculation
    ell_B = np.sqrt(solver.cfg.h / (2 * np.pi * solver.cfg.e * solver.cfg.B))
    
    print(f"Magnetic field: B = {solver.cfg.B} T")
    print(f"Magnetic length: ℓ_B = {ell_B:.2e} m")
    print(f"Magnetic length (nm): ℓ_B = {ell_B*1e9:.2f} nm")
    
    # Gaussian kernel properties
    G_q = solver.G_q
    
    print(f"\nGaussian kernel properties:")
    print(f"G_q(0,0) = {G_q[0,0]:.6f} (value at q=0)")
    print(f"G_q maximum: {np.max(G_q):.6f}")
    print(f"G_q minimum: {np.min(G_q):.6f}")
    print(f"G_q mean: {np.mean(G_q):.6f}")
    
    # Check full width at half maximum
    center_idx = (G_q.shape[0]//2, G_q.shape[1]//2)
    center_value = G_q[center_idx]
    half_max = center_value / 2
    
    # Find half maximum width
    half_max_indices = np.where(G_q >= half_max)
    if len(half_max_indices[0]) > 0:
        max_dist = np.max(np.sqrt((half_max_indices[0] - center_idx[0])**2 + 
                                 (half_max_indices[1] - center_idx[1])**2))
        print(f"Half width (grid units): {max_dist:.1f}")
        print(f"Half width (physical units): {max_dist * solver.dx * 1e9:.2f} nm")
    
    return True

def check_coulomb_kernel():
    """Detailed Coulomb kernel check"""
    print("\n=== Detailed Coulomb Kernel Check ===")
    
    cfg = SimulationConfig(
        Nx=32, Ny=32,
        B=13.0,
        potential_data=(np.linspace(0, 100, 100), np.linspace(0, 100, 100), np.zeros(100))
    )
    
    solver = ThomasFermiSolver(cfg)
    
    Vq = solver.Vq
    
    print(f"Coulomb kernel properties:")
    print(f"V_q(0,0) = {Vq[0,0]:.2e} (value at q=0)")
    print(f"V_q maximum: {np.max(Vq):.2e}")
    print(f"V_q minimum: {np.min(Vq):.2e}")
    print(f"V_q mean: {np.mean(Vq):.2e}")
    
    # Check singularity at q=0
    kx = 2 * np.pi * np.fft.fftfreq(solver.Nx, d=solver.dx)
    ky = 2 * np.pi * np.fft.fftfreq(solver.Ny, d=solver.dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    q = np.sqrt(KX**2 + KY**2)
    
    print(f"\nWave vector properties:")
    print(f"q(0,0) = {q[0,0]:.2e} m⁻¹")
    print(f"q minimum: {np.min(q):.2e} m⁻¹")
    print(f"q maximum: {np.max(q):.2e} m⁻¹")
    
    # Check physical constants
    e = solver.cfg.e
    epsilon_0 = solver.cfg.epsilon_0
    epsilon_hBN = np.sqrt(solver.cfg.epsilon_perp * solver.cfg.epsilon_parallel)
    
    print(f"\nPhysical constants:")
    print(f"Elementary charge: e = {e:.2e} C")
    print(f"Vacuum permittivity: ε₀ = {epsilon_0:.2e} F/m")
    print(f"hBN permittivity: ε_hBN = {epsilon_hBN:.2f}")
    
    return True

def check_energy_components():
    """Detailed energy component check"""
    print("\n=== Detailed Energy Component Check ===")
    
    cfg = SimulationConfig(
        Nx=16, Ny=16,
        B=13.0,
        potential_data=(np.linspace(0, 100, 100), np.linspace(0, 100, 100), np.zeros(100))
    )
    
    solver = ThomasFermiSolver(cfg)
    
    # Test density distribution
    test_density = np.ones((solver.Nx, solver.Ny)) * 0.5
    
    # Calculate each energy component separately
    nu_eff = solver.gaussian_convolve(test_density)
    n_eff = nu_eff * solver.D
    
    # Coulomb energy
    n_fft = np.fft.fft2(n_eff)
    n_q = n_fft * solver.dA
    E_C = 0.5 / solver.A_total * np.sum(solver.Vq * np.abs(n_q) ** 2) * solver.J_to_meV
    
    # External potential energy
    E_phi = np.sum(-solver.cfg.e * solver.Phi * n_eff) * solver.dA * solver.J_to_meV
    
    # Exchange-correlation energy
    E_xc = np.sum(solver.exc_interp(nu_eff)) * solver.dA * solver.D
    
    total_energy = E_phi + E_xc + E_C
    
    print(f"Energy components:")
    print(f"  Coulomb energy: {E_C:.6f} meV")
    print(f"  External potential energy: {E_phi:.6f} meV")
    print(f"  Exchange-correlation energy: {E_xc:.6f} meV")
    print(f"  Total energy: {total_energy:.6f} meV")
    
    # Check coefficients
    print(f"\nCoefficient check:")
    print(f"  Density: n_eff = {n_eff[0,0]:.2e} m⁻²")
    print(f"  Area element: dA = {solver.dA:.2e} m²")
    print(f"  Total area: A_total = {solver.A_total:.2e} m²")
    print(f"  Energy conversion factor: J_to_meV = {solver.J_to_meV:.2e}")
    
    return True

def plot_detailed_analysis():
    """Detailed analysis visualization"""
    print("\n=== Detailed Analysis Visualization ===")
    
    cfg = SimulationConfig(
        Nx=64, Ny=64,
        B=13.0,
        potential_data=(np.linspace(0, 100, 100), np.linspace(0, 100, 100), np.zeros(100))
    )
    
    solver = ThomasFermiSolver(cfg)
    
    # Wave vector
    kx = 2 * np.pi * np.fft.fftfreq(solver.Nx, d=solver.dx)
    ky = 2 * np.pi * np.fft.fftfreq(solver.Ny, d=solver.dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    q = np.sqrt(KX**2 + KY**2)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Wave vector
    im1 = axes[0,0].imshow(q.T, origin='lower')
    axes[0,0].set_title('Wave Vector Magnitude |q|')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Gaussian kernel
    im2 = axes[0,1].imshow(solver.G_q.T, origin='lower')
    axes[0,1].set_title('Gaussian Kernel G_q')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Coulomb kernel
    im3 = axes[0,2].imshow(np.log10(np.abs(solver.Vq)).T, origin='lower')
    axes[0,2].set_title('Coulomb Kernel V_q (log scale)')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Test function
    test_func = np.random.random((solver.Nx, solver.Ny))
    im4 = axes[1,0].imshow(test_func.T, origin='lower')
    axes[1,0].set_title('Test Function')
    plt.colorbar(im4, ax=axes[1,0])
    
    # FFT result
    fft_result = np.fft.fft2(test_func)
    im5 = axes[1,1].imshow(np.abs(fft_result).T, origin='lower')
    axes[1,1].set_title('FFT Amplitude')
    plt.colorbar(im5, ax=axes[1,1])
    
    # Convolution result
    convolved = solver.gaussian_convolve(test_func)
    im6 = axes[1,2].imshow(convolved.T, origin='lower')
    axes[1,2].set_title('Convolved Result')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig('detailed_fft_analysis.png', dpi=300, bbox_inches='tight')
    print("Detailed analysis saved to 'detailed_fft_analysis.png'")
    plt.close()

def main():
    """Main function"""
    print("Starting detailed Fourier transform coefficient verification...\n")
    
    checks = [
        ("FFT Coefficients", check_fft_coefficients),
        ("Gaussian Kernel", check_gaussian_kernel),
        ("Coulomb Kernel", check_coulomb_kernel),
        ("Energy Components", check_energy_components),
    ]
    
    for check_name, check_func in checks:
        try:
            check_func()
            print(f"{check_name}: ✓ Complete\n")
        except Exception as e:
            print(f"{check_name}: ✗ Error - {e}\n")
    
    # Visualization
    try:
        plot_detailed_analysis()
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("Detailed verification completed.")

if __name__ == "__main__":
    main() 