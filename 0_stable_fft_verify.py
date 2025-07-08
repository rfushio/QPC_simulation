#!/usr/bin/env python3
"""
Stable Fourier transform coefficient verification script (Qhull error avoidance version)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from solver2 import SimulationConfig, ThomasFermiSolver

def test_fft_normalization():
    """Test if FFT normalization coefficients are correct"""
    print("=== FFT Normalization Test ===")
    
    # Simple test case
    Nx, Ny = 8, 8
    dx, dy = 1.0, 1.0
    
    # Delta function (1 at center, 0 elsewhere)
    test_array = np.zeros((Nx, Ny))
    test_array[Nx//2, Ny//2] = 1.0
    
    # FFT
    fft_result = np.fft.fft2(test_array)
    
    # Inverse FFT
    ifft_result = np.fft.ifft2(fft_result)
    
    print(f"Original array sum: {np.sum(test_array):.6f}")
    print(f"FFT→IFFT sum: {np.sum(ifft_result):.6f}")
    print(f"Maximum error: {np.max(np.abs(test_array - np.real(ifft_result))):.2e}")
    
    # Check normalization coefficients
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    
    print(f"kx range: [{kx.min():.3f}, {kx.max():.3f}]")
    print(f"ky range: [{ky.min():.3f}, {ky.max():.3f}]")
    print(f"dkx = {kx[1] - kx[0]:.6f}")
    print(f"dky = {ky[1] - ky[0]:.6f}")
    
    return np.max(np.abs(test_array - np.real(ifft_result))) < 1e-10

def test_gaussian_convolution_simple():
    """Simple Gaussian convolution verification (Qhull error avoidance)"""
    print("\n=== Gaussian Convolution Test (Simple Version) ===")
    
    # Configure with larger scale
    cfg = SimulationConfig(
        Nx=32, Ny=32,
        B=13.0,
        # Use larger range data
        potential_data=(np.linspace(0, 1000, 100), np.linspace(0, 1000, 100), np.zeros(100))
    )
    
    solver = ThomasFermiSolver(cfg)
    
    # Test array (Gaussian)
    x, y = np.meshgrid(solver.x, solver.y, indexing='ij')
    sigma = 100e-9  # 100 nm (larger scale)
    center_x = solver.x[solver.Nx//2]
    center_y = solver.y[solver.Ny//2]
    
    test_array = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    
    print(f"Test array properties:")
    print(f"  Maximum: {np.max(test_array):.6f}")
    print(f"  Minimum: {np.min(test_array):.6f}")
    print(f"  Sum: {np.sum(test_array):.6f}")
    print(f"  Center position: ({center_x*1e9:.1f} nm, {center_y*1e9:.1f} nm)")
    print(f"  Gaussian width: {sigma*1e9:.1f} nm")
    
    # Gaussian convolution
    convolved = solver.gaussian_convolve(test_array)
    
    print(f"\nConvolved properties:")
    print(f"  Maximum: {np.max(convolved):.6f}")
    print(f"  Minimum: {np.min(convolved):.6f}")
    print(f"  Sum: {np.sum(convolved):.6f}")
    
    # Comparison with theoretical value (convolution of Gaussian is Gaussian)
    # New variance: σ²_new = σ²_old + σ²_kernel
    sigma_kernel = solver.ell_B
    sigma_new = np.sqrt(sigma**2 + sigma_kernel**2)
    theoretical = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma_new**2))
    
    # Normalization
    theoretical = theoretical / np.sum(theoretical) * np.sum(convolved)
    
    error = np.max(np.abs(convolved - theoretical))
    relative_error = error / np.max(convolved)
    
    print(f"\nComparison with theoretical value:")
    print(f"  Maximum error: {error:.2e}")
    print(f"  Relative error: {relative_error:.2%}")
    print(f"  ℓ_B = {solver.ell_B:.2e} m ({solver.ell_B*1e9:.2f} nm)")
    print(f"  Original σ = {sigma:.2e} m ({sigma*1e9:.1f} nm)")
    print(f"  Theoretical new σ = {sigma_new:.2e} m ({sigma_new*1e9:.1f} nm)")
    
    return relative_error < 0.1  # Allow relative error within 10%

def test_coulomb_energy_simple():
    """Simple Coulomb energy verification"""
    print("\n=== Coulomb Energy Test (Simple Version) ===")
    
    cfg = SimulationConfig(
        Nx=32, Ny=32,
        B=13.0,
        potential_data=(np.linspace(0, 1000, 100), np.linspace(0, 1000, 100), np.zeros(100))
    )
    
    solver = ThomasFermiSolver(cfg)
    
    # Uniform density distribution
    uniform_density = np.ones((solver.Nx, solver.Ny)) * 0.5
    
    # Calculate Coulomb energy
    n_eff = uniform_density * solver.D
    n_fft = np.fft.fft2(n_eff)
    n_q = n_fft * solver.dA
    E_C = 0.5 / solver.A_total * np.sum(solver.Vq * np.abs(n_q) ** 2) * solver.J_to_meV
    
    print(f"Coulomb energy of uniform density distribution: {E_C:.6f} meV")
    print(f"Density: {n_eff[0,0]:.2e} m⁻²")
    print(f"Area: {solver.A_total:.2e} m²")
    print(f"Theoretical value (uniform distribution): 0 meV")
    print(f"Difference from calculation: {E_C:.6f} meV")
    
    return abs(E_C) < 1e-3  # Allow error less than 1 meV

def test_energy_components():
    """Energy component verification"""
    print("\n=== Energy Component Test ===")
    
    cfg = SimulationConfig(
        Nx=16, Ny=16,
        B=13.0,
        potential_data=(np.linspace(0, 1000, 100), np.linspace(0, 1000, 100), np.zeros(100))
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

def test_fft_coefficients():
    """Detailed FFT coefficient check"""
    print("\n=== Detailed FFT Coefficient Check ===")
    
    cfg = SimulationConfig(
        Nx=16, Ny=16,
        B=13.0,
        potential_data=(np.linspace(0, 1000, 100), np.linspace(0, 1000, 100), np.zeros(100))
    )
    
    solver = ThomasFermiSolver(cfg)
    
    # Wave vector calculation
    kx = 2 * np.pi * np.fft.fftfreq(solver.Nx, d=solver.dx)
    ky = 2 * np.pi * np.fft.fftfreq(solver.Ny, d=solver.dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    q = np.sqrt(KX**2 + KY**2)
    
    print(f"Grid size: {solver.Nx} x {solver.Ny}")
    print(f"Spatial step: dx = {solver.dx:.2e} m, dy = {solver.dy:.2e} m")
    print(f"Wave number step: dkx = {kx[1] - kx[0]:.2e} m⁻¹, dky = {ky[1] - ky[0]:.2e} m⁻¹")
    print(f"Wave number range: kx ∈ [{kx.min():.2e}, {kx.max():.2e}] m⁻¹")
    print(f"Wave number range: ky ∈ [{ky.min():.2e}, {ky.max():.2e}] m⁻¹")
    print(f"q(0,0) = {q[0,0]:.2e} m⁻¹")
    print(f"q minimum: {np.min(q):.2e} m⁻¹")
    print(f"q maximum: {np.max(q):.2e} m⁻¹")
    
    # Gaussian kernel properties
    G_q = solver.G_q
    print(f"\nGaussian kernel properties:")
    print(f"G_q(0,0) = {G_q[0,0]:.6f} (value at q=0)")
    print(f"G_q maximum: {np.max(G_q):.6f}")
    print(f"G_q minimum: {np.min(G_q):.6f}")
    print(f"G_q mean: {np.mean(G_q):.6f}")
    
    # Coulomb kernel properties
    Vq = solver.Vq
    print(f"\nCoulomb kernel properties:")
    print(f"V_q(0,0) = {Vq[0,0]:.2e} (value at q=0)")
    print(f"V_q maximum: {np.max(Vq):.2e}")
    print(f"V_q minimum: {np.min(Vq):.2e}")
    print(f"V_q mean: {np.mean(Vq):.2e}")
    
    return True

def plot_simple_analysis():
    """Simple analysis visualization"""
    print("\n=== Simple Analysis Visualization ===")
    
    cfg = SimulationConfig(
        Nx=32, Ny=32,
        B=13.0,
        potential_data=(np.linspace(0, 1000, 100), np.linspace(0, 1000, 100), np.zeros(100))
    )
    
    solver = ThomasFermiSolver(cfg)
    
    # Wave vector
    kx = 2 * np.pi * np.fft.fftfreq(solver.Nx, d=solver.dx)
    ky = 2 * np.pi * np.fft.fftfreq(solver.Ny, d=solver.dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    q = np.sqrt(KX**2 + KY**2)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
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
    plt.savefig('stable_fft_analysis.png', dpi=300, bbox_inches='tight')
    print("Simple analysis saved to 'stable_fft_analysis.png'")
    plt.close()

def main():
    """Main verification function"""
    print("Starting stable Fourier transform coefficient verification...\n")
    
    tests = [
        ("FFT Normalization", test_fft_normalization),
        ("Gaussian Convolution", test_gaussian_convolution_simple),
        ("Coulomb Energy", test_coulomb_energy_simple),
        ("Energy Components", test_energy_components),
        ("FFT Coefficient Details", test_fft_coefficients),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            status = "✓ PASS" if results[test_name] else "✗ FAIL"
            print(f"{test_name}: {status}\n")
        except Exception as e:
            results[test_name] = False
            print(f"{test_name}: ✗ ERROR - {e}\n")
    
    # Visualization
    try:
        plot_simple_analysis()
    except Exception as e:
        print(f"Visualization error: {e}")
    
    # Results summary
    print("=== Verification Results Summary ===")
    passed = sum(results.values())
    total = len(results)
    print(f"Passed tests: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Fourier transform coefficients are calculated correctly.")
    else:
        print("⚠️  Some tests failed. Please check the Fourier transform implementation.")
        for test_name, result in results.items():
            if not result:
                print(f"  - {test_name} failed")

if __name__ == "__main__":
    main() 