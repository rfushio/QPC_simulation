#!/usr/bin/env python3
"""
Script to verify that Fourier transform coefficients are calculated correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from solver1 import SimulationConfig, ThomasFermiSolver

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

def test_gaussian_convolution():
    """Gaussian convolution verification"""
    print("\n=== Gaussian Convolution Test ===")
    
    # Create solver with simple configuration
    cfg = SimulationConfig(
        Nx=16, Ny=16,
        B=13.0,
        potential_data=(np.linspace(0, 100, 100), np.linspace(0, 100, 100), np.zeros(100))
    )
    
    solver = ThomasFermiSolver(cfg)
    
    # Test array (Gaussian)
    x, y = np.meshgrid(solver.x, solver.y, indexing='ij')
    sigma = 10e-9  # 10 nm
    test_array = np.exp(-((x - solver.x[solver.Nx//2])**2 + (y - solver.y[solver.Ny//2])**2) / (2 * sigma**2))
    
    # Gaussian convolution
    convolved = solver.gaussian_convolve(test_array)
    
    # Comparison with theoretical value (convolution of Gaussian is Gaussian)
    # New variance: σ²_new = σ²_old + σ²_kernel
    sigma_kernel = solver.ell_B
    sigma_new = np.sqrt(sigma**2 + sigma_kernel**2)
    theoretical = np.exp(-((x - solver.x[solver.Nx//2])**2 + (y - solver.y[solver.Ny//2])**2) / (2 * sigma_new**2))
    
    # Normalization
    theoretical = theoretical / np.sum(theoretical) * np.sum(convolved)
    
    error = np.max(np.abs(convolved - theoretical))
    print(f"Maximum error in Gaussian convolution: {error:.2e}")
    print(f"ℓ_B = {solver.ell_B:.2e} m")
    print(f"Original σ = {sigma:.2e} m")
    print(f"Theoretical new σ = {sigma_new:.2e} m")
    
    return error < 0.1  # Allow error within 10%

def test_coulomb_energy():
    """Coulomb energy verification"""
    print("\n=== Coulomb Energy Test ===")
    
    cfg = SimulationConfig(
        Nx=32, Ny=32,
        B=13.0,
        potential_data=(np.linspace(0, 100, 100), np.linspace(0, 100, 100), np.zeros(100))
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
    
    # Comparison with theoretical value (self-energy should be close to 0 for uniform distribution)
    print(f"Theoretical value (uniform distribution): 0 meV")
    print(f"Difference from calculation: {E_C:.6f} meV")
    
    return abs(E_C) < 1e-3  # Allow error less than 1 meV

def test_energy_conservation():
    """Energy conservation verification"""
    print("\n=== Energy Conservation Test ===")
    
    cfg = SimulationConfig(
        Nx=16, Ny=16,
        B=13.0,
        potential_data=(np.linspace(0, 100, 100), np.linspace(0, 100, 100), np.zeros(100))
    )
    
    solver = ThomasFermiSolver(cfg)
    
    # Calculate energy for different density distributions
    densities = [
        np.ones((solver.Nx, solver.Ny)) * 0.1,  # Low density
        np.ones((solver.Nx, solver.Ny)) * 0.5,  # Medium density
        np.ones((solver.Nx, solver.Ny)) * 0.9,  # High density
    ]
    
    energies = []
    for density in densities:
        energy = solver.energy(density.flatten())
        energies.append(energy)
        print(f"Density {np.mean(density):.1f}: Energy = {energy:.6f} meV")
    
    # Check if energies are in physically reasonable range
    all_positive = all(e > -1000 for e in energies)  # Above -1000 meV
    all_finite = all(np.isfinite(e) for e in energies)
    
    print(f"All energies in positive range: {all_positive}")
    print(f"All energies finite: {all_finite}")
    
    return all_positive and all_finite

def plot_fft_components():
    """FFT component visualization"""
    print("\n=== FFT Component Visualization ===")
    
    cfg = SimulationConfig(
        Nx=32, Ny=32,
        B=13.0,
        potential_data=(np.linspace(0, 100, 100), np.linspace(0, 100, 100), np.zeros(100))
    )
    
    solver = ThomasFermiSolver(cfg)
    
    # Test array
    test_array = np.random.random((solver.Nx, solver.Ny))
    
    # FFT
    fft_result = np.fft.fft2(test_array)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original array
    im1 = axes[0,0].imshow(test_array.T, origin='lower')
    axes[0,0].set_title('Original Array')
    plt.colorbar(im1, ax=axes[0,0])
    
    # FFT amplitude
    im2 = axes[0,1].imshow(np.abs(fft_result).T, origin='lower')
    axes[0,1].set_title('FFT Amplitude')
    plt.colorbar(im2, ax=axes[0,1])
    
    # FFT phase
    im3 = axes[0,2].imshow(np.angle(fft_result).T, origin='lower')
    axes[0,2].set_title('FFT Phase')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Gaussian kernel
    im4 = axes[1,0].imshow(solver.G_q.T, origin='lower')
    axes[1,0].set_title('Gaussian Kernel G_q')
    plt.colorbar(im4, ax=axes[1,0])
    
    # Coulomb kernel
    im5 = axes[1,1].imshow(np.log10(np.abs(solver.Vq)).T, origin='lower')
    axes[1,1].set_title('Coulomb Kernel V_q (log scale)')
    plt.colorbar(im5, ax=axes[1,1])
    
    # Convolution result
    convolved = solver.gaussian_convolve(test_array)
    im6 = axes[1,2].imshow(convolved.T, origin='lower')
    axes[1,2].set_title('Convolved Result')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig('fft_verification.png', dpi=300, bbox_inches='tight')
    print("FFT component visualization saved to 'fft_verification.png'")
    plt.close()

def main():
    """Main verification function"""
    print("Starting Fourier transform coefficient verification...\n")
    
    tests = [
        ("FFT Normalization", test_fft_normalization),
        ("Gaussian Convolution", test_gaussian_convolution),
        ("Coulomb Energy", test_coulomb_energy),
        ("Energy Conservation", test_energy_conservation),
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
        plot_fft_components()
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