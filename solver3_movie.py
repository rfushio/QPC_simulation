### Matryoshka solver that uses smaller grids for each stage as an initial guess based on solver2.py ###

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Tuple, List

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom

# Re-use the full solver2 implementation for all low-level routines
from solver2 import SimulationConfig as _BaseConfig, ThomasFermiSolver as _BaseSolver


# -----------------------------------------------------------------------------
# 1. Configuration class – single N specifying both Nx and Ny
# -----------------------------------------------------------------------------

@dataclass
class SimulationConfig(_BaseConfig):
    """Configuration for *solver3* – single square grid size N.

    All parameters from *solver2.SimulationConfig* are inherited.  Only the
    grid definition changes: the user specifies *N* instead of separate *Nx*
    and *Ny*.  Internally we still populate ``Nx`` and ``Ny`` so that the
    parent solver code works unchanged.
    """

    # Square grid size (replaces Nx & Ny)
    N: int = 64

    # ---------------- NEW PARAMETERS ----------------
    # Maximum number of accepted minima before early-exit on coarse stages (<N).
    # If <=0 the optimiser will run full ``cfg.niter`` iterations.
    coarse_accept_limit: int = 3

    # Frames-per-second for the generated ν-evolution movie
    movie_fps: int = 5

    # The parent dataclass already defines Nx, Ny.  We set default ``None`` so
    # that they do not appear twice in the generated ``__init__`` signature and
    # are filled in *__post_init__*.
    Nx: Optional[int] = None  # type: ignore[assignment]
    Ny: Optional[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:  # noqa: D401 – dataclass hook
        # Copy other field validation of parent (if any)
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type: ignore[misc]
        # Bind square size to parent attributes expected by BaseSolver
        self.Nx = self.Ny = int(self.N)


# -----------------------------------------------------------------------------
# 2. Thomas-Fermi solver with progressive-resolution optimisation
# -----------------------------------------------------------------------------

class ThomasFermiSolver(_BaseSolver):
    """Thomas–Fermi solver with progressive grid refinement.

    The algorithm starts on a coarse 16×16 grid, doubles the resolution until
    the requested *N* is reached (powers of two) and finally handles the last
    *non-power-of-two* step if necessary.  After each refinement the *ν*
    distribution from the previous level is interpolated to serve as the
    initial guess.  Only the final resolution results are kept on disk.
    """

    # ---------------------------- public API ---------------------------------

    def __init__(self, cfg: _BaseConfig):
        super().__init__(cfg)

    # ---------------------- multi-resolution optimise ------------------------

    def optimise(self):  # type: ignore[override]
        """Run optimisation using progressive grid refinement."""

        target_N: int = self.Nx  # equals cfg.N

        print(f"[solver3] Progressive optimisation up to N={target_N}")

        # Build resolution ladder; even for small grids, keep our custom loop
        if target_N <= 32:
            resolutions: List[int] = [target_N]
        else:
            resolutions = []
            n = 32
            while n < target_N:
                resolutions.append(n)
                if n * 2 > target_N:
                    resolutions.append(target_N)
                    break
                n *= 2
            else:
                if resolutions[-1] != target_N:
                    resolutions.append(target_N)

        # ------------------------------------------------------------------
        # Iterate over resolutions, carrying ν as initial guess (starting at 32)
        # ------------------------------------------------------------------
        start_time = time.time()

        # Aggregate energy history across all stages (0th entry will be first initial energy)
        self.energy_history: list[float] = []

        prev_nu: Optional[np.ndarray] = None  # type: ignore[assignment]
        prev_N: Optional[int] = None

        self.nu_frames: list[np.ndarray] = []  # Will hold frames for FINAL stage only

        for res in resolutions:
            stage_cfg = replace(self.cfg, N=res)  # type: ignore[arg-type]
            stage_solver = ThomasFermiSolver(stage_cfg)

            print(f"[solver3] >>> Starting stage with N={res}")

            # Initialise frame list for this stage (used only if final)
            stage_solver.nu_frames = []  # type: ignore[attr-defined]

            # --------------------------------------------------------------
            # Prepare initial density guess
            # --------------------------------------------------------------
            if prev_nu is not None and prev_N is not None:
                # Bilinear interpolation of previous ν to new grid
                stage_solver.nu0 = self._resample_nu(prev_nu, prev_N, res).flatten()
            # else: keep classical TF guess (already set in __init__)

            # Store initial frame (smoothed) *before* optimisation starts
            initial_frame = stage_solver.gaussian_convolve(stage_solver.nu0.reshape((res, res)))
            stage_solver.nu_frames.append(initial_frame)  # type: ignore[attr-defined]

            # Energy before optimisation (initial guess)
            E_initial = stage_solver.energy(stage_solver.nu0.copy())
            print(f"[solver3] N={res}: initial energy = {E_initial:.6e} meV")

            # --------------------------------------------------------------
            # Run optimisation – stop right after first accepted minimum
            # --------------------------------------------------------------
            _accepted_x: list[np.ndarray] = []
            accepted_count = 0  # NEW: track number of accepted minima in this stage
            iter_idx = 0        # iteration counter within this stage
            best_E = E_initial  # best energy so far in this stage

            class _EarlyStop(Exception):
                """Internal signal to stop basinhopping early for coarse grids."""

            def _bh_callback(x: np.ndarray, f: float, accept: bool):  # noqa: D401
                nonlocal accepted_count, iter_idx, best_E
                iter_idx += 1

                status = "accepted" if accept else "rejected"
                print(f"[solver3] N={res} iter={iter_idx}: {status} E={f:.6e} meV")
                if accept:
                    accepted_count += 1
                    _accepted_x.append(x.copy())
                    # Record energy history
                    stage_solver.energy_history.append(float(f))

                    # Record frame (smoothed ν)
                    nu_current = stage_solver.gaussian_convolve(x.reshape((res, res)))
                    stage_solver.nu_frames.append(nu_current)  # type: ignore[attr-defined]

                    # Print new global minimum info
                    if f < best_E - 1e-12:
                        print(
                            f"[solver3] N={res} iter={iter_idx}: NEW global minimum ΔE = {best_E - f:.6e}"
                        )
                        best_E = float(f)

                    # Early-stop logic (only for coarse stages)
                    if res < target_N and stage_solver.cfg.coarse_accept_limit > 0 and accepted_count >= stage_solver.cfg.coarse_accept_limit:  # type: ignore[attr-defined]
                        raise _EarlyStop

            # Basinhopping call -------------------------------------------------
            stage_solver.energy_history = [stage_solver.energy(stage_solver.nu0.copy())]
            bounds = [(0.0, 1.0)] * (stage_solver.Nx * stage_solver.Ny)

            from scipy.optimize import basinhopping  # local import to avoid circular

            try:
                result = basinhopping(
                    stage_solver.energy,
                    stage_solver.nu0.copy(),
                    minimizer_kwargs={
                        "method": "L-BFGS-B",
                        "bounds": bounds,
                        "options": {
                            "maxiter": stage_solver.cfg.lbfgs_maxiter,
                            "maxfun": stage_solver.cfg.lbfgs_maxfun,
                            "ftol": 1e-6,
                            "eps": 1e-8,
                        },
                    },
                    niter=stage_solver.cfg.niter,
                    stepsize=0.01,
                    disp=False,
                    callback=_bh_callback,
                )
            except _EarlyStop:
                # expected early termination after first accepted step (coarse grid)
                result = None

            # Determine best_x and energy -------------------------------------
            if result is not None:
                best_x = result.x
                E_new = float(result.fun)
            elif _accepted_x:
                best_x = _accepted_x[-1]
                E_new = float(stage_solver.energy(best_x))
            else:
                best_x = stage_solver.nu0.copy()
                E_new = E_initial

            if E_new != E_initial:
                print(
                    f"[solver3] N={res}: accepted new global minimum E = {E_new:.6e} meV "
                    f"(ΔE = {E_initial - E_new:.6e})"
                )
            else:
                print(f"[solver3] N={res}: no better minimum found; keeping initial energy")

            stage_solver.nu_opt = best_x.reshape((stage_solver.Nx, stage_solver.Ny))
            stage_solver.nu_smoothed = stage_solver.gaussian_convolve(stage_solver.nu_opt)

            # Store simple optimisation_result placeholder (final stage may overwrite)
            stage_solver.optimisation_result = {
                "N": res,
                "fun": E_new,
            }

            # Prepare for next resolution
            prev_nu = stage_solver.nu_opt
            prev_N = res

            # If this was the final resolution, copy attributes back to *self*
            if res == target_N:
                # Copy attributes, including frames, back to *self*
                self.__dict__.update(stage_solver.__dict__)
                # Store frames of final stage for video creation
                self.nu_frames = stage_solver.nu_frames  # type: ignore[attr-defined]
                break

            # Append energy history, avoid duplicating last value between stages
            if res == target_N:
                # final stage – take full history
                self.energy_history.extend(stage_solver.energy_history)
            else:
                # intermediate – skip last entry (will be start of next stage)
                self.energy_history.extend(stage_solver.energy_history[:-1])

        # Record total execution time (from first 16×16 start)
        self.execution_time = time.time() - start_time
        print(
            f"Progressive optimisation completed in {self.execution_time:.2f} s "
            f"({self.execution_time / 60:.2f} min)"
        )

        # Ensure optimisation_result exists (already set when final stage copied)
        if not hasattr(self, "optimisation_result"):
            self.optimisation_result = {
                "N": target_N,
                "fun": float(self.energy(self.nu_opt.flatten())),
            }

        return self.optimisation_result  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Override save_results to add video creation
    # ------------------------------------------------------------------
    def save_results(self, output_dir: Path | str = "results") -> None:  # type: ignore[override]
        # First run parent implementation to save arrays & metadata
        super().save_results(output_dir)

        # Create movie from stored frames (if any)
        if hasattr(self, "nu_frames") and self.nu_frames:
            out_path = Path(output_dir)
            video_path = out_path / "nu_evolution.mp4"

            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("inferno")

            # -------------------- MP4 via imageio-ffmpeg --------------------
            try:
                import imageio  # type: ignore
                import imageio_ffmpeg  # type: ignore  # triggers ffmpeg binary download if missing

                writer = imageio.get_writer(
                    video_path,
                    format="ffmpeg",
                    fps=self.cfg.movie_fps,  # type: ignore[attr-defined]
                    codec="libx264",
                )
                for frame in self.nu_frames:
                    norm = np.clip(frame, 0.0, 1.0)
                    rgba = cmap(norm)
                    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                    writer.append_data(rgb)
                writer.close()
                print(f"Movie saved to {video_path.resolve()}")
                return  # success
            except Exception as e:
                print("[solver3_movie] MP4 creation failed (", e, ") – falling back to GIF / PNG stack")

            # ------------------------ GIF fallback -------------------------
            try:
                import imageio  # type: ignore  # reuse import
                gif_path = video_path.with_suffix(".gif")
                images = []
                for frame in self.nu_frames:
                    norm = np.clip(frame, 0.0, 1.0)
                    rgba = cmap(norm)
                    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                    images.append(rgb)
                imageio.mimsave(gif_path, images, fps=self.cfg.movie_fps)  # type: ignore[attr-defined]
                print(f"GIF saved to {gif_path.resolve()}")
            except Exception as e2:
                # --------------------- PNG stack fallback -------------------
                print("[solver3_movie] GIF creation also failed:", e2)
                png_dir = out_path / "nu_frames"
                png_dir.mkdir(exist_ok=True)
                for i, frame in enumerate(self.nu_frames):
                    norm = np.clip(frame, 0.0, 1.0)
                    plt.imsave(png_dir / f"frame_{i:04d}.png", norm, cmap="inferno", vmin=0.0, vmax=1.0)
                print(f"Saved individual frames to {png_dir.resolve()}")

    # ------------------------------------------------------------------
    # Helper: resample ν field from *prev_N*×*prev_N* to *new_N*×*new_N*
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_nu(nu_prev: np.ndarray, prev_N: int, new_N: int) -> np.ndarray:
        """Bilinear resize of the ν field to a new square resolution."""
        if prev_N == new_N:
            return nu_prev.copy()

        # Create source & target coordinate grids in [0, 1]
        src = np.linspace(0.0, 1.0, prev_N)
        tgt = np.linspace(0.0, 1.0, new_N)
        rgi = RegularGridInterpolator((src, src), nu_prev, bounds_error=False, fill_value=None)  # type: ignore[arg-type]
        X_new, Y_new = np.meshgrid(tgt, tgt, indexing="ij")
        nu_new = rgi(np.stack([X_new, Y_new], axis=-1))
        return nu_new


# Convenience aliases ---------------------------------------------------------
SimulationConfig3 = SimulationConfig
ThomasFermiSolver3 = ThomasFermiSolver
