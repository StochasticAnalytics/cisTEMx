#!/usr/bin/env python3
"""
Inner parameter sweep driver for the make_template_result peak upsampling
experiment. Loops over a list of MIP filenames, derives the sibling
psi/theta/phi/defocus/pixel_size/histogram files, then for each combination of
sweep parameters invokes make_template_result via stdin (heredoc-style).

The make_template_result binary must be built with cisTEM_test_peak_upsampling_sweep
defined (the source-level #define is enabled by default).

Outer sweeps over image-level conditions (defocus, pixel size buckets, etc.)
are intentionally out of scope - this script is the inner engine.

Usage:
    peak_upsampling_sweep.py \
        --reconstruction /path/to/reconstruction.mrc \
        --mip-list /path/to/mip_filenames.txt \
        --output-csv results.csv \
        --n-procs 1                  # crank up after dry run

Parallelism: process-level via ProcessPoolExecutor. Each iteration runs inside
its own short-lived TemporaryDirectory; the .mrc / coordinate outputs are
junk and get torn down with the tempdir. The CSV is written serially by the
main process as results stream back in via as_completed() - no locking needed.

Each MIP filename in --mip-list must contain the substring "mip" so we can
derive sibling files by substring replacement (mip -> psi, mip -> theta, etc).
"""

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import mrcfile

# Fixed inputs that should not vary across the inner sweep.
MIN_PEAK_RADIUS = 10.0
RESULT_NUMBER = 1
BINNING_FACTOR = 4.0
SLAB_THICKNESS = 2000.0

# Sibling file types corresponding to the make_template_result prompts.
SIBLING_TYPES = ("psi", "theta", "phi", "defocus", "pixel_size")

# Padding modes (kept in sync with Image::FindPeakWithIntegerCoordinatesForManyPeaksSweep).
PADDING_MODE_MIRROR = 0
PADDING_MODE_ZERO = 1
PADDING_MODE_HANN = 2

# Peak width is measured at WIDTH_FRACTION of the peak's excursion above baseline.
# 0.5 = FWHM-above-baseline; 0.67 ~= 2/3-max width (narrower, more robust to long
# tails). The C++ side defaults to 0.5 if not passed.
WIDTH_FRACTION = 0.5

# Per-peak annulus geometry: radii are inner/outer multiples of the measured FWHM
# (in MIP px, computed per peak). When FWHM is unavailable for a peak, fall back to
# fixed pixel radii. All radii in the output diagnostic file are reported in Angstroms.
ANNULUS_INNER_FWHM_MULT = 1.5
ANNULUS_OUTER_FWHM_MULT = 4.0
ANNULUS_FALLBACK_INNER_PX = 12.0
ANNULUS_FALLBACK_OUTER_PX = 24.0


def is_smooth(n: int, largest_factor: int = 5) -> bool:
    """Return True if n's prime factorization uses only primes <= largest_factor."""
    if n <= 0:
        return False
    for p in (2, 3, 5, 7, 11):
        if p > largest_factor:
            break
        while n % p == 0:
            n //= p
    return n == 1


def next_smooth(n: int, largest_factor: int = 5, enforce_even: bool = True) -> int:
    """Snap n upward to the next smooth FFT-friendly size.

    Mirrors src/core/functions.cpp:ReturnClosestFactorizedUpper. Defaults match
    the cisTEM convention of 5-smooth even sizes.
    """
    if enforce_even and n % 2:
        n += 1
    while not is_smooth(n, largest_factor):
        n += 2 if enforce_even else 1
    return n


@dataclass(frozen=True)
class WorkItem:
    mip: str
    psi: str
    theta: str
    phi: str
    defocus: str
    pixel_size_img: str
    reconstruction: str
    pixel_size: float
    threshold: float
    sweep_peak_threshold_scale: float
    sweep_original_peak_size: int
    sweep_padding_multiplier: int
    sweep_upsample_factor: int
    sweep_padding_mode: int
    binary: str
    log_dir: str

    @property
    def tag(self) -> str:
        return (
            f"{Path(self.mip).stem}"
            f"_s{self.sweep_peak_threshold_scale:.2f}"
            f"_o{self.sweep_original_peak_size}"
            f"_p{self.sweep_padding_multiplier}"
            f"_u{self.sweep_upsample_factor}"
            f"_m{self.sweep_padding_mode}"
        )


def derive_sibling(mip_path: str, type_name: str) -> str:
    """Replace the last 'mip' substring with `type_name` to get the sibling file."""
    if "mip" not in mip_path:
        raise ValueError(
            f"MIP path {mip_path!r} does not contain 'mip'; cannot derive {type_name}"
        )
    head, _, tail = mip_path.rpartition("mip")
    return f"{head}{type_name}{tail}"


def derive_histogram(mip_path: str, suffix: str = "_histogram.txt") -> str:
    """Best-effort guess: replace 'mip' with 'histogram' and .mrc with .txt."""
    candidate = mip_path.replace("mip", "histogram")
    candidate = re.sub(r"\.mrc$", ".txt", candidate)
    if Path(candidate).is_file():
        return candidate
    stem = re.sub(r"[_-]?mip.*$", "", mip_path)
    return f"{stem}{suffix}"


def parse_threshold_from_histogram(path: str) -> float:
    pattern = re.compile(r"Expected threshold\s*=\s*([0-9.+-eE]+)")
    with open(path, "r") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    raise RuntimeError(f"No 'Expected threshold = ...' header in {path}")


def read_pixel_size(mrc_path: str) -> float:
    with mrcfile.open(mrc_path, permissive=True, mode="r") as mrc:
        px = float(mrc.voxel_size.x)
    if px <= 0.0:
        raise RuntimeError(f"Invalid pixel size {px} in {mrc_path}")
    return px


def build_stdin(
    item: WorkItem,
    coords_out: str,
    result_image_out: str,
    slab_out: str,
) -> str:
    """Assemble the line-per-prompt stdin payload for make_template_result.

    Order matches MakeTemplateResult::DoInteractiveUserInput when
    read_coordinates == false.
    """
    lines = [
        "No",                                # Read coordinates from file?
        item.mip,                             # Input MIP file
        item.psi,                             # Input psi file
        item.theta,                           # Input theta file
        item.phi,                             # Input phi file
        item.defocus,                         # Input defocus file
        item.pixel_size_img,                  # Input pixel size file
        coords_out,                           # Output x,y,z coordinate file
        f"{item.threshold:.6f}",              # Peak threshold
        f"{MIN_PEAK_RADIUS:.6f}",             # Min Peak Radius (px.)
        str(RESULT_NUMBER),                   # Result number to process
        item.reconstruction,                  # Input template reconstruction
        result_image_out,                     # Output 2D projection montage
        slab_out,                             # Output slab volume montage
        f"{SLAB_THICKNESS:.6f}",              # Sample thickness (A)
        f"{item.pixel_size:.6f}",             # Pixel size of images (A)
        f"{BINNING_FACTOR:.6f}",              # Binning factor for slab
        "Yes",                                # Use peak sampling correction
    ]
    return "\n".join(lines) + "\n"


def count_peaks(coord_file: str) -> int:
    if not Path(coord_file).is_file():
        return -1
    n = 0
    with open(coord_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            n += 1
    return n


PEAK_FIELDS = (
    "x", "y", "subpixel_dx_px", "subpixel_dy_px",
    "height_orig", "height_upsampled",
    "upsample_status",
    "fwhm_valid", "fwhm_x_A", "fwhm_y_A",
    "ann_inner_A", "ann_outer_A",
    "ann_min", "ann_med", "ann_max",
)


def parse_diag_rows(path: str) -> list[dict]:
    """Parse the per-peak diagnostic sidecar written by make_template_result.

    Returns ONE dict per peak (no aggregation). Columns:
      x, y                               - integer MIP pixel coordinates
      subpixel_dx_px, subpixel_dy_px     - sub-pixel offsets in original MIP px
      height_orig, height_upsampled      - raw correlation values
      fwhm_valid                         - 1 if FWHM measured, 0 if fallback used
      fwhm_x_A, fwhm_y_A                 - FWHM in Angstroms (-1 when invalid)
      ann_inner_A, ann_outer_A           - annulus radii used for THIS peak (Angstroms)
      ann_min, ann_med, ann_max          - background stats in the annulus (raw values)

    Empty list if file is missing or has no data rows.
    """
    if not Path(path).is_file():
        return []
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("x,y,"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 15:
                continue
            try:
                rows.append({
                    "x": int(parts[0]),
                    "y": int(parts[1]),
                    "subpixel_dx_px": float(parts[2]),
                    "subpixel_dy_px": float(parts[3]),
                    "height_orig": float(parts[4]),
                    "height_upsampled": float(parts[5]),
                    "upsample_status": int(parts[6]),
                    "fwhm_valid": int(parts[7]),
                    "fwhm_x_A": float(parts[8]),
                    "fwhm_y_A": float(parts[9]),
                    "ann_inner_A": float(parts[10]),
                    "ann_outer_A": float(parts[11]),
                    "ann_min": float(parts[12]),
                    "ann_med": float(parts[13]),
                    "ann_max": float(parts[14]),
                })
            except ValueError:
                continue
    return rows


def run_one(item: WorkItem) -> tuple[dict, list[dict]]:
    """Worker entry point. Runs in a child process.

    Returns (iteration_metadata, list_of_per_peak_rows). The metadata dict
    carries the iteration-level fields (tag, sweep params, returncode, etc.)
    and is merged into each per-peak row by the main process when writing the
    CSV. On run failure the per-peak list is empty - the main process still
    writes one record (with empty peak fields) so failures are visible.
    """
    log_path = str(Path(item.log_dir) / f"{item.tag}.log")

    with tempfile.TemporaryDirectory(prefix="peak_sweep_") as tmpdir:
        coords_out = str(Path(tmpdir) / "coordinates.txt")
        result_image_out = str(Path(tmpdir) / "result.mrc")
        slab_out = str(Path(tmpdir) / "slab.mrc")
        diag_out = str(Path(tmpdir) / "peakdiag.csv")

        cmd = [
            item.binary,
            f"--sweep-peak-threshold-scale={item.sweep_peak_threshold_scale}",
            f"--sweep-original-peak-size={item.sweep_original_peak_size}",
            f"--sweep-padding-multiplier={item.sweep_padding_multiplier}",
            f"--sweep-upsample-factor={item.sweep_upsample_factor}",
            f"--sweep-padding-mode={item.sweep_padding_mode}",
            f"--sweep-width-fraction={WIDTH_FRACTION}",
            f"--sweep-annulus-inner-fwhm-mult={ANNULUS_INNER_FWHM_MULT}",
            f"--sweep-annulus-outer-fwhm-mult={ANNULUS_OUTER_FWHM_MULT}",
            f"--sweep-annulus-fallback-inner-px={ANNULUS_FALLBACK_INNER_PX}",
            f"--sweep-annulus-fallback-outer-px={ANNULUS_FALLBACK_OUTER_PX}",
            f"--sweep-diag-out={diag_out}",
        ]
        stdin_text = build_stdin(item, coords_out, result_image_out, slab_out)

        with open(log_path, "w") as logfh:
            proc = subprocess.run(
                cmd,
                input=stdin_text,
                stdout=logfh,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                cwd=tmpdir,
            )
        rc = proc.returncode
        n_peaks = count_peaks(coords_out) if rc == 0 else -1
        peak_rows = parse_diag_rows(diag_out) if rc == 0 else []

    meta = {
        "mip_path": item.mip,
        "pixel_size": item.pixel_size,
        "wanted_threshold": item.threshold,
        "sweep_peak_threshold_scale": item.sweep_peak_threshold_scale,
        "sweep_original_peak_size": item.sweep_original_peak_size,
        "sweep_padding_multiplier": item.sweep_padding_multiplier,
        "sweep_upsample_factor": item.sweep_upsample_factor,
        "sweep_padding_mode": item.sweep_padding_mode,
        "n_peaks": n_peaks,
        "returncode": rc,
        "log": log_path,
        "tag": item.tag,
    }
    return meta, peak_rows


def parameter_grid() -> list[dict]:
    """Inner sweep grid. Defaults to a single production-equivalent point so the
    smoke test exercises the full pipeline without combinatorial fanout. Expand
    by editing one of the inner lists below once the single-point run is sane.

    Combinations whose derived sizes (padded_peak_size, upsample_peak_size) are
    not 5-smooth even integers are dropped - those would still work in the C++
    method but with poor FFT performance. The C++ method does not block them.
    """
    grid = []
    seen = set()
    for scale, orig, padmul, upfac, mode in product(
        [0.85],                                                          # peak_threshold_scale
        [4, 6, 8, 12, 16],                                                # original_peak_size (sweep)
        [2, 3],                                                           # padding multiplier (sweep)
        [2, 4, 8, 16],                                                    # upsample factor (sweep)
        [PADDING_MODE_MIRROR],                                            # padding mode
    ):
        # Skip padding-mode variations when upsampling is disabled (they all
        # produce identical results since the padding never gets used).
        if scale == 1.0 and mode != PADDING_MODE_MIRROR:
            continue
        padded = padmul * orig
        upsampled = upfac * padded
        if not (orig % 2 == 0 and is_smooth(padded, 7) and is_smooth(upsampled, 7)):
            continue
        key = (scale, orig, padmul, upfac, mode)
        if key in seen:
            continue
        seen.add(key)
        grid.append({
            "sweep_peak_threshold_scale": scale,
            "sweep_original_peak_size": orig,
            "sweep_padding_multiplier": padmul,
            "sweep_upsample_factor": upfac,
            "sweep_padding_mode": mode,
        })
    return grid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reconstruction", required=True, help="Input template reconstruction (.mrc)")
    ap.add_argument("--mip-list", required=True, help="File with one input MIP path per line")
    ap.add_argument("--output-csv", default="peak_sweep_results.csv", help="Aggregated results CSV")
    ap.add_argument("--binary", default="make_template_result",
                    help="make_template_result binary path or name on PATH")
    ap.add_argument("--log-dir", default="sweep_logs", help="Per-iteration stdout/stderr log directory")
    ap.add_argument("--n-procs", type=int, default=1,
                    help="Number of worker processes (default 1; iterations are independent across image x params)")
    args = ap.parse_args()

    if not Path(args.reconstruction).is_file():
        print(f"ERROR: reconstruction not found: {args.reconstruction}", file=sys.stderr)
        return 2

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    with open(args.mip_list, "r") as fh:
        mip_paths = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if not mip_paths:
        print(f"ERROR: no MIP paths read from {args.mip_list}", file=sys.stderr)
        return 2

    grid = parameter_grid()
    print(f"Sweeping {len(grid)} parameter combinations across {len(mip_paths)} images "
          f"= {len(grid) * len(mip_paths)} total runs (n_procs={args.n_procs})")

    # Resolve per-image metadata once (pixel size, threshold, sibling paths).
    work_items: list[WorkItem] = []
    for mip in mip_paths:
        try:
            sibs = {t: derive_sibling(mip, t) for t in SIBLING_TYPES}
            hist = derive_histogram(mip)
            pixel_size = read_pixel_size(mip)
            threshold = parse_threshold_from_histogram(hist)
        except Exception as exc:
            print(f"[skip] {mip}: {exc}", file=sys.stderr)
            continue

        for params in grid:
            work_items.append(WorkItem(
                mip=mip,
                psi=sibs["psi"],
                theta=sibs["theta"],
                phi=sibs["phi"],
                defocus=sibs["defocus"],
                pixel_size_img=sibs["pixel_size"],
                reconstruction=args.reconstruction,
                pixel_size=pixel_size,
                threshold=threshold,
                sweep_peak_threshold_scale=params["sweep_peak_threshold_scale"],
                sweep_original_peak_size=params["sweep_original_peak_size"],
                sweep_padding_multiplier=params["sweep_padding_multiplier"],
                sweep_upsample_factor=params["sweep_upsample_factor"],
                sweep_padding_mode=params["sweep_padding_mode"],
                binary=args.binary,
                log_dir=args.log_dir,
            ))

    if not work_items:
        print("No runnable work items after metadata resolution.", file=sys.stderr)
        return 2

    iteration_fields = [
        "mip_path", "pixel_size", "wanted_threshold",
        "sweep_peak_threshold_scale", "sweep_original_peak_size",
        "sweep_padding_multiplier", "sweep_upsample_factor", "sweep_padding_mode",
        "n_peaks", "returncode", "log", "tag",
    ]
    fieldnames = iteration_fields + list(PEAK_FIELDS)
    csv_fh = open(args.output_csv, "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    def emit(meta: dict, peaks: list[dict]) -> None:
        if peaks:
            for p in peaks:
                row = {**meta, **p}
                writer.writerow(row)
        else:
            # Failure or zero-peak iteration: still write one row so the iteration
            # is visible in the CSV with empty peak fields.
            writer.writerow(meta)
        csv_fh.flush()

    completed = 0
    total = len(work_items)
    if args.n_procs <= 1:
        # Inline path keeps debugging easy and matches process-level semantics.
        for item in work_items:
            meta, peaks = run_one(item)
            emit(meta, peaks)
            completed += 1
            print(f"[{completed}/{total}] {meta['tag']}: rc={meta['returncode']} "
                  f"peaks={meta['n_peaks']} rows_emitted={len(peaks) or 1}")
    else:
        with ProcessPoolExecutor(max_workers=args.n_procs) as pool:
            futures = {pool.submit(run_one, item): item for item in work_items}
            for fut in as_completed(futures):
                try:
                    meta, peaks = fut.result()
                except Exception as exc:
                    item = futures[fut]
                    print(f"[error] {item.tag}: {exc}", file=sys.stderr)
                    meta = {
                        "mip_path": item.mip,
                        "pixel_size": item.pixel_size,
                        "wanted_threshold": item.threshold,
                        "sweep_peak_threshold_scale": item.sweep_peak_threshold_scale,
                        "sweep_original_peak_size": item.sweep_original_peak_size,
                        "sweep_padding_multiplier": item.sweep_padding_multiplier,
                        "sweep_upsample_factor": item.sweep_upsample_factor,
                        "sweep_padding_mode": item.sweep_padding_mode,
                        "n_peaks": -1,
                        "returncode": -1,
                        "log": "",
                        "tag": item.tag,
                    }
                    peaks = []
                emit(meta, peaks)
                completed += 1
                print(f"[{completed}/{total}] {meta['tag']}: rc={meta['returncode']} "
                      f"peaks={meta['n_peaks']} rows_emitted={len(peaks) or 1}")

    csv_fh.close()
    print(f"Done. Results -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
