import os
from datetime import datetime, timezone
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as scim
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from scipy.optimize import curve_fit


def _write_fits_product(
    data: np.ndarray,
    save_path: str | os.PathLike,
    header: Header | None = None,
    source_image: str | os.PathLike | None = None,
    history: Sequence[str] | None = None,
    extra_keywords: dict[str, tuple[object, str]] | None = None,
) -> None:
    """Write a FITS image with optional provenance and metadata."""
    output_header = Header()
    if header is not None:
        output_header.update(WCS(header).to_header())
    if source_image is not None:
        output_header["INPUTIMG"] = (os.fspath(source_image), "Source image")
    if extra_keywords is not None:
        for key, value in extra_keywords.items():
            output_header[key] = value
    output_header["DATE"] = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "File creation date",
    )
    if history is not None:
        for entry in history:
            output_header.add_history(entry)
    fits.writeto(os.fspath(save_path), data, header=output_header, overwrite=True)


def ggm(
    sigma: float,
    image: str | os.PathLike | None = None,
    data: np.ndarray | None = None,
    header: Header | None = None,
    save_path: str | os.PathLike | None = None,
) -> np.ndarray:
    """Apply a Gaussian filter and optionally write a FITS result."""
    if image is not None:
        if data is not None or header is not None:
            raise ValueError("Pass either `image` or `data`/`header`, not both.")
        data = np.array(fits.getdata(image), dtype=np.float32)
        header = fits.getheader(image)

    if data is None:
        raise ValueError("Pass either `image` or `data`.")
    data[data <= 0] = np.mean(data)
    data = np.arcsinh(data)
    result = scim.gaussian_filter(data, sigma=sigma)

    if save_path is not None:
        if image is not None:
            source_image = image
        else:
            source_image = None
        output_dir = os.fspath(save_path)
        os.makedirs(output_dir, exist_ok=True)
        output_name = f"ggm_{sigma}.fits"
        _write_fits_product(
            result,
            os.path.join(output_dir, output_name),
            header=header,
            source_image=source_image,
            history=[f"Applied gaussian_filter with sigma={sigma}"],
            extra_keywords={
                "GGM": (True, "Image produced by ggm"),
                "GGMSIG": (float(sigma), "Gaussian filter sigma in pixels"),
            },
        )

    return result


def make_radial_mask(
    img_array: np.ndarray,
    radii: tuple[float, float],
    center_pixel: tuple[float, float] | None = None,
) -> np.ndarray:
    """Create a boolean mask for pixels whose radius falls within bounds."""
    len_y, len_x = img_array.shape
    if center_pixel is None:
        center_pixel = (len_x / 2.0, len_y / 2.0)

    x = np.arange(len_x, dtype=np.float32) - center_pixel[0]
    y = np.arange(len_y, dtype=np.float32) - center_pixel[1]
    x_grid, y_grid = np.meshgrid(x, y)
    radius = np.sqrt(x_grid**2 + y_grid**2)

    return (radius >= radii[0]) & (radius <= radii[1])


def _radius_map(
    shape: tuple[int, int],
    center_pixel: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return the radius at each pixel for an image shape."""
    len_y, len_x = shape
    if center_pixel is None:
        center_pixel = (len_x / 2.0, len_y / 2.0)

    x = np.arange(len_x, dtype=np.float32) - center_pixel[0]
    y = np.arange(len_y, dtype=np.float32) - center_pixel[1]
    x_grid, y_grid = np.meshgrid(x, y)
    return np.sqrt(x_grid**2 + y_grid**2)


def _load_image_array(infile: str | os.PathLike | np.ndarray) -> np.ndarray:
    """Load an image from a FITS path or array-like input."""
    if isinstance(infile, (str, os.PathLike)):
        return np.array(fits.getdata(infile), dtype=np.float32)
    return np.array(infile, dtype=np.float32, copy=False)


def _load_image_and_header(
    infile: str | os.PathLike | np.ndarray,
) -> tuple[np.ndarray, Header | None]:
    """Load an image array and return a FITS header when available."""
    if isinstance(infile, (str, os.PathLike)):
        return (
            np.array(fits.getdata(infile), dtype=np.float32),
            fits.getheader(infile),
        )
    return np.array(infile, dtype=np.float32, copy=False), None


def _counts_to_snr(
    counts: np.ndarray,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Convert counts and optional background into an approximate per-pixel S/N."""
    if background is None:
        return np.sqrt(np.clip(counts, a_min=0.0, a_max=None))

    background = np.clip(background, a_min=0.0, a_max=None)
    signal = np.clip(counts - background, a_min=0.0, a_max=None)
    variance = np.clip(counts + background, a_min=1e-6, a_max=None)
    return signal / np.sqrt(variance)


def adaptive_kernel_sigma_map(
    counts_image: str | os.PathLike | np.ndarray,
    target_counts: float | None = None,
    target_snr: float | None = None,
    background_image: str | os.PathLike | np.ndarray | None = None,
    min_sigma: float = 1.0,
    max_sigma: float = 32.0,
    sigma_step: float = 1.0,
    sigma_to_radius_factor: float = 3.0,
    exposure_image: str | os.PathLike | np.ndarray | None = None,
    min_exposure: float = 0.0,
) -> np.ndarray:
    """Choose the first Gaussian scale whose enclosed counts or S/N meets the target."""
    counts = _load_image_array(counts_image)
    background = (
        None if background_image is None else _load_image_array(background_image)
    )
    exposure = None if exposure_image is None else _load_image_array(exposure_image)

    if background is not None and background.shape != counts.shape:
        raise ValueError("`background_image` must match `counts_image` shape.")
    if exposure is not None and exposure.shape != counts.shape:
        raise ValueError("`exposure_image` must match `counts_image` shape.")
    if target_counts is None and target_snr is None:
        raise ValueError("Pass `target_counts` or `target_snr`.")
    if target_counts is not None and target_snr is not None:
        raise ValueError("Pass only one of `target_counts` or `target_snr`.")
    if target_counts is not None and target_counts <= 0:
        raise ValueError("`target_counts` must be positive.")
    if target_snr is not None and target_snr <= 0:
        raise ValueError("`target_snr` must be positive.")
    if min_sigma <= 0 or max_sigma < min_sigma or sigma_step <= 0:
        raise ValueError("Sigma bounds must satisfy 0 < min_sigma <= max_sigma.")
    if sigma_to_radius_factor <= 0:
        raise ValueError("`sigma_to_radius_factor` must be positive.")

    sigma_values = np.arange(min_sigma, max_sigma + 0.5 * sigma_step, sigma_step)
    sigma_map = np.full(counts.shape, max_sigma, dtype=np.float32)
    assigned = np.zeros(counts.shape, dtype=bool)
    valid_mask = np.isfinite(counts)

    if exposure is not None:
        valid_mask &= np.isfinite(exposure) & (exposure > min_exposure)
    if background is not None:
        valid_mask &= np.isfinite(background)

    snr_image = _counts_to_snr(counts, background=background)
    counts_for_scale = np.clip(counts, a_min=0.0, a_max=None)
    if background is not None:
        counts_for_scale = np.clip(counts - background, a_min=0.0, a_max=None)

    for sigma in sigma_values:
        radius_pixels = sigma * sigma_to_radius_factor
        enclosed_counts = scim.gaussian_filter(
            counts_for_scale,
            sigma=sigma,
            mode="nearest",
        ) * (2.0 * np.pi * radius_pixels**2)
        enclosed_snr = scim.gaussian_filter(
            snr_image,
            sigma=sigma,
            mode="nearest",
        ) * np.sqrt(2.0 * np.pi * sigma**2)

        if target_counts is not None:
            reached = enclosed_counts >= target_counts
        else:
            reached = enclosed_snr >= target_snr

        select = valid_mask & (~assigned) & reached
        sigma_map[select] = float(sigma)
        assigned[select] = True

    sigma_map[~valid_mask] = np.nan
    return sigma_map


def adaptive_smooth_image(
    image: str | os.PathLike | np.ndarray,
    sigma_map: np.ndarray,
    sigma_values: Sequence[float] | None = None,
) -> np.ndarray:
    """Approximate adaptive smoothing by blending fixed-scale smooths per local sigma."""
    data = _load_image_array(image)
    if data.shape != sigma_map.shape:
        raise ValueError("`sigma_map` must match image shape.")

    if sigma_values is None:
        sigma_values = np.unique(sigma_map[np.isfinite(sigma_map)])
    sigma_array = np.asarray(
        sorted(float(value) for value in sigma_values), dtype=float
    )
    if sigma_array.size == 0:
        return np.full_like(data, np.nan, dtype=np.float32)

    valid_data = np.isfinite(data)
    filled_data = np.where(valid_data, data, 0.0).astype(np.float32)
    valid_weights = valid_data.astype(np.float32)
    smoothed_layers: list[np.ndarray] = []
    for sigma in sigma_array:
        smoothed_sum = scim.gaussian_filter(filled_data, sigma=sigma, mode="nearest")
        smoothed_weight = scim.gaussian_filter(
            valid_weights,
            sigma=sigma,
            mode="nearest",
        )
        smoothed_layer = np.divide(
            smoothed_sum,
            smoothed_weight,
            out=np.full_like(filled_data, np.nan, dtype=np.float32),
            where=smoothed_weight > 1e-6,
        )
        smoothed_layers.append(smoothed_layer)
    smoothed_stack = np.stack(smoothed_layers, axis=0)

    sigma_clipped = np.clip(sigma_map, sigma_array[0], sigma_array[-1])
    upper_index = np.searchsorted(sigma_array, sigma_clipped, side="left")
    upper_index = np.clip(upper_index, 0, len(sigma_array) - 1)
    lower_index = np.clip(upper_index - 1, 0, len(sigma_array) - 1)

    lower_sigma = sigma_array[lower_index]
    upper_sigma = sigma_array[upper_index]
    sigma_span = upper_sigma - lower_sigma
    upper_weight = np.divide(
        sigma_clipped - lower_sigma,
        sigma_span,
        out=np.zeros_like(sigma_clipped, dtype=np.float32),
        where=sigma_span > 0,
    )
    lower_weight = 1.0 - upper_weight

    lower_values = np.take_along_axis(
        smoothed_stack,
        lower_index[np.newaxis, ...],
        axis=0,
    )[0]
    upper_values = np.take_along_axis(
        smoothed_stack,
        upper_index[np.newaxis, ...],
        axis=0,
    )[0]
    smoothed = lower_values * lower_weight + upper_values * upper_weight
    smoothed[~np.isfinite(sigma_map)] = np.nan
    return smoothed.astype(np.float32)


def adaptive_ggm(
    counts_image: str | os.PathLike | np.ndarray,
    exposure_image: str | os.PathLike | np.ndarray,
    background_image: str | os.PathLike | np.ndarray | None = None,
    target_counts: float | None = 1024.0,
    target_snr: float | None = None,
    min_sigma: float = 1.0,
    max_sigma: float = 32.0,
    sigma_step: float = 1.0,
    sigma_to_radius_factor: float = 3.0,
    use_log: bool = True,
    min_exposure_fraction: float = 0.05,
    save_path: str | os.PathLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an adaptive GGM map from counts and exposure products."""
    counts, counts_header = _load_image_and_header(counts_image)
    exposure = _load_image_array(exposure_image)
    if counts.shape != exposure.shape:
        raise ValueError(
            "`counts_image` and `exposure_image` must have the same shape."
        )

    background = (
        None if background_image is None else _load_image_array(background_image)
    )
    if background is not None and background.shape != counts.shape:
        raise ValueError("`background_image` must match `counts_image` shape.")
    if not 0.0 <= min_exposure_fraction < 1.0:
        raise ValueError("`min_exposure_fraction` must be in [0, 1).")
    if target_counts is not None and target_snr is not None:
        raise ValueError("Pass only one of `target_counts` or `target_snr`.")

    finite_exposure = exposure[np.isfinite(exposure) & (exposure > 0)]
    if finite_exposure.size == 0:
        raise ValueError("`exposure_image` contains no positive finite pixels.")
    min_exposure = float(np.max(finite_exposure) * min_exposure_fraction)

    intensity_numerator = counts if background is None else counts - background
    intensity = np.divide(
        intensity_numerator,
        exposure,
        out=np.full_like(counts, np.nan, dtype=np.float32),
        where=np.isfinite(exposure) & (exposure > min_exposure),
    )
    intensity[intensity <= 0] = np.nan

    sigma_map = adaptive_kernel_sigma_map(
        counts_image=counts,
        target_counts=target_counts,
        target_snr=target_snr,
        background_image=background,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_step=sigma_step,
        sigma_to_radius_factor=sigma_to_radius_factor,
        exposure_image=exposure,
        min_exposure=min_exposure,
    )

    smoothed_intensity = adaptive_smooth_image(intensity, sigma_map=sigma_map)
    gradient_input = np.array(smoothed_intensity, dtype=np.float32, copy=True)
    gradient_input[~np.isfinite(gradient_input)] = np.nan
    if use_log:
        positive = np.isfinite(gradient_input) & (gradient_input > 0)
        gradient_input[positive] = np.log10(gradient_input[positive])
        gradient_input[~positive] = np.nan

    filled_input = np.array(gradient_input, copy=True)
    fill_value = float(np.nanmedian(filled_input[np.isfinite(filled_input)]))
    if not np.isfinite(fill_value):
        fill_value = 0.0
    filled_input[~np.isfinite(filled_input)] = fill_value

    ggm_map = np.hypot(
        scim.sobel(filled_input, axis=1, mode="nearest"),
        scim.sobel(filled_input, axis=0, mode="nearest"),
    ).astype(np.float32)
    invalid_mask = ~np.isfinite(smoothed_intensity) | ~np.isfinite(sigma_map)
    ggm_map[invalid_mask] = np.nan

    if save_path is not None:
        os.makedirs(os.path.dirname(os.fspath(save_path)) or ".", exist_ok=True)
        metadata = {
            "AGGM": (True, "Image produced by adaptive_ggm"),
            "MINSIG": (float(min_sigma), "Minimum adaptive Gaussian sigma in pixels"),
            "MAXSIG": (float(max_sigma), "Maximum adaptive Gaussian sigma in pixels"),
            "SIGSTEP": (float(sigma_step), "Adaptive sigma grid spacing in pixels"),
            "LOGGGM": (bool(use_log), "Computed GGM on log10(smoothed intensity)"),
        }
        if target_counts is not None:
            metadata["TGCOUNT"] = (
                float(target_counts),
                "Target enclosed counts per adaptive kernel",
            )
        if target_snr is not None:
            metadata["TGSNR"] = (
                float(target_snr),
                "Target enclosed signal-to-noise per adaptive kernel",
            )
        _write_fits_product(
            ggm_map,
            save_path,
            header=counts_header,
            source_image=counts_image
            if isinstance(counts_image, (str, os.PathLike))
            else None,
            history=[
                "Computed adaptive kernel sigma map from counts statistics",
                "Built exposure-corrected adaptive smoothing product",
                "Computed gradient magnitude from adaptively smoothed image",
            ],
            extra_keywords=metadata,
        )

    return ggm_map, smoothed_intensity, sigma_map


def azimuthal_profile(
    image: str | os.PathLike | np.ndarray,
    center_pixel: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the azimuthally averaged profile as integer-radius bins."""
    data = _load_image_array(image)
    radius_map = _radius_map(data.shape, center_pixel=center_pixel)
    radius_index = np.floor(radius_map).astype(np.int32)

    flat_radius = radius_index.ravel()
    flat_data = data.ravel()
    counts = np.bincount(flat_radius)
    sums = np.bincount(flat_radius, weights=flat_data)
    valid = counts > 0

    radii = np.arange(len(counts), dtype=np.float32)[valid]
    profile = (sums[valid] / counts[valid]).astype(np.float32)
    return radii, profile


def background_radius_from_profile(
    image: str | os.PathLike | np.ndarray,
    center_pixel: tuple[float, float] | None = None,
    background_sigma: float = 3.0,
) -> float:
    """Return the radius where the azimuthal profile first drops to background + N sigma."""
    radii, profile = azimuthal_profile(image, center_pixel=center_pixel)
    if profile.size == 0:
        return 0.0

    tail_start = max(int(0.8 * profile.size), 1)
    tail_profile = profile[tail_start:]
    background_level = float(np.median(tail_profile))
    background_scatter = float(
        1.4826 * np.median(np.abs(tail_profile - background_level))
    )
    threshold = background_level + background_sigma * background_scatter

    below_threshold = np.flatnonzero(profile <= threshold)
    if below_threshold.size == 0:
        return float(radii[-1])

    valid_indices = below_threshold[radii[below_threshold] > 0]
    if valid_indices.size == 0:
        return float(radii[-1])
    return float(radii[valid_indices[0]])


def double_beta_profile(
    radius: np.ndarray,
    norm_1: float,
    core_radius_1: float,
    beta_1: float,
    norm_2: float,
    core_radius_2: float,
    beta_2: float,
    background: float,
) -> np.ndarray:
    """Projected double-beta surface-brightness profile with constant background."""
    component_1 = norm_1 * (1.0 + (radius / core_radius_1) ** 2) ** (
        -3.0 * beta_1 + 0.5
    )
    component_2 = norm_2 * (1.0 + (radius / core_radius_2) ** 2) ** (
        -3.0 * beta_2 + 0.5
    )
    return component_1 + component_2 + background


def double_beta_components(
    radius: np.ndarray,
    norm_1: float,
    core_radius_1: float,
    beta_1: float,
    norm_2: float,
    core_radius_2: float,
    beta_2: float,
    background: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the two projected beta components and constant background."""
    component_1 = norm_1 * (1.0 + (radius / core_radius_1) ** 2) ** (
        -3.0 * beta_1 + 0.5
    )
    component_2 = norm_2 * (1.0 + (radius / core_radius_2) ** 2) ** (
        -3.0 * beta_2 + 0.5
    )
    background_component = np.full_like(radius, background, dtype=np.float64)
    return component_1, component_2, background_component


def fit_double_beta_model(
    image: str | os.PathLike | np.ndarray,
    center_pixel: tuple[float, float] | None = None,
    max_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a double-beta profile to the raw azimuthal surface-brightness profile."""
    radii, profile = azimuthal_profile(image, center_pixel=center_pixel)
    valid = np.isfinite(radii) & np.isfinite(profile)
    if max_radius is not None:
        valid &= radii <= max_radius

    fit_radii = radii[valid]
    fit_profile = profile[valid]
    if fit_radii.size < 8:
        raise ValueError("Not enough radial bins to fit the double-beta model.")

    background_guess = float(
        np.median(fit_profile[max(int(0.8 * fit_profile.size), 1) :])
    )
    signal_scale = max(float(np.max(fit_profile) - background_guess), 1e-6)
    radius_scale = max(float(fit_radii[-1]), 10.0)

    initial_guess = np.array(
        [
            0.7 * signal_scale,
            max(radius_scale * 0.05, 3.0),
            0.7,
            0.3 * signal_scale,
            max(radius_scale * 0.2, 10.0),
            0.7,
            background_guess,
        ],
        dtype=np.float64,
    )
    lower_bounds = np.array(
        [0.0, 1e-3, 0.34, 0.0, 1e-3, 0.34, np.min(fit_profile)], dtype=np.float64
    )
    upper_bounds = np.array(
        [
            np.inf,
            max(radius_scale * 2.0, 10.0),
            3.0,
            np.inf,
            max(radius_scale * 2.0, 10.0),
            3.0,
            np.max(fit_profile),
        ],
        dtype=np.float64,
    )

    fit_weights = np.sqrt(np.clip(np.abs(fit_profile), a_min=1e-6, a_max=None))
    best_fit, covariance = curve_fit(
        double_beta_profile,
        fit_radii,
        fit_profile,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        sigma=fit_weights,
        absolute_sigma=False,
        maxfev=50000,
    )
    return fit_radii, fit_profile, best_fit


def plot_double_beta_fit(
    image: str | os.PathLike | np.ndarray,
    output_path: str | os.PathLike,
    center_pixel: tuple[float, float] | None = None,
    max_radius: float | None = None,
) -> np.ndarray:
    """Fit and save a visualization of the raw azimuthal profile and double-beta model."""
    fit_radii, fit_profile, best_fit = fit_double_beta_model(
        image,
        center_pixel=center_pixel,
        max_radius=max_radius,
    )
    component_1, component_2, background_component = double_beta_components(
        fit_radii,
        *best_fit,
    )
    model_profile = component_1 + component_2 + background_component

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(fit_radii, fit_profile, color="black", linewidth=1.0, label="Profile")
    axis.plot(
        fit_radii,
        model_profile,
        color="tab:red",
        linewidth=2.0,
        label="Double-beta fit",
    )
    axis.plot(
        fit_radii,
        component_1 + background_component,
        color="tab:blue",
        linestyle="--",
        linewidth=1.5,
        label="Component 1 + background",
    )
    axis.plot(
        fit_radii,
        component_2 + background_component,
        color="tab:green",
        linestyle="--",
        linewidth=1.5,
        label="Component 2 + background",
    )
    axis.plot(
        fit_radii,
        background_component,
        color="tab:gray",
        linestyle=":",
        linewidth=1.2,
        label="Background",
    )
    radius_3rc2 = 3.0 * best_fit[4]
    axis.axvline(
        radius_3rc2,
        color="tab:purple",
        linestyle="-.",
        linewidth=1.3,
        label="3 x rc2",
    )
    axis.set_xlabel("Radius (pixels)")
    axis.set_ylabel("Azimuthal mean surface brightness")
    axis.set_yscale("log")
    axis.legend()
    axis.set_title("Double-beta fit to radial profile")
    parameter_text = "\n".join(
        [
            rf"$A_1 = {best_fit[0]:.3g}$",
            rf"$r_{{c,1}} = {best_fit[1]:.3g}$",
            rf"$\beta_1 = {best_fit[2]:.3g}$",
            rf"$A_2 = {best_fit[3]:.3g}$",
            rf"$r_{{c,2}} = {best_fit[4]:.3g}$",
            rf"$3r_{{c,2}} = {radius_3rc2:.3g}$",
            rf"$\beta_2 = {best_fit[5]:.3g}$",
            rf"$b_{{\mathrm{{kg}}}} = {best_fit[6]:.3g}$",
        ]
    )
    axis.text(
        0.98,
        0.98,
        parameter_text,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    figure.tight_layout()
    os.makedirs(os.path.dirname(os.fspath(output_path)) or ".", exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    return best_fit


def auto_radius_bins_equal_snr(
    image: str | os.PathLike | np.ndarray,
    n_bins: int,
    center_pixel: tuple[float, float] | None = None,
    background_sigma: float = 3.0,
) -> list[float]:
    """Choose radius bin edges so each annulus has similar cumulative SNR."""
    if n_bins < 1:
        raise ValueError("`n_bins` must be at least 1.")

    data = _load_image_array(image)
    max_radius = background_radius_from_profile(
        data,
        center_pixel=center_pixel,
        background_sigma=background_sigma,
    )
    radius_map = _radius_map(data.shape, center_pixel=center_pixel).ravel()
    within_radius = radius_map <= max_radius
    radius_map = radius_map[within_radius]
    signal = np.clip(data.ravel()[within_radius], a_min=0.0, a_max=None)
    variance = np.clip(np.abs(data.ravel()[within_radius]), a_min=1e-6, a_max=None)

    order = np.argsort(radius_map)
    radius_sorted = radius_map[order]
    signal_sorted = signal[order]
    variance_sorted = variance[order]

    cumulative_signal = np.cumsum(signal_sorted)
    cumulative_variance = np.cumsum(variance_sorted)
    cumulative_snr = cumulative_signal / np.sqrt(cumulative_variance)

    total_snr = float(cumulative_snr[-1]) if cumulative_snr.size else 0.0
    if total_snr <= 0:
        max_radius = float(radius_sorted[-1]) if radius_sorted.size else 0.0
        return np.linspace(0.0, max_radius, n_bins, dtype=np.float32).tolist()

    radius_bins: list[float] = []
    for bin_idx in range(1, n_bins + 1):
        target_snr = total_snr * bin_idx / n_bins
        snr_index = int(np.searchsorted(cumulative_snr, target_snr, side="left"))
        snr_index = min(snr_index, len(radius_sorted) - 1)
        radius_bins.append(float(radius_sorted[snr_index]))

    return radius_bins


def auto_weight_bins(
    infiles: Sequence[str | os.PathLike | np.ndarray],
    radius_bins: Sequence[float],
    center_pixel: tuple[float, float] | None = None,
) -> list[list[float]]:
    """Choose per-image annular weights from per-bin gradient-energy scores."""
    if not infiles:
        raise ValueError("`infiles` must contain at least one image.")
    if len(radius_bins) == 0:
        raise ValueError("`radius_bins` must contain at least one value.")

    images = [_load_image_array(infile) for infile in infiles]
    first_shape = images[0].shape
    if any(image.shape != first_shape for image in images):
        raise ValueError("All input images must have the same shape.")

    radius_edges = np.asarray(radius_bins, dtype=np.float32)
    radius_map = _radius_map(first_shape, center_pixel=center_pixel)
    bin_index = np.searchsorted(radius_edges, radius_map, side="left")
    valid_mask = bin_index < len(radius_edges)

    scores = np.zeros((len(images), len(radius_edges)), dtype=np.float32)
    gradient_images = [
        scim.gaussian_gradient_magnitude(image, sigma=1.0) for image in images
    ]
    for rad_index in range(len(radius_edges)):
        radial_mask = valid_mask & (bin_index == rad_index)
        if not np.any(radial_mask):
            continue
        for image_index, (image, gradient_image) in enumerate(
            zip(images, gradient_images)
        ):
            values = image[radial_mask]
            gradient_values = gradient_image[radial_mask]
            noise = float(np.std(values))
            structure = float(np.mean(np.abs(gradient_values)))
            scores[image_index, rad_index] = structure / max(noise, 1e-6)

    score_sum = np.sum(scores, axis=0)
    normalized_scores = np.divide(
        scores,
        score_sum,
        out=np.full_like(scores, 1.0 / len(images)),
        where=score_sum > 0,
    )
    return normalized_scores.tolist()


def combine_ggf(
    img_dir: str | os.PathLike,
    infiles: Sequence[str | os.PathLike | np.ndarray],
    radius_bins: Sequence[float],
    weight_bins: Sequence[Sequence[float]],
    fits_file: str | os.PathLike | None = None,
    center_pixel: tuple[float, float] | None = None,
    background_radius: float | None = None,
    background_sigma: float | None = None,
) -> np.ndarray:
    """Combine multiple GGF images into a single weighted FITS product."""
    if not infiles:
        raise ValueError("`infiles` must contain at least one image.")
    if len(infiles) != len(weight_bins):
        raise ValueError("`weight_bins` must match the number of input images.")
    if len(radius_bins) == 0:
        raise ValueError("`radius_bins` must contain at least one value.")

    images: list[np.ndarray] = []
    for infile in infiles:
        if isinstance(infile, (str, os.PathLike)):
            image = np.array(fits.getdata(infile), dtype=np.float32)
        else:
            image = np.array(infile, dtype=np.float32, copy=False)
        images.append(image)

    first_shape = images[0].shape
    if any(image.shape != first_shape for image in images):
        raise ValueError("All input images must have the same shape.")
    if any(len(weight_bin) != len(radius_bins) for weight_bin in weight_bins):
        raise ValueError("Each weight bin list must match the length of `radius_bins`.")

    radius_edges = np.asarray(radius_bins, dtype=np.float32)
    radius_map = _radius_map(first_shape, center_pixel=center_pixel)
    bin_index = np.searchsorted(radius_edges, radius_map, side="left")
    valid_mask = bin_index < len(radius_edges)

    stack = np.stack(images, axis=0)
    weights = np.asarray(weight_bins, dtype=np.float32)
    weight_sum = np.sum(weights, axis=0)
    normalized_weights = np.divide(
        weights,
        weight_sum,
        out=np.zeros_like(weights),
        where=weight_sum > 0,
    )

    final_img = np.zeros(first_shape, dtype=np.float32)
    for rad_index in range(len(radius_edges)):
        radial_mask = valid_mask & (bin_index == rad_index)
        if not np.any(radial_mask) or weight_sum[rad_index] == 0:
            continue
        final_img[radial_mask] = np.tensordot(
            normalized_weights[:, rad_index],
            stack[:, radial_mask],
            axes=(0, 0),
        )

    output_dir = os.fspath(img_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_header = Header()
    if fits_file is not None:
        output_header.update(WCS(fits.getheader(fits_file)).to_header())
        output_header["INPUTIMG"] = (os.fspath(fits_file), "Source image")
    output_header["CGGF"] = (True, "Image produced by combine_ggf")
    output_header["NBIN"] = (len(radius_bins), "Number of radial bins")
    output_header["NCOMP"] = (len(images), "Number of combined images")
    if background_radius is not None:
        output_header["BGRAD"] = (
            float(background_radius),
            "Radius where azimuthal profile reaches background",
        )
    if background_sigma is not None:
        output_header["BGSIG"] = (
            float(background_sigma),
            "Sigma threshold above background for cutoff",
        )
    for index, radius_bin in enumerate(radius_bins, start=1):
        output_header[f"RBIN{index}"] = (
            float(radius_bin),
            f"Outer radius of bin {index} in pixels",
        )
    for image_index, weight_row in enumerate(weight_bins, start=1):
        for bin_index, weight_value in enumerate(weight_row, start=1):
            output_header[f"W{image_index}B{bin_index}"] = (
                float(weight_value),
                f"Weight image {image_index} in bin {bin_index}",
            )
    output_header["DATE"] = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "File creation date",
    )
    output_header.add_history("Combined weighted GGF images by radial bin")
    output_header.add_history(
        "Radius bins stored as RBINn; weights stored as WiBj keywords"
    )

    fits.writeto(
        os.path.join(output_dir, "GGF.fits"),
        final_img,
        header=output_header,
        overwrite=True,
    )

    return final_img


def auto_combine_ggf(
    img_dir: str | os.PathLike,
    infiles: Sequence[str | os.PathLike | np.ndarray],
    source_image: str | os.PathLike | np.ndarray,
    n_bins: int,
    fits_file: str | os.PathLike | None = None,
    center_pixel: tuple[float, float] | None = None,
    background_sigma: float = 3.0,
) -> tuple[np.ndarray, list[float], list[list[float]], float]:
    """Automatically choose equal-SNR bins and per-bin weights, then combine GGF."""
    background_radius = background_radius_from_profile(
        source_image,
        center_pixel=center_pixel,
        background_sigma=background_sigma,
    )
    radius_bins = auto_radius_bins_equal_snr(
        source_image,
        n_bins=n_bins,
        center_pixel=center_pixel,
        background_sigma=background_sigma,
    )
    weight_bins = auto_weight_bins(
        infiles,
        radius_bins=radius_bins,
        center_pixel=center_pixel,
    )
    result = combine_ggf(
        img_dir=img_dir,
        infiles=infiles,
        radius_bins=radius_bins,
        weight_bins=weight_bins,
        fits_file=fits_file,
        center_pixel=center_pixel,
        background_radius=background_radius,
        background_sigma=background_sigma,
    )
    return result, radius_bins, weight_bins, background_radius
