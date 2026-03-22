from typing import Annotated

from cyclopts import App, Parameter

from .core import adaptive_ggm, auto_combine_ggf, combine_ggf, ggm, plot_double_beta_fit

app = App(help="Apply a Gaussian filter to a FITS image.")


def _parse_sigma_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_weight_bins(value: str) -> list[list[float]]:
    return [_parse_float_list(group) for group in value.split(";") if group.strip()]


def _parse_center_pixel(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    x_value, y_value = value.split(",")
    return float(x_value.strip()), float(y_value.strip())


@app.default
def main(
    image: Annotated[
        str,
        Parameter(name=["--image", "-i"], help="Path to the input FITS image."),
    ],
    sigma: Annotated[
        str,
        Parameter(
            name=["--sigma", "-s"],
            help="Gaussian filter sigma in pixels. Use commas for multiple values.",
        ),
    ],
    save_path: Annotated[
        str | None,
        Parameter(
            name=["--save-path", "-o"],
            help="Optional output directory for ggm_{sigma}.fits.",
        ),
    ] = None,
) -> None:
    """Run ggm on a FITS image path."""
    for sigma_value in _parse_sigma_list(sigma):
        ggm(image=image, sigma=sigma_value, save_path=save_path)


@app.command(name="combine-ggf")
def combine_ggf_cli(
    img_dir: Annotated[
        str,
        Parameter(
            name=["--img-dir", "-d"],
            help="Directory where the combined FITS file will be written.",
        ),
    ],
    infiles: Annotated[
        str,
        Parameter(
            name=["--infiles", "-f"],
            help="Comma-separated list of input GGF FITS files.",
        ),
    ],
    radius_bins: Annotated[
        str,
        Parameter(
            name=["--radius-bins", "-r"],
            help="Comma-separated radial bin values.",
        ),
    ],
    weight_bins: Annotated[
        str,
        Parameter(
            name=["--weight-bins", "-w"],
            help="Semicolon-separated weight rows, each row comma-separated.",
        ),
    ],
    fits_file: Annotated[
        str | None,
        Parameter(
            name=["--fits-file"],
            help="Optional FITS file used for output WCS/header provenance.",
        ),
    ] = None,
    center: Annotated[
        str | None,
        Parameter(
            name=["--center", "-c"],
            help="Optional center pixel as x,y for radial binning.",
        ),
    ] = None,
) -> None:
    """Combine weighted GGF images across radial bins."""
    combine_ggf(
        img_dir=img_dir,
        infiles=_parse_csv_list(infiles),
        radius_bins=_parse_float_list(radius_bins),
        weight_bins=_parse_weight_bins(weight_bins),
        fits_file=fits_file,
        center_pixel=_parse_center_pixel(center),
    )


@app.command(name="auto-combine-ggf")
def auto_combine_ggf_cli(
    img_dir: Annotated[
        str,
        Parameter(
            name=["--img-dir", "-d"],
            help="Directory where the combined FITS file will be written.",
        ),
    ],
    infiles: Annotated[
        str,
        Parameter(
            name=["--infiles", "-f"],
            help="Comma-separated list of input GGF FITS files.",
        ),
    ],
    source_image: Annotated[
        str,
        Parameter(
            name=["--source-image", "-i"],
            help="Input FITS image used to derive equal-SNR radial bins.",
        ),
    ],
    n_bins: Annotated[
        int,
        Parameter(
            name=["--n-bins", "-n"],
            help="Number of equal-SNR radial bins to compute.",
        ),
    ],
    fits_file: Annotated[
        str | None,
        Parameter(
            name=["--fits-file"],
            help="Optional FITS file used for output WCS/header provenance.",
        ),
    ] = None,
    center: Annotated[
        str | None,
        Parameter(
            name=["--center", "-c"],
            help="Optional center pixel as x,y for radial binning.",
        ),
    ] = None,
    background_sigma: Annotated[
        float,
        Parameter(
            name=["--background-sigma"],
            help="Profile cutoff at background plus this many sigma.",
        ),
    ] = 3.0,
) -> None:
    """Automatically choose equal-SNR bins and combine GGF images."""
    _, radius_bins, weight_bins, background_radius = auto_combine_ggf(
        img_dir=img_dir,
        infiles=_parse_csv_list(infiles),
        source_image=source_image,
        n_bins=n_bins,
        fits_file=fits_file,
        center_pixel=_parse_center_pixel(center),
        background_sigma=background_sigma,
    )
    print("background_radius=", background_radius)
    print("radius_bins=", ",".join(str(value) for value in radius_bins))
    print(
        "weight_bins=",
        ";".join(
            ",".join(str(value) for value in weight_row) for weight_row in weight_bins
        ),
    )


@app.command(name="fit-double-beta")
def fit_double_beta_cli(
    image: Annotated[
        str,
        Parameter(name=["--image", "-i"], help="Input FITS image to profile and fit."),
    ],
    output_plot: Annotated[
        str,
        Parameter(
            name=["--output-plot", "-o"],
            help="Path to save the radial-profile fit visualization.",
        ),
    ],
    center: Annotated[
        str | None,
        Parameter(
            name=["--center", "-c"],
            help="Optional center pixel as x,y for the radial profile.",
        ),
    ] = None,
    max_radius: Annotated[
        float | None,
        Parameter(
            name=["--max-radius"],
            help="Optional maximum radius in pixels to include in the fit.",
        ),
    ] = None,
) -> None:
    """Fit a double-beta model to the azimuthal profile and save a plot."""
    best_fit = plot_double_beta_fit(
        image=image,
        output_path=output_plot,
        center_pixel=_parse_center_pixel(center),
        max_radius=max_radius,
    )
    print("norm_1=", best_fit[0])
    print("core_radius_1=", best_fit[1])
    print("beta_1=", best_fit[2])
    print("norm_2=", best_fit[3])
    print("core_radius_2=", best_fit[4])
    print("3core_radius_2=", 3.0 * best_fit[4])
    print("beta_2=", best_fit[5])
    print("background=", best_fit[6])


@app.command(name="adaptive-ggm")
def adaptive_ggm_cli(
    counts_image: Annotated[
        str,
        Parameter(
            name=["--counts-image"],
            help="Merged counts FITS image used for adaptive scale selection.",
        ),
    ],
    exposure_image: Annotated[
        str,
        Parameter(
            name=["--exposure-image"],
            help="Merged exposure FITS image used for intensity correction.",
        ),
    ],
    output_image: Annotated[
        str,
        Parameter(
            name=["--output-image", "-o"],
            help="Path to the output adaptive GGM FITS image.",
        ),
    ],
    background_image: Annotated[
        str | None,
        Parameter(
            name=["--background-image"],
            help="Optional merged background FITS image.",
        ),
    ] = None,
    target_counts: Annotated[
        float | None,
        Parameter(
            name=["--target-counts"],
            help="Target enclosed counts used to choose the local smoothing scale.",
        ),
    ] = 1024.0,
    target_snr: Annotated[
        float | None,
        Parameter(
            name=["--target-snr"],
            help="Alternative target signal-to-noise used for local smoothing scale.",
        ),
    ] = None,
    min_sigma: Annotated[
        float,
        Parameter(
            name=["--min-sigma"],
            help="Minimum Gaussian sigma in pixels for the adaptive scale grid.",
        ),
    ] = 1.0,
    max_sigma: Annotated[
        float,
        Parameter(
            name=["--max-sigma"],
            help="Maximum Gaussian sigma in pixels for the adaptive scale grid.",
        ),
    ] = 32.0,
    sigma_step: Annotated[
        float,
        Parameter(
            name=["--sigma-step"],
            help="Step size in pixels for the adaptive sigma grid.",
        ),
    ] = 1.0,
    sigma_to_radius_factor: Annotated[
        float,
        Parameter(
            name=["--sigma-to-radius-factor"],
            help="Approximate aperture radius in units of sigma.",
        ),
    ] = 3.0,
    min_exposure_fraction: Annotated[
        float,
        Parameter(
            name=["--min-exposure-fraction"],
            help="Mask exposures below this fraction of the maximum valid exposure.",
        ),
    ] = 0.05,
    log_scale: Annotated[
        bool,
        Parameter(
            name=["--log", "--linear"],
            negative="--linear",
            help="Compute the gradient on log10 intensity or linear intensity.",
        ),
    ] = True,
) -> None:
    """Create an adaptive GGM FITS image from counts and exposure maps."""
    _, _, sigma_map = adaptive_ggm(
        counts_image=counts_image,
        exposure_image=exposure_image,
        background_image=background_image,
        target_counts=target_counts,
        target_snr=target_snr,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_step=sigma_step,
        sigma_to_radius_factor=sigma_to_radius_factor,
        use_log=log_scale,
        min_exposure_fraction=min_exposure_fraction,
        save_path=output_image,
    )
    valid_sigma = sigma_map[sigma_map == sigma_map]
    if valid_sigma.size == 0:
        raise ValueError(
            "No valid adaptive scales were selected; check the exposure mask."
        )
    print("sigma_min=", float(valid_sigma.min()))
    print("sigma_max=", float(valid_sigma.max()))


if __name__ == "__main__":
    app()
