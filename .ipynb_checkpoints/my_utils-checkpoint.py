import numpy as np
from astropy import units, coordinates
from tqdm import tqdm
import matplotlib.patheffects as patheffects
import matplotlib.patches as patches

def generate_random_sample(all_stars, n_stars, dist_uncertainty, feh_uncertainty):
    """Generate a random sample with n_stars"""
    #Written by Julie Imig
    r_bins = 10  # 2 kpc sized bings = from the SRD
    r_samp = np.linspace(0, 20, r_bins + 1)

    uncertain_dists = [
        np.random.normal(d, d * dist_uncertainty) for d in all_stars["Dist"]
    ]
    uncertain_fehs = [
        np.random.normal(f, feh_uncertainty) for f in all_stars["Fe/H_evolved"]
    ]
    # uncertain_dists = [d for d in all_stars["Dist"]]
    # uncertain_fehs = [d for d in all_stars["Fe/H_initial"]]

    # Generate coordinates for the uncertain distances
    all_coords = coordinates.SkyCoord(
        frame="galactic",
        l=all_stars["l"],
        b=all_stars["b"],
        distance=uncertain_dists,
        unit=(units.deg, units.deg, units.kpc),
    )
    all_coords = all_coords.transform_to(coordinates.Galactocentric)
    gal_x = all_coords.x / units.kpc
    gal_y = all_coords.y / units.kpc
    gal_z = all_coords.z / units.kpc
    gal_r = np.sqrt(((gal_x) ** 2.0) + ((gal_y) ** 2.0))

    return_sample = []
    for i, r in enumerate(r_samp[:-1]):
        samp = np.where(
            (gal_r >= r)
            & (gal_r <= r_samp[i + 1])
            & (np.abs(gal_z) <= 0.1)
            & (gal_x > 0)
        )[0]
        if len(samp) >= int(n_stars / r_bins):
            random_select = np.random.choice(
                samp, int(n_stars / r_bins), replace=False
            )
            return_sample += [ix for ix in random_select]
        else:
            return_sample += [ix for ix in samp]

    return_dict = {
        "gal_x": np.array(gal_x)[return_sample],
        "gal_y": np.array(gal_y)[return_sample],
        "gal_z": np.array(gal_z)[return_sample],
        "gal_r": np.array(gal_r)[return_sample],
        "feh": np.array(uncertain_fehs)[return_sample],
    }

    return return_dict

def run_gradient_sim(all_stars, test_samples, dist_uncertainty, feh_uncertainty):
    #Written by Julie Imig
    slope = []
    slope_err = []
    median_slopes = []
    median_errs = []

    for n in tqdm(test_samples):
        samp = generate_random_sample(all_stars, n, dist_uncertainty, feh_uncertainty)
        m, cov = np.polyfit(samp["gal_r"], samp["feh"], 1, cov=True)
        slope.append(m[0])
        slope_err.append(np.sqrt(cov[0][0]))

        running_median = []
        running_scatter = []
        for r_bin in np.arange(0, 15, 1):
            r_mask = np.where(
                (samp["gal_r"] >= r_bin)
                & (samp["gal_r"] <= r_bin + 0.5)
                & (np.abs(samp["gal_z"]) <= 0.1)  # focus on midplane for now
                & (np.abs(samp["gal_x"]) > 0)  # far side of disk only
            )[0]
            median_feh = np.nanmedian(samp["feh"][r_mask])
            running_median.append(median_feh)
            running_scatter.append(np.nanstd(samp["feh"][r_mask]))

        running_median = np.array(running_median)
        med_slope, cov = np.polyfit(
            np.array(np.arange(0, 15, 1))[np.isfinite(running_median)],
            np.array(running_median)[np.isfinite(running_median)],
            1,
        )
        med_scatter = np.nanmedian(running_scatter)
        median_slopes.append(med_slope)
        median_errs.append(med_scatter)

    return slope, slope_err, median_slopes, median_errs

def plot_sun_and_gc(ax, size=1, r_sun=-8.122, add_bar=False, labels=True):
    """Marks the sun and the GC on a plot"""
    # Plot GC
    for i, mc in enumerate(["w", "k"]):
        ax.scatter(
            [0],
            [0],
            marker="+",
            c=mc,
            zorder=12 + i,
            s=(400 - (i * 100)) * size,
            lw=(5 - i * 2) * size,
            snap=False,
        )
    # Plot Sun
    ax.scatter(
        [r_sun],
        [0],
        marker="o",
        c="k",
        facecolor="k",
        s=250 * size,
        zorder=12,
        label="Sun",
        lw=2 * size,
        path_effects=[patheffects.withStroke(linewidth=5, foreground="w")],
    )
    # dot in center for sun symbol
    ax.scatter(
        [r_sun],
        [0],
        marker=".",
        c="w",
        s=50 * size,
        zorder=13,
        snap=False,
    )
    if labels:
        ax.text(
            r_sun,
            1 * size,
            "Sun",
            horizontalalignment="center",
            verticalalignment="bottom",
            zorder=13,
            color="k",
            fontsize=24 * size,
            weight="bold",
            path_effects=[
                patheffects.withStroke(linewidth=5 * size, foreground="w")
            ],
        )
        ax.text(
            0,
            1 * size,
            "Galactic\nCenter",
            horizontalalignment="center",
            verticalalignment="bottom",
            zorder=13,
            color="k",
            fontsize=24 * size,
            weight="bold",
            path_effects=[
                patheffects.withStroke(linewidth=5 * size, foreground="w")
            ],
        )
    if add_bar:  # ellipse for bar location
        bar_angle = 25 if r_sun > 0 else -25
        gbar = patches.Ellipse(
            (0, 0),
            10.0,
            0.4 * 10.0,
            angle=bar_angle,
            facecolor="None",
            edgecolor="k",
        )
        ax.add_patch(gbar)

    return