import numpy as np
import os

from astropy import coordinates, units
from astropy.table import Table
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import lombscargle

from my_utils import generate_distributed_sample, run_gradient_sim


def load_synthetic_data(directory="hge_gums_marshall"):
    # Load in the data
    all_csv_files = [
        f.name
        for f in os.scandir(f"{directory}")
        if f.name.endswith(".csv") and f.name.startswith("hge_gums_marshall_")
    ]

    # Concatenate all files
    all_stars = [pd.read_csv(f"{directory}/{f}") for f in all_csv_files]
    all_stars = pd.concat(all_stars)
    print(len(all_stars['l']))
    print(all_stars.columns)

    # Convert (l,b,dist) to Galactic (R,Z,phi)
    all_coords = coordinates.SkyCoord(
        frame="galactic",
        l=all_stars["l"],
        b=all_stars["b"],
        distance=all_stars["Dist"],
        unit=(units.deg, units.deg, units.kpc),
    )

    all_coords = all_coords.transform_to(coordinates.Galactocentric)
    gal_x = all_coords.x / units.kpc
    gal_y = all_coords.y / units.kpc
    gal_z = all_coords.z / units.kpc
    gal_r = np.sqrt(((all_coords.x) ** 2.0) + ((all_coords.y) ** 2.0)) / units.kpc

    return all_stars, gal_r, gal_z, gal_x, gal_y


def downsample_data(data, distance_min=6.0, hmag_max=16, metallicity_threshold=None,
                     use_APO=True, use_LCO=True, exclude_midplane=False, midplane_b_limit=2.0):
    """
    Build a boolean selection mask over `data`: distance/magnitude visibility
    limits, optional metallicity floor, optional midplane exclusion at inner
    longitudes, and site visibility. Distance/abundance uncertainty and
    radial-distribution shaping are sample_data's job, not this function's.
    """
    if not use_LCO:
        raise NotImplementedError("LCO visibility cut is not yet defined")

    masks = [data['Dist'] > distance_min, data['2MASS_H'] < hmag_max]

    if metallicity_threshold is not None:
        masks.append(data['Fe/H_evolved'] > metallicity_threshold)

    if exclude_midplane:
        masks.append(np.logical_or.reduce([
            abs(data['b']) > midplane_b_limit,
            np.logical_and.reduce([data['l'] > 10., data['l'] < 350.]),
        ]))

    if not use_APO:
        masks.append(np.logical_or.reduce([data['l'] < 10., data['l'] > 180.]))

    return np.logical_and.reduce(masks)


def sample_data(all_stars, dist_unc, abund_unc, rad_distrib,
                 distance_min=6.0, hmag_max=16, outer_rad_array=[4.0, 8.0, 15.0], **_ignored): # outer_z_array, phi_bin_edges
    """
    Apply selection cuts, perturb distances/abundances, then return the
    largest sample  across outer_rad_array bins.
    """
    gd = downsample_data(all_stars, distance_min=distance_min, hmag_max=hmag_max)

    dat = Table()
    dat['l'] = all_stars['l'][gd]
    dat['b'] = all_stars['b'][gd]
    dat['Dist'] = all_stars['Dist'][gd] + np.random.normal(0, dist_unc * all_stars['Dist'][gd])
    dat['Dist'][dat['Dist'] < distance_min] = np.nan
    dat['Fe/H_evolved'] = all_stars['Fe/H_evolved'][gd] + np.random.normal(0, abund_unc, size=gd.sum())

    coords = coordinates.SkyCoord(frame="galactic", l=dat['l'], b=dat['b'], distance=dat['Dist'],
                                   unit=(units.deg, units.deg, units.kpc)).transform_to(coordinates.Galactocentric)
    gal_r = np.sqrt(coords.x**2 + coords.y**2) / units.kpc

    percents = np.array(rad_distrib) / 100.
    inner_edges = [0.0] + outer_rad_array[:-1]
    available = np.array([np.sum((gal_r >= lo) & (gal_r < hi)) for lo, hi in zip(inner_edges, outer_rad_array)])

    # Largest total N that keeps every bin's share within what's available
    n_max = int(np.floor(min(a / p for a, p in zip(available, percents) if p > 0)))

    return generate_distributed_sample(dat, n_max, outer_radii=outer_rad_array, percents=percents)


def run_scenario(data, n_stars_array, **params):
    """Sweep n_stars_array, drawing a fresh random subsample of each size from data, and fit the gradient."""
    label = params.get("label", str(params))
    n_available = len(data['l'])

    scenario_results = []
    for this_n_stars in n_stars_array:
        n_draw = min(int(this_n_stars), n_available)
        idx = np.random.choice(n_available, n_draw, replace=False)
        sample = {key: np.asarray(val)[idx] for key, val in data.items()}

        slope, slope_err, median_slopes, median_errs = run_gradient_sim(
            sample, n_draw, dist_uncertainty=params.get("dist_unc"),
            feh_uncertainty=params.get("abund_unc"), add_noise=False,
        )

        scenario_results.append({
            "label": label,
            **{k: (tuple(v) if k == "rad_distrib" else v) for k, v in params.items() if k != "label"},
            "n_stars_requested": this_n_stars,
            "n_stars_actual": n_draw,
            "med_slope": median_slopes[0],
            "slope_err": slope_err[0],
        })

    return scenario_results

def flatten_gradient(data, result, **_ignored):
    med_slope = result['med_slope']
    metallicity_detrend = data['Fe/H_evolved'] - med_slope * data['R']
    return metallicity_detrend

def wrap_azimuth(azimuth): # if needed
    """Return a boolean mask for azimuth values that are continuous at azimuth=0."""
    azimuth_wrapped = np.copy(azimuth)
    azimuth_wrapped[azimuth > np.pi] -= 2 * np.pi 
    return azimuth_wrapped

def apply_fft(azimuth_wrapped , metallicity_detrend **_ignored):
    # Apply FFT to the detrended metallicity data
    # requires np.fft.fftshift and np.fft.fftfreq to get the correct frequency bins
    n_fft = len(metallicity_detrend)
    fft_result = np.fft.fftshift(np.fft.fft(metallicity_detrend, n = n_fft))
    fft_frequencies = np.fft.fftfreq(n_fft)
    return fft_result, fft_frequencies

def apply_lombscargle(azimuth_wrapped, metallicity_detrend, **_ignored):
    frequencies = np.linspace(0.01, 10, 1000) 
    
    # Apply Lomb-Scargle periodogram to the detrended metallicity data
    pdgram = lombscargle(azimuth_wrapped, metallicity_detrend, frequencies)

    return pdgram, frequencies

def apply_custom_metallicity_profile():
    pass