import numpy as np
from astropy import units, coordinates
from tqdm import tqdm
import matplotlib.patheffects as patheffects
import matplotlib.patches as patches

def generate_distributed_sample(all_stars, n_stars, outer_radii=[5.,8.,12.], percents=[0.5,0.25,0.25], verbose=False):
    """Subsamples the input to have certain percentages in specified radial bins"""
    #Written by Gail Zasowski
    # outer_radii = array of values for the outer edges of the radial bins (assumes the innermost one starts at R=0 kpc)
    # percents = array of values (same length as outer_radii) for the percentage of stars to land in each radial bin
    # TO DO: add a keyword to force strict distributions (which may reduce the total sample to much less than n_stars) 
    #   or not (which may not return the exact distribution or exact n_stars)
    if np.sum(percents) != 1.0:
        print('Warning: percents do not add to 1.0')
    
    # Generate coordinates for the given distances, we don't know if they're noisy or not
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
    gal_r = np.sqrt(((gal_x) ** 2.0) + ((gal_y) ** 2.0))

    return_sample = []

    n_radii = len(outer_radii)
    
    for i_radius, this_radius in enumerate(outer_radii):
        if i_radius==0:
            inner_radius = 0.
        else:
            inner_radius = outer_radii[i_radius-1]
        samp = np.where(
            (gal_r >= inner_radius) & (gal_r < this_radius)
        )[0]
        n_needed = int(n_stars * percents[i_radius])
        if verbose:
            print('Bin {}: {} avail, {} needed'.format(i_radius+1, len(samp), int(n_needed)))
        if len(samp) >= n_needed:
            random_select = np.random.choice(
                samp, n_needed, replace=False
            )
            return_sample += [ix for ix in random_select]
        else:
            return_sample += [ix for ix in samp]

    return_dict = {
        "gal_x": np.array(gal_x)[return_sample],
        "gal_y": np.array(gal_y)[return_sample],
        "gal_z": np.array(gal_z)[return_sample],
        "gal_r": np.array(gal_r)[return_sample],
        "Dist": np.array(all_stars['Dist'])[return_sample],
        "Fe/H_evolved": np.array(all_stars['Fe/H_evolved'])[return_sample],
        "l": np.array(all_stars['l'])[return_sample],
        "b": np.array(all_stars['b'])[return_sample],
        "feh": np.array(all_stars["Fe/H_evolved"])[return_sample],
    }

    return return_dict

def generate_noisy_subsample(all_stars, n_stars, dist_uncertainty, feh_uncertainty):
    """Generate a random subsample with n_stars, and added noise on distance and abundance"""
    #Based on script by Julie Imig, modified by Gail Zasowski
    
    n_input_stars = len(all_stars['Dist'])
    
    uncertain_dists = [
        abs(np.random.normal(d, d * dist_uncertainty)) for d in all_stars["Dist"]
    ]
    uncertain_fehs = [
        np.random.normal(f, feh_uncertainty) for f in all_stars["Fe/H_evolved"]
    ]

    # Generate updated coordinates for the uncertain distances
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
    
    # Down-select if possible, return all stars if not
    return_sample = []
    samp = np.where((gal_x > -10))[0] #This is basically meaningless
    if len(samp) > n_stars:
        random_select = np.random.choice(samp, int(n_stars), replace=False)
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
    	
def generate_subsample(all_stars, n_stars):
    """Generate a random subsample with n_stars"""
    #Based on script by Julie Imig, modified by Gail Zasowski
    
    n_input_stars = len(all_stars['Dist'])
    
    # Generate coordinates for the given distances
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
    gal_r = np.sqrt(((gal_x) ** 2.0) + ((gal_y) ** 2.0))
    
    # Down-select if possible, return all stars if not
    return_sample = []
    samp = np.where((gal_x > -10))[0] #This is basically meaningless
    if len(samp) > n_stars:
        random_select = np.random.choice(samp, int(n_stars), replace=False)
        return_sample += [ix for ix in random_select]
    else:
        return_sample += [ix for ix in samp]
	
    return_dict = {
        "gal_x": np.array(gal_x)[return_sample],
        "gal_y": np.array(gal_y)[return_sample],
        "gal_z": np.array(gal_z)[return_sample],
        "gal_r": np.array(gal_r)[return_sample],
        "feh": np.array(all_stars['Fe/H_evolved'])[return_sample],
    }

    return return_dict

def generate_random_sample(all_stars, n_stars, dist_uncertainty, feh_uncertainty):
    """Generate a random sample with n_stars"""
    #Written by Julie Imig
    r_bins = 10  # 2 kpc sized bins = from the SRD
    r_samp = np.linspace(0, 20, r_bins + 1)

    uncertain_dists = [
        abs(np.random.normal(d, d * dist_uncertainty)) for d in all_stars["Dist"]
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
            #& (np.abs(gal_z) <= 0.1)
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

def run_gradient_sim(all_stars, test_samples, dist_uncertainty, feh_uncertainty, add_noise=False):
	#Written by Julie Imig, modified by Gail Zasowski
	slope = []
	slope_err = []
	median_slopes = []
	median_errs = []
	
	#for n in tqdm([test_samples]):
	for n in np.array([test_samples]):
		#samp = generate_random_sample(all_stars, n, dist_uncertainty, feh_uncertainty)
		if add_noise:
			samp = generate_noisy_subsample(all_stars, n, dist_uncertainty, feh_uncertainty)
		else:
			samp = generate_subsample(all_stars, n)

		m, cov = np.polyfit(samp["gal_r"], samp["feh"], 1, cov=True)
		slope.append(m[0])
		slope_err.append(np.sqrt(cov[0][0]))
		
		running_median = []
		running_scatter = []
		for r_bin in np.arange(0, 15, 1):
			r_mask = np.where(
				(samp["gal_r"] >= r_bin)
				& (samp["gal_r"] <= r_bin + 0.5)
				#& (np.abs(samp["gal_z"]) <= 0.1)  # focus on midplane for now
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