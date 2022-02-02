#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import copy
import time
try:
    import _Pickle as pickle
except:
    import pickle

import numpy
import matplotlib
from matplotlib import pyplot
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats

matplotlib.rcParams['text.usetex'] = False
major, minor = matplotlib.__version__.split(".")[0:2]
if (int(major) <= 2) and (int(minor) < 2):
    matplotlib.rcParams['text.latex.unicode'] = True

if len(sys.argv) > 1:
    DIR_ADDITION = sys.argv[1]
else:
    DIR_ADDITION = "initialvar-00132" #None

t0 = time.time()
def status_message(msg):
    secs = int(round(time.time() - t0, 0))
    hh = secs // 3600
    mm = (secs - hh*3600) // 60
    ss = (secs - hh*3600) - mm*60
    timestamp = "{}:{}:{}".format(hh, str(mm).rjust(2, "0"), \
        str(ss).rjust(2, "0"))
    print("{} - {}".format(timestamp, msg))


# # # # #
# CONSTANTS

# ANALYSIS SPECIFICATIONS
# Overwrite the temporary data file?
OVERWRITE_TMP = False
# Set the number of runs to count for every simulation.
NRUNS = 10

# Set the plot frequency for years, using a log space.
PLOT_YEARS = numpy.round(numpy.logspace(2, 5, num=25, base=10), \
    decimals=0).astype(int)
PLOT_YEARS = numpy.round(PLOT_YEARS/100).astype(int)*100
# Set the six years to plot as a bar plot.
#BARPLOT_YEARS = numpy.round(numpy.logspace(2, 5, num=6, base=10), \
#    decimals=0).astype(int)
#BARPLOT_YEARS = numpy.round(BARPLOT_YEARS/100).astype(int)*100
# Log-space approach like the above, but rounded to nearest 1000:
#BARPLOT_YEARS = [100, 300, 2000, 6000, 25000, 100000]
# Alternative linear approach:
BARPLOT_YEARS = [100, 20000, 40000, 60000, 80000, 100000]

# Choose what to plot in the example panels: "hist" for the logged histrogram,
# or "norm" for a normal distribution with logged mean and standard deviation.
# Upside of the histogram is that it feels more empirical, downside is that it
# is not of the highest resolution (i.e. not too many bins).
EXAMPLE_PLOT_CONTENT = "hist"


# SIMULATION SPECIFICS
# Number of years that the simulation ran for (if it did not go extinct).
MAX_YEARS = 100000
# Frequency of recording in simulation years.
LOG_FREQ = 100
# Compute the number of logs to expect, to limit the pre-allocated size of 
# vectors for each run.
N_LOGS = MAX_YEARS // LOG_FREQ
# Nutrition level that the reported proportions were from (in kJ/day).
ENV_NUTRITION_NEUTRAL = 11083
# Level of contamination danger (probability of death per kJ).
ENV_DANGER = { \
    "none":     0, \
    "low":      2.0883e-9, \
    "mid":      5.1114e-9, \
    "high":     8.2622e-9, \
    "higher":   1.1551e-8, \
    "highest":  1.4993e-8, \
    }
ENV_DANGER_ORDER = ["none", "low", "mid", "high", "higher", "highest"]
# Starting values of each trait.
STARTING_TRAIT_VALUE = { \
    "m_culture": 0.95, \
    "m_pheno":   0.40, \
    }

# VARIABLES
# Dict of variables that are stored across several columns in the data files.
MULTIVARS = {}
# Add age histogram to the multivars.
MULTIVARS["age_hist"] = []
age_bins = list(range(0,81,5)) + [100]
for i in range(0, len(age_bins)-1):
    MULTIVARS["age_hist"].append("p_age_{}-{}".format( \
        age_bins[i], age_bins[i+1]))
# Add culture to the multivars.
MULTIVARS["culture_hist"] = []
culture_bins = numpy.linspace(0.0, 1.0, num=31)
for i in range(0, culture_bins.shape[0]-1):
    MULTIVARS["culture_hist"].append("pdf_culture_{}-{}".format( \
        int(round(1000*culture_bins[i],0)), \
        int(round(1000*culture_bins[i+1],0))))
# Add the restricted-range culture to the multivars.
MULTIVARS["culture_hist_restricted"] = []
culture_bins_restricted = numpy.linspace(0.7, 0.8, num=31)
for i in range(0, culture_bins_restricted.shape[0]-1):
    MULTIVARS["culture_hist_restricted"].append("pdf_culture_{}-{}".format( \
        int(round(1000*culture_bins[i],0)), \
        int(round(1000*culture_bins[i+1],0))))
# Add phenotype to the multivars.
MULTIVARS["phenotype_hist"] = []
pheno_bins = numpy.linspace(0.0, 1.0, num=31)
for i in range(0, pheno_bins.shape[0]-1):
    MULTIVARS["phenotype_hist"].append("pdf_pheno_{}-{}".format( \
        int(round(1000*pheno_bins[i],0)), \
        int(round(1000*pheno_bins[i+1],0))))
# Add genotype to the multivars.
MULTIVARS["genotype_hist"] = []
geno_bins = numpy.linspace(0.0, 1.0, num=102)
allele_varieties = numpy.round(geno_bins[:-1]+numpy.diff(geno_bins)/2.0, 2)
for i in range(0, allele_varieties.shape[0]):
    MULTIVARS["genotype_hist"].append("p_geno_{}".format( \
        int(round(100*allele_varieties[i],0))))

# Add sex-differentiated multivars.
for sex in ["male", "female"]:
    for var in ["culture_hist", "phenotype_hist"]:
        MULTIVARS["{}_{}".format(sex,var)] = copy.deepcopy(MULTIVARS[var])
        for i, varlbl in enumerate(MULTIVARS["{}_{}".format(sex,var)]):
            MULTIVARS["{}_{}".format(sex,var)][i] = "{}_{}".format(sex, varlbl)

# Create positions and tick labels for multivariable histograms.
MULTIVAR_BINS = {}
for multivar in MULTIVARS.keys():
    if multivar == "age_hist":
        x = age_bins
        xticks = [var.replace("p_age_","") for var in MULTIVARS[multivar]]
        xlabel = "Age (years)"
        ylabel = "Probability density"
    elif "culture_hist" in multivar:
        x = culture_bins
        xticks = [var.replace("pdf_culture_","") for var in MULTIVARS[multivar]]
        xlabel = "Cultural trait (AU)"
        ylabel = "Probability density"
    elif multivar == "culture_hist_restricted":
        x = culture_bins_restricted
        xticks = [var.replace("pdf_culture_","") for var in MULTIVARS[multivar]]
        xlabel = "Cultural trait (AU)"
        ylabel = "Probability density"
    elif "phenotype_hist" in multivar:
        x = pheno_bins
        xticks = [var.replace("pdf_pheno_","") for var in MULTIVARS[multivar]]
        xlabel = "Phenotype (AU)"
        ylabel = "Probability density"
    elif multivar == "genotype_hist":
        x = allele_varieties
        xticks = [var.replace("p_geno_","") for var in MULTIVARS[multivar]]
        xlabel = "Allele"
        ylabel = "Proportion in population"
    if multivar not in ["genotype_hist"]:
        x = x[:-1] + numpy.diff(x)/2.0
    MULTIVAR_BINS[multivar] = {"x":x, "xticks":xticks, "xlabel":xlabel, \
        "ylabel":ylabel}

# COLOUR
# Equalise the vlim in the culture/genes grid plots?
EQUALISE_GRID_VLIM = True
# Set the vlim to a predefined value using a dict, or set the variable to None
# to automatically choose the values. Note: this does not overrule 
# EQUALISE_GRID_VLIM!
PRESET_GRID_VLIM = { \
    "m_culture":    0.4, \
    "m_pheno":      0.4, \
    }
# Set the colour map for indicating years.
YEAR_CMAP = "viridis"
# Set the colour map for indicating cost of contamination levels.
KAPPA_CMAP = "RdPu"
# Set the colour map for allele types (represented as continuous).
ALLELE_FREQ_CMAP = "Greens_r" #"YlGn_r" #"YlOrBr"
# From Wong, B. (2011). Points of view: Color blindness. Nature Methods, 8:441.
# Based on the RGB values in Wong's palette.
COLS = { \
    "orange": "#e69d00", \
    "lightblue": "#56b3e9", \
    "green": "#009e74", \
    "yellow": "#f0e442", \
    "blue": "#0072b2", \
    "vermillion": "#d55e00", \
    "pink": "#cc79a7", \
    }

# FILES AND FOLDERS
DIR = os.path.dirname(os.path.abspath(__file__))
if DIR_ADDITION is None:
    DATADIR = os.path.join(DIR, "data")
else:
    DATADIR = os.path.join(DIR, "data_{}".format(DIR_ADDITION))
if not os.path.isdir(DATADIR):
    raise Exception("Could not find data directory!")
if DIR_ADDITION is None:
    OUTDIR = os.path.join(DIR, "output")
else:
    OUTDIR = os.path.join(DIR, "output_{}".format(DIR_ADDITION))
TMPDIR = os.path.join(OUTDIR, "temp_data")
FIGDIR = os.path.join(OUTDIR, "figures")
CHECKDIR = os.path.join(OUTDIR, "check_figures")
STATSDIR = os.path.join(OUTDIR, "stats_outcomes")
for dirpath in [OUTDIR, TMPDIR, FIGDIR, CHECKDIR, STATSDIR]:
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)


# # # # #
# EXTRACT DATA

# Check if a temporary data file exists.
tmp_data_file = os.path.join(TMPDIR, "data.pickle")
if os.path.isfile(tmp_data_file) and not OVERWRITE_TMP:
    status_message("Loading data from temporary file.")
    with open(tmp_data_file,"rb") as f:
        data = pickle.load(f)

# Load the data from text files.
else:
    # Start with an empty dict.
    data = {}
    
    # Auto-detect all simulated conditions.
    conditions = os.listdir(DATADIR)
    env_conditions = []
    for con in conditions:
        for dirname in os.listdir(os.path.join(DATADIR, con)):
            if dirname not in env_conditions:
                env_conditions.append(dirname)
    
    # Loop through all meta-conditions.
    for ci, con in enumerate(conditions):
        
        status_message("Loading {} files from condition '{}'".format( \
            len(env_conditions), con))
    
        # Parse the condition name. The name will look like the following:
        # "disgust-{}_culture-{}_cultrestrict-{}_genes-{}_genetype-{}_mutation-{}"
        settings = {}
        for setting in ["disgust", "culture", "cultrestrict", "genes", \
            "genetype", "mutation"]:
            # Find the index from where this setting is embedded in the string.
            si = con.find("{}-".format(setting)) + len(setting) + 1
            # Find the setting in the condition name.
            if setting != "genetype":
                if "_" in con[si:]:
                    ei = si + con[si:].find("_")
                    settings[setting] = int(con[si:ei])
                else:
                    settings[setting] = int(con[si:])
            # I should probably not have used underscores in the genetypes; makes
            # it significantly harder to parse by splitting by underscores.
            else:
                si = con.find("{}-".format(setting)) + len(setting)+2
                allowed = ["monogenic_highest", "monogenic_lowest", \
                    "monogenic_average", "polygenic"]
                for gene_type in allowed:
                    if gene_type in con[si:]:
                        break
                settings[setting] = copy.deepcopy(gene_type)
        
        # Start with an empty data dict.
        data[con] = {}
        
        # Loop through all environmental conditions.
        for ei, econ in enumerate(env_conditions):
            
            status_message("\tLoading environment {}/{}".format(ei+1, 
                len(env_conditions)))
    
            # Start with an empty data dict.
            data[con][econ] = {}
        
            # Parse the condition name. The name will look like the following:
            # "environment_danger-mid_environment_nutrition-100_environment_contamination-5"
            # This will overwrite only the environment-specific keys in the current
            # settings dict.
            for setting in ["environment_danger", "environment_nutrition", \
                "environment_contamination"]:
                # Find the index from where this setting is embedded in the string.
                si = econ.find("{}-".format(setting)) + len(setting) + 1
                # Find the setting in the condition name.
                if "_" in econ[si:]:
                    ei = si + econ[si:].find("_")
                    settings[setting] = econ[si:ei]
                else:
                    settings[setting] = econ[si:]
                # Convert numerical settings from string to int.
                if setting in ["environment_nutrition", \
                    "environment_contamination"]:
                    settings[setting] = int(settings[setting])
            
            # Find all files in this condition.
            dpath = os.path.join(DATADIR, con, econ)
            if not os.path.isdir(dpath):
                continue
            all_fnames = os.listdir(dpath)
            all_fnames.sort()
            
            # Ignore all file names that shouldn't be in here.
            rem_files = []
            for fname in all_fnames:
                remove = False
                # Remove all stupid OS X files. (Thanks, Apple.)
                if fname[:2] == "._":
                    remove = True
                # Remove temporary lock files.
                if fname[:6] == ".~lock":
                    remove = True
                # Remove any non-CSV file.
                name, ext = os.path.splitext(fname)
                if ext != ".csv":
                    remove = True
                # Remove files that don't abide by the naming scheme.
                if name[:3] != "run":
                    remove = True
                # Remove the file.
                if remove:
                    rem_files.append(fname)
            for fname in rem_files:
                all_fnames.remove(fname)
        
            # Throw an error if there are too few files.
            if len(all_fnames) < NRUNS:
                raise Exception("Found only {} files in ".format(len(all_fnames)) \
                    + "condition {}-{}, ".format(con, econ) \
                    + "but {} runs were required (see NRUNS constant).".format(NRUNS))
        
            # Loop through all files.
            for fi, fname in enumerate(all_fnames[:NRUNS]):
        
                # Load the raw data.
                fpath = os.path.join(DATADIR, con, econ, fname)
                raw = numpy.loadtxt(fpath, delimiter=",", unpack=True, dtype=str)
                # Count the number of variables/columns (k) and the number of
                # observations/years/rows (n).
                k, n = raw.shape
                n -= 1
                
                # Reformat the data as a dict.
                d = {}
                for i in range(raw.shape[0]):
                    var = raw[i,0]
                    val = raw[i,1:]
                    d[var] = val.astype(numpy.float32)
                
                # Create a new entry in the main data dict if one does not exist yet.
                if not data[con][econ]:
                    # Add all single-column variables.
                    for var in d.keys():
                        data[con][econ][var] = numpy.zeros((N_LOGS,NRUNS), \
                            dtype=numpy.float32) * numpy.NaN
                    # Add all variables that span across several columns.
                    for var in MULTIVARS.keys():
                        shape = (N_LOGS, len(MULTIVARS[var]), NRUNS)
                        data[con][econ][var] = numpy.zeros(shape, \
                            dtype=numpy.float32) * numpy.NaN
                
                # Save the data in the main dict.
                for var in d.keys():
                    # Avoid assuming the same number of years in each run, as this
                    # can be different between extinct populations.
                    if data[con][econ][var].shape[0] < d[var].shape[0]:
                        ei = data[con][econ][var].shape[0]
                        data[con][econ][var][:,fi] = d[var][:ei]
                    elif data[con][econ][var].shape[0] > d[var].shape[0]:
                        ei = d[var].shape[0]
                        data[con][econ][var][:ei,fi] = d[var]
                    else:
                        data[con][econ][var][:,fi] = d[var]
                
                # Compile multivar data.
                for multivar in MULTIVARS.keys():
                    # Check if we should remove and skip this multivar.
                    skip = False
                    if (not settings["culture"]) and (multivar in ["culture_hist", \
                        "culture_hist_restricted"]):
                        skip = True
                    if settings["cultrestrict"] and (multivar == "culture_hist"):
                        skip = True
                    if (not settings["cultrestrict"]) and \
                        (multivar == "culture_hist_restricted"):
                        skip = True
                    if (not settings["genes"]) and (multivar in ["genotype_hist", \
                        "phenotype_hist"]):
                        skip = True
                    if (not settings["mutation"]) and (multivar == "genotype_hist"):
                        skip = True
                    if skip:
                        if multivar in data[con][econ].keys():
                            del(data[con][econ][multivar])
                        continue
                    # Collect all variables.
                    for vi, var in enumerate(MULTIVARS[multivar]):
                        # Avoid assuming the same number of years in each run, as this
                        # can be different between extinct populations.
                        if data[con][econ][multivar].shape[0] < d[var].shape[0]:
                            ei = data[con][econ][multivar].shape[0]
                            data[con][econ][multivar][:,vi,fi] = d[var][:ei]
                        elif data[con][econ][multivar].shape[0] > d[var].shape[0]:
                            ei = d[var].shape[0]
                            data[con][econ][multivar][:ei,vi,fi] = d[var]
                        else:
                            data[con][econ][multivar][:,vi,fi] = d[var]
            
            # Add the settings.
            data[con][econ]["settings"] = copy.deepcopy(settings)
    
    # Save the data dictionary.
    status_message("Writing loaded data to temporary file.")
    with open(tmp_data_file,"wb") as f:
        p = pickle.Pickler(f)
        p.fast = True
        p.dump(data)


# # # # #
# DESCRIPTIVES

# Open a text file to write descriptive statistics to.
with open(os.path.join(STATSDIR, "descriptives.csv"), "w") as f:
    # Write a header to the file.
    header = ["condition", "environment", "p_male", "p_fertile", "p_pair", \
        "birth_rate", "m_age_over_5", "sd_age_over_5", "m_children_over_40", \
        "sd_children_over_40", "m_energy", "sd_energy", "min_energy", \
        "max_energy"]
    f.write(",".join(header))
    # Loop through conditions.
    all_conditions = list(data.keys())
    all_conditions.sort()
    for con in all_conditions:
        env_conditions = list(data[con].keys())
        env_conditions.sort()
        for econ in env_conditions:
            # Skip if empty.
            if not data[con][econ]:
                continue
            # Construct a line with averages across all the things we need.
            line = [con, econ]
            # Compute average proportions, and standard deviations across runs.
            for var in header[2:]:
                line.append(numpy.mean(data[con][econ][var], axis=None))
            # Write to file.
            f.write("\n" + ",".join(map(str, line)))

# Write M and SD for pheno and restricted culture at the first recorded year.
#with open(os.path.join(STATSDIR, "year-0_restricted_culture.csv"), "w") as f:
#    # Write a header to the file.
#    header = ["condition", "m_pheno", "sd_pheno", "m_culture", "sd_culture"]
#    f.write(",".join(header))
#    # Loop through conditions.
#    all_conditions = [ \
#        "disgust-0_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-0", \
#        "disgust-0_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-1", \
#        "disgust-1_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-0", \
#        "disgust-1_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-1", \
#        ]
#    all_conditions.sort()
#    for con in all_conditions:
#        env_conditions = list(data[con].keys())
#        env_conditions.sort()
#        for econ in env_conditions:
#            # Skip if empty.
#            if not data[con][econ]:
#                continue
#            # Construct a line with averages across all the things we need.
#            line = [con, econ]
#            # Compute average proportions, and standard deviations across runs.
#            for var in header[1:]:
#                line.append(numpy.mean(data[con][econ][var][0,:], axis=None))
#            # Write to file.
#            f.write("\n" + ",".join(map(str, line)))
#

# # # # #
# FIGURE 1

# FIGURE 1: Four-panel figure showing the main findings.
fig, axes = pyplot.subplots(nrows=2, ncols=2, figsize=(18,12), dpi=600)
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, \
    hspace=0.3, wspace=0.3)

# Define a few things to do with each trait component.
traits = ["genetic", "culture"]
var = { \
    "culture": { \
        "m": "m_culture", \
        "sd": "sd_culture", \
        "hist": "culture_hist", \
        "start": STARTING_TRAIT_VALUE["m_culture"], \
        "xlim": [0.6, 1.0]
        }, \
    "genetic": { \
        "m": "m_pheno", \
        "sd": "sd_pheno", \
        "hist": "phenotype_hist", \
        "start": STARTING_TRAIT_VALUE["m_pheno"], \
        "xlim": [0.1, 0.5]
        }, \
    }

# Find a dataset that does not have any NaNs across the years.
for con in data.keys():
    for econ in data[con].keys():
        for i in range(data[con][econ]["year"].shape[1]):
            if numpy.sum(numpy.isnan(data[con][econ]["year"][:,i])) == 0:
                example_years = data[con][econ]["year"][:,i]
                break
# Create a normed colour map to indicate years. Usage: cmap(norm(year))
cmap = matplotlib.cm.get_cmap(YEAR_CMAP)
min_year = example_years[0]
max_year = example_years[-1]
norm = matplotlib.colors.Normalize(vmin=0, vmax=max_year)
# Find the indices of all years to plot in histogram plots.
#year_indices = numpy.where(numpy.isin(example_years, PLOT_YEARS))[0]
year_indices = numpy.array(range(example_years.shape[0]), dtype=numpy.int32)

# PANEL A (top-left): Two sub-panels, each showing phenotype histograms.
# PANEL B (bottom-left): Two sub-panels, each showing phenotype histograms.
# Both panels consist of two sub-panels: left for genetic and right for 
# cultural drift. Panel A shows an environment with natural selection, and
# Panel B a control environment without.
con = "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1"
econs = [ \
    "environment_danger-higher_environment_nutrition-100_environment_contamination-10", \
    "environment_danger-none_environment_nutrition-100_environment_contamination-0", \
    ]
# Find the highest value, and consequently set the y limit.
max_ylim = 1.0
for econ in econs:
    for key in ["culture_hist", "phenotype_hist"]:
        mx = numpy.nanmax(numpy.nanmean(data[con][econ][key][:,:,:], axis=2))
        if mx > max_ylim:
            max_ylim = copy.copy(mx)
# Plot the pabels.
for ei, econ in enumerate(econs):
    # Split the axis into two.
    ax1 = axes[ei,0]
    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes("right", size="100%", pad=0.25)
    # Add a colour bar axis too.
    bax = divider.append_axes("right", size="10%", pad=0.05)
    # Loop through the sub-axes to plot histograms for polygenic traits.
    for ai, ax in enumerate([ax1,ax2]):
        # Set the sub-axis title, using an in-plot annotation.
        buff = 0.1 * (var[traits[ai]]["xlim"][1] - var[traits[ai]]["xlim"][0])
        x_ = var[traits[ai]]["xlim"][0] + buff
        y_ = max_ylim * 0.9
        ax.annotate(traits[ai].title(), (x_,y_), fontsize=16)
        # Draw a vertical line to indicate the strating position.
        ax.axvline(x=var[traits[ai]]["start"], lw=3, ls="--", \
            color="#000000", alpha=0.5)
        # Shade the area that indicates more disgust approach.
        ax.axvspan(var[traits[ai]]["start"], var[traits[ai]]["xlim"][1], \
            color="#000000", alpha=0.1)
        # Loop through all years that should be plotted.
        for i in year_indices:
            # Get the current year.
            year = data[con][econ]["year"][i,0]
            # Plot either the histogram or a normal distribution with measured
            # mean and standard deviation.
            if EXAMPLE_PLOT_CONTENT == "hist":
                # Compute the average and standard error across runs.
                y = data[con][econ][var[traits[ai]]["hist"]][i,:,:]
                m = numpy.mean(y, axis=1)
                sem = numpy.std(y, axis=1) / numpy.sqrt(y.shape[1])
                # Plot the average and shade the error.
                ax.plot(MULTIVAR_BINS[var[traits[ai]]["hist"]]["x"], m, "-", \
                    lw=2, color=cmap(norm(year)), alpha=1.0)
                #ax.fill_between(MULTIVAR_BINS[var[traits[ai]]["hist"]]["x"], \
                #    m-sem, m+sem, color=cmap(norm(year)), alpha=0.3)
            elif EXAMPLE_PLOT_CONTENT == "norm":
                # Create the normal distribution that matches the current year,
                # using the average and pooled SD over simulation runs.
                x = numpy.linspace(0.0, 1.0, 1000)
                y = scipy.stats.norm.pdf(x, \
                    loc=numpy.nanmean(data[con][econ][var[traits[ai]]["m"]][i,:]), \
                    scale=numpy.nanmean(data[con][econ][var[traits[ai]]["sd"]][i,:]))
                # Plot the distribution.
                ax.plot(x, y, "-", lw=2, color=cmap(norm(year)), alpha=1.0)
        # Set ticks and limits.
        xticks = numpy.round(numpy.linspace(var[traits[ai]]["xlim"][0], \
            var[traits[ai]]["xlim"][1], 5), decimals=2)
        xticklabels = list(map(str, xticks))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=14, rotation=80)
        ax.set_xlim(var[traits[ai]]["xlim"])
        ax.set_ylim(bottom=0)
    # Equalise y limits.
    ax1.set_ylim(top=max_ylim)
    ax2.set_ylim(top=max_ylim)
    # Remove ticks from the right plot's y axis.
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    # Set colour bar ticks.
    xticks = range(0, int(max_year+1), int(max_year//5))
    xticklabels = [str(int(x//1000))+"k" for x in xticks]
    xticklabels[0] = "0"
    # Add the colour bar.
    cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, norm=norm, \
        ticks=xticks, orientation='vertical')
    cbar.set_ticklabels(xticklabels)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Years since population onset", fontsize=18)
    # Set the title for the axis.
    if ei == 0:
        ax1.set_title(r"Trait drift under natural selection", \
            horizontalalignment='left', x=0.1, y=1.02, fontsize=20)
    elif ei == 1:
        ax1.set_title(r"Trait drift $without$ natural selection (control)", \
            horizontalalignment='left', x=0.1, y=1.02, fontsize=20)
    # Only set an x label on the bottom plot.
    if ei == 1:
        lbl = r"$\mathsf{\Leftarrow avoidance \quad}$ Disgust $\mathsf{\quad approach \Rightarrow}$"
        ax1.set_xlabel(lbl, fontsize=20, fontweight="bold", \
            horizontalalignment="center", x=1.0)
    # Set a y-label.
    ax1.set_ylabel("Probability density", fontsize=18)

# PANEL C
# This panel consists of two sub-panels, each displaying the drift in traits
# (y-axis) over time (x-axis) as a function of environmental danger (kappa, in
# separate lines). The top sub-panel is for the genetic component, the other
# panel is for culture.
con = "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1"
conecon = "environment_danger-none_environment_nutrition-100_environment_contamination-10"
econs = []
env_dangers = ["low", "mid", "high", "higher", "highest"]
kappa = numpy.zeros(len(env_dangers), dtype=numpy.float64)
for ei, env_danger in enumerate(env_dangers):
    econs.append( \
        "environment_danger-{}_environment_nutrition-100_environment_contamination-10".format( \
        env_danger))
    kappa[ei] = ENV_DANGER[env_danger]
# Get all the data together.
m = {}
sd = {}
shape = (len(econs), example_years.shape[0])
for trait in traits:
    c = numpy.nanmean(data[con][conecon][var[trait]["m"]], axis=1)
    m[trait] = numpy.zeros(shape, dtype=numpy.float64)
    sd[trait] = numpy.zeros(shape, dtype=numpy.float64)
    for ei, econ in enumerate(econs):
        m[trait][ei,:] = numpy.nanmean(data[con][econ][var[trait]["m"]], \
            axis=1) - c
        sd[trait][ei,:] = numpy.nanmean(data[con][econ][var[trait]["sd"]], \
            axis=1)
ylim = [min([numpy.min(m[trait]) for trait in traits]), \
    max([numpy.max(m[trait]) for trait in traits])]
buff = 0.2*(ylim[1]-ylim[0])
ylim[0] -= buff
ylim[1] += buff
# Split the axis into two.
ax1 = axes[0,1]
divider = make_axes_locatable(ax1)
ax2 = divider.append_axes("bottom", size="100%", pad=0.25)
ax = [ax1, ax2]
# Set the axis title.
ax1.set_title(r"Trait evolution compared to control", \
    horizontalalignment='left', x=0.05, y=1.02, fontsize=20)
# Create a normed colour map to indicate years. Usage: cmap(norm(year))
cmap = matplotlib.cm.get_cmap(KAPPA_CMAP)
kmax = numpy.max(kappa)
norm = matplotlib.colors.Normalize(vmin=kmax*-0.5, vmax=kmax)
# Set ticks and ticklabels.
yticks = [0, -0.1, -0.2]
xticks = range(0, int(max_year+1), int(max_year//5))
xticklabels = [str(int(x//1000))+"k" for x in xticks]
xticklabels[0] = "0"

# Plot all the lines.
for ai in range(len(traits)):
    # Set the sub-axis title, using an in-plot annotation.
    x_ = 0.85 * example_years[-1]
    y_ = ylim[0] + 0.85 * (ylim[1]-ylim[0])
    ax[ai].annotate(traits[ai].title(), (x_,y_), fontsize=16)
    # Plot a horizontal line to indicate 0.
    ax[ai].plot(example_years, numpy.zeros(example_years.shape), "--", \
        color="#000000", alpha=0.5, label=r"$\kappa$=0 (control)")
    # Plot all the lines (averages) and shadings (pooled SDs).
    for ei in range(m[trait].shape[0]):
        col = cmap(norm(kappa[ei]))
        ax[ai].plot(example_years, m[traits[ai]][ei,:], "-", lw=3, color=col, \
            alpha=0.5, label=r"$\kappa$={}".format(kappa[ei]))
        ax[ai].fill_between(example_years, \
            m[traits[ai]][ei,:] + sd[traits[ai]][ei,:], \
            m[traits[ai]][ei,:] - sd[traits[ai]][ei,:], \
            color=col, alpha=0.2)
    # Set the y limits.
    ax[ai].set_yticks(yticks)
    ax[ai].set_ylim(ylim)
    # Set the x limits and ticks.
    ax[ai].set_xlim([example_years[0], example_years[-1]])
    ax[ai].set_xticks(xticks)
    if ai < len(ax)-1:
        ax[ai].set_xticklabels([])
    else:
        ax[ai].set_xticklabels(xticklabels, fontsize=12)
# Add the legend to the lower-left of the upper subplot.
ax1.legend(loc="lower left", ncol=2, fontsize=14)
# Add axis labels.
ax2.set_ylabel( \
    r"$\mathsf{\Leftarrow avoidance \;}$ Disgust $\mathsf{\; approach \Rightarrow}$", \
    fontsize=18, fontweight="bold", horizontalalignment="center", y=1.0)
ax2.set_xlabel("Years since population onset", fontsize=18)


# PANEL D
# This panel will be a stacked plot for allele frequency in the genotype.
con = "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1"
econ = "environment_danger-higher_environment_nutrition-100_environment_contamination-10"
# Choose the axis for this plot.
ax = axes[1,1]
# Split the axis into two, to add a colour bar.
divider = make_axes_locatable(ax)
bax = divider.append_axes("right", size="5%", pad=0.05)

# Get the data together in a matrix with shape (m,n) where m is the number of
# bins in the allele histogram, and n is the number years for which we have a 
# recording.
n_year_logs, n_alleles, n_runs = data[con][econ]["genotype_hist"].shape
stack = numpy.zeros((n_alleles, n_year_logs), dtype=numpy.float64)
# Create a normed colour map to indicate years. Usage: cmap(norm(year))
cmap = matplotlib.cm.get_cmap(ALLELE_FREQ_CMAP)
norm = matplotlib.colors.Normalize(vmin=0.1, vmax=0.65)
cols = []
for val in allele_varieties:
    cols.append(cmap(norm(val)))
# Get the data for the stack.
for i in range(n_alleles):
    stack[i,:] = numpy.nanmean(data[con][econ]["genotype_hist"][:,i,:], axis=1)
cumsum = numpy.zeros((10, stack.shape[1]), dtype=numpy.float64)
for i in range(cumsum.shape[0]):
    cumsum[i,:] = numpy.nansum(stack[:(i+1)*10,:], axis=0)
# Plot the stack.
ax.stackplot(example_years, stack, colors=cols, alpha=1.0, antialiased=False)
# Plot the deciles within the stack.
for i in range(cumsum.shape[0]):
    # Create a gap in the line to annotate the percentile number.
    gap = (round(example_years.shape[0]*0.47), \
        round(example_years.shape[0]*0.52))
    # Plot the 0-40% of the line.
    ax.plot(example_years[:gap[0]], cumsum[i,:gap[0]], "--", lw=2, \
        color="#000000", alpha=0.5)
    # Plot 60-100% of the line.
    ax.plot(example_years[gap[1]:], cumsum[i,gap[1]:], "--", lw=2, \
        color="#000000", alpha=0.5)
    # Write the decile number, but only those <=60 (the rest is all too
    # cluttered together).
    if i < 6:
        ax.annotate(r"$\mathsf{" + "{}".format((i+1)*10)+r"^{th}}$", \
            (example_years[gap[0]], cumsum[i,gap[0]]-0.01), \
            color="#000000", alpha=0.5, fontsize=12)
# Add the colour bar.
cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, norm=norm, \
    ticks=[0, 1], orientation='vertical')
#cbar.set_ticklabels(["0", "1"])
cbar.ax.tick_params(labelsize=12)
cbar.set_label( \
    r"$\mathsf{\Leftarrow avoidance \;}$ Disgust $\mathsf{\; approach \Rightarrow}$", \
    fontsize=18, fontweight="bold")
# Set axis limits.
ax.set_ylim([0.0, 1.0])
ax.set_xlim([example_years[0], example_years[-1]])
# Set axis ticks and ticklabels.
xticks = range(0, int(max_year+1), int(max_year//5))
xticklabels = [str(int(x//1000))+"k" for x in xticks]
xticklabels[0] = "0"
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=12)
# Set axis labels.
ax.set_title("Genotype changes under natural selection", \
    horizontalalignment='left', x=0.05, y=1.02, fontsize=20)
ax.set_ylabel("Allele frequency", fontsize=18)
ax.set_xlabel("Years since population onset", fontsize=18)

# SAVE
# Save the figure.
fig.savefig(os.path.join(FIGDIR, "fig-01.jpg"))
pyplot.close(fig)


# # # # #
# GRID PLOTS

# Conditions to plot a grid for.
grid_cons = [ \
    "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1", \
    ]

# Run through all to-be-plotted conditions.
for ci, con in enumerate(grid_cons):
    # Parse the settings to find the relevant environmental settings.
    env = { \
        "environment_danger":           [], \
        "environment_nutrition":        [], \
        "environment_contamination":    [], \
        }
    for econ in data[con].keys():
        for key in env.keys():
            if data[con][econ]["settings"][key] not in env[key]:
                env[key].append(data[con][econ]["settings"][key])
    # Organise the environmental settings.
    danger_order = ["none", "low", "mid", "high", "higher", "highest"]
    env_danger = []
    for level in danger_order:
        if level in env["environment_danger"]:
            env_danger.append(level)
            env["environment_danger"].remove(level)
    if len(env["environment_danger"]) > 0:
        env_danger.extend(env["environment_danger"])
    env["environment_danger"] = env_danger
    env["environment_nutrition"].sort()
    env["environment_contamination"].sort()
    
    # Organise the number of subplots, and the x- and y-axes for each.
    n_panels = len(env["environment_nutrition"])
    nutrition_kj = ENV_NUTRITION_NEUTRAL * \
        (numpy.array(env["environment_nutrition"], dtype=numpy.float32)/100.0)
    nutrition_rounded = numpy.round(nutrition_kj, decimals=0).astype( \
        numpy.int32)
    title_labels = list(map(str, nutrition_rounded))
    x_labels = list(map(str, env["environment_contamination"]))
    x_ticks = list(range(0, len(x_labels)))
    y_labels = list(map(str, [ENV_DANGER[ed] for ed in env["environment_danger"]]))
    y_ticks = list(range(0, len(y_labels)))
    
    # Create a new figure.
    fig, ax = pyplot.subplots(ncols=n_panels, nrows=3, sharey=True, \
        sharex=True, figsize=(n_panels*7.0, 3*6.5), dpi=600.0)
    fig.subplots_adjust(left=0.06, right=0.95, bottom=0.05, top=0.9, \
        wspace=0.1, hspace=0.1)
    fig.suptitle(r"Environmental nutrition (kJ/day)", fontsize=42)
    unicode_font_prop = matplotlib.font_manager.FontProperties( \
        fname=os.path.join(DIR, "DejaVuSans.ttf"))

    # Construct the data matrices.
    m = {}
    p_extinct = {}
    vlim = { \
        "m_culture": 0.0, \
        "m_pheno": 0.0, \
        }
    ms = {}
    d_chance = {}
    for ai in range(n_panels):
        m[ai] = {}
        p_extinct[ai] = {}
        ms[ai] = {}
        d_chance[ai] = {}
        for var in ["m_culture", "m_pheno"]:
            shape = [len(y_labels), len(x_labels)]
            m[ai][var] = numpy.zeros(shape, dtype=numpy.float32) * numpy.NaN
            p_extinct[ai][var] = numpy.zeros(shape, dtype=numpy.float32) \
                * numpy.NaN
            ms[ai][var] = numpy.zeros(shape, dtype=numpy.float32) * numpy.NaN
            for i, env_contamination in enumerate(env["environment_contamination"]):
                for j, env_danger in enumerate(env["environment_danger"]):
                    # Construct the environmental condition name.
                    econ = "environment_danger-{}_environment_nutrition-{}_environment_contamination-{}".format( \
                        env_danger, env["environment_nutrition"][ai], \
                        env_contamination)
                    # Compute the proportion of surviving populations at the 
                    # final year of each simulation.
                    n_extinct = numpy.sum(numpy.isnan( \
                        data[con][econ][var][-1,:]).astype(float))
                    p_extinct[ai][var][j,i] = float(n_extinct) / \
                        float(data[con][econ][var].shape[1])
                    # For all living populations, first average over runs, and 
                    # then compute the difference between the start and the 
                    # final year, the average, and the pooled SD.
                    if p_extinct[ai][var][j,i] < 1:
                        avg = numpy.nanmean(data[con][econ][var], axis=1)
                        m[ai][var][j,i] = avg[-1] - STARTING_TRAIT_VALUE[var]
                        ms[ai][var][j,i] = avg[-1]
            # Compute the most extreme values.
            vlim_ = numpy.nanmax(numpy.abs(m[ai][var]))
            if vlim_ > vlim[var]:
                vlim[var] = copy.copy(vlim_)
            # Compute the difference between each environment and the control 
            # environment (0 contamination, 0 cost).
            d_chance[ai][var] = ms[ai][var] - ms[ai][var][0,0]
        # Compute the difference map.
        m[ai]["difference"] = m[ai]["m_culture"] - m[ai]["m_pheno"]
        # The proportion of extinction should be the same across traits, so
        # for the difference we can just copy one. This really doesn't have
        # to be split out by variable, but it was just easier for a lazy
        # and wasteful programmer.
        p_extinct[ai]["difference"] = numpy.copy(p_extinct[ai]["m_culture"])
    # Set the overruling vlim if one was set.
    if PRESET_GRID_VLIM != None:
        for key in PRESET_GRID_VLIM:
            vlim[key] = PRESET_GRID_VLIM[key] + 0.004
    # Overrule the individual vlim to set them all to the highest option.
    if EQUALISE_GRID_VLIM:
        max_vlim = max(vlim["m_culture"], vlim["m_pheno"])
        vlim["m_culture"] = max_vlim
        vlim["m_pheno"] = max_vlim
    # The maximum colour intensity for the difference is set as whichever trait
    # has the highest intensity.
    vlim["difference"] = max(vlim["m_culture"], vlim["m_pheno"])
    # Set the vlim for Cohen's d.
    vlim["d_chance"] = 0.244
    # Compute the limit and steps for the colour bar.
    cbar_ticks = {}
    for var in vlim.keys():
        vstep = numpy.round(0.5*vlim[var], decimals=3) - 0.002
        cbar_ticks[var] = [-2*vstep, -vstep, 0.0, vstep, 2*vstep]

    # Loop through the subplots.
    for ai in range(n_panels):
        
        # Set the axis title.
#        if ai == n_panels//2:
#            # The following should work with Tex, BUT that gives the WORST
#            # grief with the unicode character, so just leaving this in for
#            # the unlikely event that the unicode bug is fixed / worked around.
##            title = \
##                r"\fontsize{{{}pt}}{{3em}}\selectfont{{}}{{{}\\\\}}".format( \
##                40, "Cost of contamination") + \
##                r"{{\fontsize{{{}pt}}{{3em}}\selectfont{{}}{{{}}}}}".format( \
##                32, r"$P(death|kJ_{contaminated}$")
#            # The following works fine without Tex.
#            title = r"Cost of contamination, $P(death|kJ_{contaminated}$"
#        else:
#            title = r"\\"
        title = r"{}".format(title_labels[ai])
        ax[0,ai].set_title(title, fontsize=32)
        
        # Plot the matrices.
        for aj, var in enumerate(["m_culture", "m_pheno", "d_chance"]):
            
            # Set the axis' background colour to grey.
            ax[aj,ai].set_facecolor("#888888")
            
            # Choose the colour map.
            if var in ["m_culture", "m_pheno"]:
                cmap = "RdBu_r"
            elif var in ["difference"]:
                cmap = "BrBG"
            elif var in ["d_chance"]:
                cmap = "PiYG_r"
            else:
                cmap = "gray"
            cmap = matplotlib.cm.get_cmap(cmap)
            
            # Choose the colour bar axis label.
            if var == "m_culture":
                clbl = r"Culture $\Delta \bar{s}$"
            elif var == "m_pheno":
                clbl = r"Genotype $\Delta \bar{s}$"
            elif var == "difference":
                clbl = "Difference"
            elif var == "d_chance":
                clbl = "Difference from control"
            
            # Make space for a colour bar.
            divider = make_axes_locatable(ax[aj,ai])
            bax = divider.append_axes("right", size="5%", pad=0.05)
            # Add a colour bar.
            norm = matplotlib.colors.Normalize(vmin=-vlim[var], \
                vmax=vlim[var])
            cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, \
                norm=norm, ticks=cbar_ticks[var], orientation='vertical')
            if ai == ax.shape[1]-1:
                cbar.set_label(clbl, fontsize=32)
                cbar.ax.tick_params(labelsize=16)
            else:
                cbar.ax.set_yticklabels([])

            # Plot the matrix.
            if var == "d_chance":
                # Draw culture in the top-left triangles.
                for i in range(d_chance[ai]["m_culture"].shape[0]):
                    for j in range(d_chance[ai]["m_culture"].shape[1]):
                        # Draw the top-left triangle. We're extending it 
                        # ever-so-slightly towards the bottom and the right,
                        # so that the fill extends to underneath the other
                        # triangle. If we don't, we end up with a line of
                        # the background colour shimmering through. This is a
                        # product of the anti-aliasing, but it looks weird
                        # without, so... If it works, eh?
                        x = numpy.array([i-0.50, i+0.52, i-0.50])
                        y = numpy.array([j-0.52, j+0.50, j+0.50])
                        ax[aj,ai].fill(x, y, \
                            facecolor=cmap(norm(d_chance[ai]["m_culture"][j,i])), \
                            edgecolor="none", antialiased=True)
#                        # Draw a triangle to outline the filled area.
#                        xy = numpy.zeros((3,2), dtype=numpy.float64)
#                        xy[:,0] = x
#                        xy[:,1] = y
#                        triangle = Polygon(xy, closed=True, ls="-", lw=1, \
#                            edgecolor="#000000", facecolor="none", alpha=1.0)
#                        ax[aj,ai].add_patch(triangle)
                # Draw genes in the bottom-right triangles.
                for i in range(d_chance[ai]["m_pheno"].shape[0]):
                    for j in range(d_chance[ai]["m_pheno"].shape[1]):
                        # Draw the bottom-right triangle.
                        x = numpy.array([i-0.5, i+0.5, i+0.5])
                        y = numpy.array([j-0.5, j-0.5, j+0.5])
                        ax[aj,ai].fill(x, y, \
                            facecolor=cmap(norm(d_chance[ai]["m_pheno"][j,i])), \
                            edgecolor="none", antialiased=True)
#                        # Draw a triangle to outline the filled area.
#                        xy = numpy.zeros((3,2), dtype=numpy.float64)
#                        xy[:,0] = x
#                        xy[:,1] = y
#                        triangle = Polygon(xy, closed=True, ls="-", lw=1, \
#                            edgecolor="#000000", facecolor="none", alpha=1.0)
#                        ax[aj,ai].add_patch(triangle)
                # Draw a square outline for each box.
                for i in range(d_chance[ai]["m_culture"].shape[0]):
                    for j in range(d_chance[ai]["m_culture"].shape[1]):
                        xy = numpy.zeros((4,2), dtype=numpy.float64)
                        xy[:,0] = numpy.array([i-0.5, i-0.5, i+0.5, i+0.5])
                        xy[:,1] = numpy.array([j-0.5, j+0.5, j+0.5, j-0.5])
                        square = Polygon(xy, closed=True, ls="-", lw=2, \
                            edgecolor="#000000", facecolor="none", alpha=1.0, \
                            antialiased=True)
                        ax[aj,ai].add_patch(square)
                # Equalise the aspect ratio.
                ax[aj,ai].set_xlim([-0.51, d_chance[ai]["m_pheno"].shape[0]-0.49])
                ax[aj,ai].set_ylim([-0.51, d_chance[ai]["m_pheno"].shape[1]-0.49])
                ax[aj,ai].set_aspect("equal")
                # Top-left triangle shows culture.
#                ax[aj,ai].imshow(d_chance[ai]["m_culture"], cmap=cmap, \
#                    vmax=vlim[var], vmin=-vlim[var], origin="lower", \
#                    interpolation="none", aspect="equal")
                # Lower triangle shows genetics.
            else:
                ax[aj,ai].imshow(m[ai][var], cmap=cmap, vmax=vlim[var], \
                    vmin=-vlim[var], origin="lower", interpolation="none", \
                    aspect="equal")
        
            # Annotate the dead populations.
            if var == "d_chance":
                var = "difference"
            for i in range(m[ai][var].shape[0]):
                for j in range(m[ai][var].shape[1]):
                    if p_extinct[ai][var][i,j] > 0:
                        # Skull unicode: \u1f480
                        # Skull and crossbones unicode: \u2620
                        txt = ax[aj,ai].annotate(u"\u2620", (j-0.3,i-0.2), \
                            fontsize=42, fontproperties=unicode_font_prop, \
                            color="#000000")
                        txt.set_alpha(p_extinct[ai][var][i,j])

            # Set the x- and y-ticks.
            ax[aj,ai].set_xticks(x_ticks)
            ax[aj,ai].set_yticks(y_ticks)
            # Remove tick labels (for now).
            ax[aj,ai].set_xticklabels([])
            ax[aj,ai].set_yticklabels([])
        
    # Set tick labels.
    for aj in range(ax.shape[0]):
        ax[aj,0].set_yticklabels(y_labels, fontsize=16)
    for ai in range(ax.shape[1]):
        ax[-1,ai].set_xticklabels(x_labels, fontsize=16)
    
    # Set axis labels.
    ax[-1,n_panels//2].set_xlabel("Environmental contamination (%)", \
        fontsize=42)
    ax[1,0].set_ylabel("Cost of contamination", fontsize=42)
    
    # Save and close the figure.
    fig.savefig(os.path.join(FIGDIR, "gridplot_{}.jpg".format(con)))
    pyplot.close(fig)

raise Exception("DEBUG")


# # # # #
# STATISTICS

# Create a dict to contain stats, so we can plot them later.
stats = {}

# STATS 1: Comparison of culture+polygenic simulations with and without
# natural selection.
comparisons = { \
    "without-mutation":
    ["disgust-0_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-0", \
        "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-0"], \
    "with-mutation":
    ["disgust-0_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1", \
        "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1"], \
    "combined":
    [["disgust-0_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-0", \
        "disgust-0_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1"], \
        ["disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-0", \
        "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1"]], \
    "combined-restricted":
    [["disgust-0_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-0", \
        "disgust-0_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-1"], \
        ["disgust-1_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-0", \
        "disgust-1_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-1"]], \
    }
# List of variables to compare.
comparison_vars = ["m_culture", "m_pheno"]
# Loop through all comparisons.
for ci, comp_name in enumerate(comparisons.keys()):
    stats[comp_name] = {}
    (con1, con2) = comparisons[comp_name]
    for vi, var in enumerate(comparison_vars):
        stats[comp_name][var] = {}

        # GET DATA
        if type(con1) == list:
            years = data[con1[0]]["year"][:,0]
            n_list = [data[con]["n"] for con in con1]
            n_data1 = numpy.hstack(n_list)
            m_list = [data[con][var] for con in con1]
            m_data1 = numpy.hstack(m_list)
            sd_list = [data[con][var.replace("m_", "sd_")] for con in con1]
            sd_data1 = numpy.hstack(sd_list)
        else:
            years = data[con1]["year"][:,0]
            n_data1 = data[con1]["n"]
            m_data1 = data[con1][var]
            sd_data1 = data[con1][var.replace("m_", "sd_")]
        if type(con2) == list:
            n_list = [data[con]["n"] for con in con2]
            n_data2 = numpy.hstack(n_list)
            m_list = [data[con][var] for con in con2]
            m_data2 = numpy.hstack(m_list)
            sd_list = [data[con][var.replace("m_", "sd_")] for con in con2]
            sd_data2 = numpy.hstack(sd_list)
        else:
            n_data2 = data[con2]["n"]
            m_data2 = data[con2][var]
            sd_data2 = data[con2][var.replace("m_", "sd_")]

        # COMPARE SIMULATIONS
        # Welch's t-test at each year in the simulations, to compare
        # simulation runs.
        n1 = m_data1.shape[1]
        m1 = numpy.mean(m_data1, axis=1)
        sd1 = numpy.std(m_data1, axis=1)
        n2 = m_data2.shape[1]
        m2 = numpy.mean(m_data2, axis=1)
        sd2 = numpy.std(m_data2, axis=1)
        t_sim = (m1 - m2) / numpy.sqrt(((sd1**2)/n1) + ((sd2**2)/n2))
        v_sim = ((((sd1**2)/n1) + ((sd2**2)/n2))**2) \
            / (((sd1**4)/(n1**2*(n1-1))) + ((sd2**4)/(n2**2*(n2-1))))
        df_sim = numpy.round(v_sim, decimals=0).astype(int)
        p_sim = 2 * (1.0 - scipy.stats.t.cdf(numpy.abs(t_sim), df_sim))
        stats[comp_name][var]["sim_welch_t"] = numpy.copy(t_sim)
        stats[comp_name][var]["sim_welch_p"] = numpy.copy(p_sim)
        # One-sample t-test to compare simulations against null (M==0.75)
        t_sim1_1samp, p_sim1_1samp = scipy.stats.ttest_1samp(m_data1, 0.75, \
            axis=1)
        p_05 = p_sim1_1samp
        t_sim2_1samp, p_sim2_1samp = scipy.stats.ttest_1samp(m_data2, 0.75, \
            axis=1)
        # Write stats to file.
        with open(os.path.join(STATSDIR, "sim_stats_{}_{}.csv".format( \
            comp_name, var)), "w") as f:
            header = ["year", "n_sim", \
                "welch_t", "welch_df", "welch_p", \
                "1samp1_t", "1samp1_p", \
                "1samp2_t", "1samp2_p"]
            f.write(",".join(header))
            for i in range(t_sim.shape[0]):
                line = [int(years[i]), n1, \
                    t_sim[i], df_sim[i], p_sim[i], \
                    t_sim1_1samp[i], p_sim1_1samp[i], \
                    t_sim2_1samp[i], p_sim2_1samp[i]]
                f.write("\n" + ",".join(map(str, line)))

        # COMPARE SIMULATED POPULATIONS
        # Compute population sizes across runs, weighted means, and pooled
        # standard deviations.
        n1 = numpy.sum(n_data1, axis=1).astype(float)
        m1 = numpy.sum(m_data1*n_data1, axis=1) / n1
        sd1 = numpy.sum(sd_data1 * n_data1, axis=1) / n1
        n2 = numpy.sum(n_data2, axis=1).astype(float)
        m2 = numpy.sum(m_data2*n_data2, axis=1) / n2
        sd2 = numpy.sum(sd_data2 * n_data2, axis=1) / n2
        # Compute Welch's t-test from the populations.
        t = (m1 - m2) / numpy.sqrt(((sd1**2)/n1) + ((sd2**2)/n2))
        v = ((((sd1**2)/n1) + ((sd2**2)/n2))**2) \
            / (((sd1**4)/(n1**2*(n1-1))) + ((sd2**4)/(n2**2*(n2-1))))
        df = numpy.round(v, decimals=0)
        p = 2 * (1.0 - scipy.stats.t.cdf(numpy.abs(t), df))
        # One sample t-test to compare against null (M==0.75)
        t_1samp = (m2 - 0.75) / (sd2 / numpy.sqrt(n2))
        p_1samp = 2 * (1.0 - scipy.stats.t.cdf(numpy.abs(t_1samp), n2-1))
        # Write stats to file.
        with open(os.path.join(STATSDIR, "pop_stats_{}_{}.csv".format( \
            comp_name, var)), "w") as f:
            header = ["year", "N1", "M1", "SD1", "N2", "M2", "SD2",\
                "welch_t", "welch_df", "welch_p", \
                "1samp_t", "1samp_p"]
            f.write(",".join(header))
            for i in range(n1.shape[0]):
                line = [int(years[i]), \
                    n1[i], m1[i], sd1[i], n2[i], m2[i], sd2[i], \
                    t[i], df[i], p[i], t_1samp[i], p_1samp[i]]
                f.write("\n" + ",".join(map(str, line)))


# STATS 2: P(low allele) difference from 0.5 (one-sample t-tests) in
# simulations WITHOUT mutation.
test_conditions = ["polygenic", "monogenic_high", "monogenic_low"]
stats["p_low_allele"] = {}
for ci, test_con in enumerate(test_conditions):
    # Select the condition.
    if test_con == "polygenic":
        con = "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-polygenic_mutation-0"
    elif test_con == "monogenic_high":
        con = "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-monogenic_highest_mutation-0"
    elif test_con == "monogenic_low":
        con = "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-monogenic_lowest_mutation-0"
    # Test whether the proportion of low alleles is different from 0.5
    t, p = scipy.stats.ttest_1samp(data[con]["p_low_allele"], 0.5, axis=1)
    stats["p_low_allele"][test_con] = {}
    stats["p_low_allele"][test_con]["t"] = numpy.copy(t)
    stats["p_low_allele"][test_con]["p"] = numpy.copy(p)
    # Write stats to file.
    with open(os.path.join(STATSDIR, "sim_stats_no-mutation_low-allele_{}.csv".format( \
        test_con, var)), "w") as f:
        header = ["year", "n_sim", "t", "p"]
        f.write(",".join(header))
        for i in range(t.shape[0]):
            line = [int(years[i]), data[con]["p_low_allele"].shape[1], t[i], \
                p[i]]
            f.write("\n" + ",".join(map(str, line)))

# STATS 3: P(allele<0.75) difference from 0.5 (one-sample t-tests) in
# simulations WITH mutation.
test_conditions = ["polygenic", "monogenic_high", "monogenic_low"]
geno_vars = ["p_geno_70", "p_geno_71", "p_geno_72", "p_geno_73", "p_geno_74"]
half_var = "p_geno_75"
stats["p_lower_allele"] = {}
for ci, test_con in enumerate(test_conditions):
    # Select the condition.
    if test_con == "polygenic":
        con = "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-polygenic_mutation-1"
    elif test_con == "monogenic_high":
        con = "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-monogenic_highest_mutation-1"
    elif test_con == "monogenic_low":
        con = "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-monogenic_lowest_mutation-1"
    # Collect the data.
    y = numpy.dstack([data[con][geno] for geno in geno_vars])
    prop = numpy.sum(y, axis=2)
    if half_var is not None:
        prop += data[con][half_var] / 2.0
    # Test whether the proportion of low alleles is different from 0.5
    t, p = scipy.stats.ttest_1samp(prop, 0.5, axis=1)
    stats["p_lower_allele"][test_con] = {}
    stats["p_lower_allele"][test_con]["t"] = numpy.copy(t)
    stats["p_lower_allele"][test_con]["p"] = numpy.copy(p)
    # Write stats to file.
    with open(os.path.join(STATSDIR, "sim_stats_mutation_low-allele_{}.csv".format( \
        test_con, var)), "w") as f:
        header = ["year", "n_sim", "t", "p"]
        f.write(",".join(header))
        for i in range(t.shape[0]):
            line = [int(years[i]), y.shape[1], t[i], p[i]]
            f.write("\n" + ",".join(map(str, line)))


# # # # #
# FIGURES

# These plots are figures for the manuscript.

# FIGURE 1: Four-panel figure showing the main findings.
fig, axes = pyplot.subplots(nrows=2, ncols=2, figsize=(18,12), dpi=300)
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, \
    hspace=0.3, wspace=0.3)
# Create a normed colour map to indicate years. Usage: cmap(norm(year))
cmap = matplotlib.cm.get_cmap(YEAR_CMAP)
min_year = data[data.keys()[0]]["year"][0,0]
max_year = data[data.keys()[0]]["year"][-1,0]
norm = matplotlib.colors.Normalize(vmin=0, vmax=max_year)
# Find the indices of all years to plot in histogram plots.
year_indices = numpy.where(numpy.isin(data[data.keys()[0]]["year"][:,0], \
    PLOT_YEARS))[0]

# PANEL A (top left): Histograms of the cultural trait.
ax = axes[0,0]
# Draw a vertical line across x=0.75 to indicate the strating position.
ax.axvline(x=0.75, lw=3, ls="--", color="#000000", alpha=0.5)
# Shade the area that indicates less-than-average disgust avoidance.
ax.axvspan(0.75, 1.00, color="#000000", alpha=0.1)
# Loop through all years that should be plotted.
for i in year_indices:
    # Get the current year.
    year = data[data.keys()[0]]["year"][i,0]
    # Compute the average and standard error across runs for both the mutation
    # and non-mutation runs.
    con1 = "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-0"
    con2 = "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1"
    y = numpy.hstack((data[con1]["culture_hist"][i,:,:], \
        data[con2]["culture_hist"][i,:,:]))
    m = numpy.mean(y, axis=1)
    sem = numpy.std(y, axis=1) / numpy.sqrt(y.shape[1])
    # Plot the average and shade the error.
    ax.plot(MULTIVAR_BINS["culture_hist"]["x"], m, "-", lw=1, \
        color=cmap(norm(year)), alpha=0.8)
    ax.fill_between(MULTIVAR_BINS["culture_hist"]["x"], m-sem, m+sem, \
        color=cmap(norm(year)), alpha=0.3)
# Set colour bar ticks.
xticks = range(0, int(max_year+1), int(max_year//5))
xticklabels = [str(int(x//1000))+"k" for x in xticks]
xticklabels[0] = "0"
# Add the colour bar.
divider = make_axes_locatable(ax)
bax = divider.append_axes("right", size="5%", pad=0.05)
cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, norm=norm, \
    ticks=xticks, orientation='vertical')
cbar.set_ticklabels(xticklabels)
cbar.ax.tick_params(labelsize=12)
cbar.set_label("Years since population onset", fontsize=18)
# Finish this plot.
#ax.set_xticks(MULTIVAR_BINS["culture_hist"]["x"])
#ax.set_xticklabels(MULTIVAR_BINS["culture_hist"]["xticks"], rotation=80)
xticks = numpy.round(numpy.arange(0.5, 1.01, 0.1), decimals=1)
xticklabels = list(map(str, xticks))
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=14, rotation=80)
ax.set_xlim(0.5, 1.0)
ax.set_ylim(bottom=0)
ax.set_xlabel(MULTIVAR_BINS["culture_hist"]["xlabel"], fontsize=20)
ax.set_ylabel(MULTIVAR_BINS["culture_hist"]["ylabel"], fontsize=20)

# PANEL B (top right): Trait (culture, phenotype with mutation, phenotype
#   without mutation, and controls (without natural selection for disgust)
ax = axes[0,1]
## Draw a horizontal line across y=0.75 to indicate the strating position.
#ax.axhline(y=0.75, lw=1, ls="--", color="#000000", alpha=0.3)
# Draw all the conditions we want in here.
draw_specs = [ \
    # Culture when culture and polygenic are modelled.
    ["Culture", "m_culture", "sd_culture", \
        ["disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-0", \
        "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1"], \
        "-", COLS["orange"]], \
    # Phenotype when culture and polygenic (with mutations) are modelled.
    ["Phenotype (polygenic)", "m_pheno", "sd_pheno", \
        "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1", \
        "-", COLS["green"]], \
    # Phenotype when only monogenic (high allele dominant, with mutations) is 
    # modelled.
    ["Phenotype (monogenic, high dominant)", "m_pheno", "sd_pheno", \
        "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-monogenic_highest_mutation-1", \
        "--", COLS["pink"]], \
    # Phenotype when only monogenic (low allele dominant, with mutations) is 
    # modelled.
    ["Phenotype (monogenic, low dominant)", "m_pheno", "sd_pheno", \
        "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-monogenic_lowest_mutation-1", \
        "--", COLS["vermillion"]], 
    ]
t = data[data.keys()[0]]["year"][:,0]
for line_specs in draw_specs:
    lbl, mean_var, sd_var, con, ls, col = line_specs
    # Compute mean and pooled SD across runs (for culture, we combine the runs
    # with and without mutations, as they do not impact culture).
    if type(con) == list:
        m_list = [data[c][mean_var] for c in con]
        sd_list = [data[c][sd_var] for c in con]
        m = numpy.mean(numpy.hstack(m_list), axis=1)
        sd = numpy.mean(numpy.hstack(sd_list), axis=1)
    else:
        m = numpy.mean(data[con][mean_var], axis=1)
        sd = numpy.mean(data[con][sd_var], axis=1)
    # Plot the average, and shade the standard deviation.
    ax.plot(t, m, ls, lw=3, color=col, label=lbl, alpha=0.5)
    ax.fill_between(t, m-sd, m+sd, color=col, alpha=0.3)
## Add invisible lines to add line styles to legend.
#ax.plot([-100, -90], [-100,-90], "--", lw=3, color="#000000", alpha=0.5, \
#    label="Control")
# Set x ticks.
xticks = range(0, int(max_year+1), int(max_year//5))
xticklabels = [str(int(x//1000))+"k" for x in xticks]
xticklabels[0] = "0"
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=14)
# Finish this plot.
ax.set_xlim(t[0]-100, t[-1])
ax.set_ylim(0.5, 0.8)
ax.legend(loc="lower left", fontsize=14)
ax.set_xlabel("Years since population onset", fontsize=20)
ax.set_ylabel("Trait in population (AU)", fontsize=20)

# PANEL C (bottom-left): Two sub-panels, each showing phenotype histograms.
#   One is for with and one for without mutation.
ax1 = axes[1,0]
divider = make_axes_locatable(ax1)
# Split the axis into two.
ax2 = divider.append_axes("right", size="100%", pad=0.25)
# Add a colour bar axis too.
bax = divider.append_axes("right", size="10%", pad=0.05)
# Loop through the sub-axes to plot histograms for polygenic traits.
cons = [ \
    "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1",
    "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-0",
    ]
for ai, ax in enumerate([ax1,ax2]):
    # Set the sub-axis title.
    if ai == 0:
        ax.set_title("Mutations", fontsize=16)
    elif ai == 1:
        ax.set_title("No mutations", fontsize=16)
    # Draw a vertical line across x=0.75 to indicate the strating position.
    ax.axvline(x=0.75, lw=3, ls="--", color="#000000", alpha=0.5)
    # Shade the area that indicates less-than-average disgust avoidance.
    ax.axvspan(0.75, 1.00, color="#000000", alpha=0.1)
    # Loop through all years that should be plotted.
    for i in year_indices:
        # Get the current year.
        year = data[cons[ai]]["year"][i,0]
        # Compute the average and standard error across runs for both the
        # mutation and non-mutation runs.
        y = data[cons[ai]]["phenotype_hist"][i,:,:]
        m = numpy.mean(y, axis=1)
        sem = numpy.std(y, axis=1) / numpy.sqrt(y.shape[1])
        # Plot the average and shade the error.
        ax.plot(MULTIVAR_BINS["phenotype_hist"]["x"], m, "-", lw=1, \
            color=cmap(norm(year)), alpha=0.8)
        ax.fill_between(MULTIVAR_BINS["phenotype_hist"]["x"], m-sem, m+sem, \
            color=cmap(norm(year)), alpha=0.3)
    # Set ticks and limits.
    xticks = numpy.round(numpy.arange(0.73, 0.771, 0.01), decimals=2)
    xticklabels = list(map(str, xticks))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=14, rotation=80)
    ax.set_xlim(0.73, 0.77)
    ax.set_ylim(bottom=0)
# Equalise y limits.
max_ylim = max([ax1.get_ylim()[1], ax2.get_ylim()[1]])
ax1.set_ylim(top=max_ylim)
ax2.set_ylim(top=max_ylim)
# Remove ticks from the right plot's y axis.
ax2.set_yticks([])
ax2.set_yticklabels([])
# Set colour bar ticks.
xticks = range(0, int(max_year+1), int(max_year//5))
xticklabels = [str(int(x//1000))+"k" for x in xticks]
xticklabels[0] = "0"
# Add the colour bar.
cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, norm=norm, \
    ticks=xticks, orientation='vertical')
cbar.set_ticklabels(xticklabels)
cbar.ax.tick_params(labelsize=12)
cbar.set_label("Years since population onset", fontsize=18)
# Finish this plot.
ax1.set_xlabel(MULTIVAR_BINS["phenotype_hist"]["xlabel"], fontsize=20, \
    horizontalalignment='right', x=1.4)
ax1.set_ylabel(MULTIVAR_BINS["phenotype_hist"]["ylabel"], fontsize=20)

# PANEL D: Bar graphs showing the prevalence of high and low allele in
#   polygenic, monogenic_highest, and monogenic_lowest in the population.
ax = axes[1,1]
ax.axhline(y=0.5, lw=3, ls="--", color="#000000", alpha=0.3)
bar_year_indices = numpy.where(numpy.isin(data[data.keys()[0]]["year"][:,0], \
    BARPLOT_YEARS))[0]
bar_x = range(1, 9, 2)
bar_w = 0.25
cons = [ \
    ["disgust-0_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-0",
        "disgust-0_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-0"],
    "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-polygenic_mutation-0",
    "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-monogenic_highest_mutation-0",
    "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-monogenic_lowest_mutation-0",
    ]
for ci, con in enumerate(cons):
    for bi, i in enumerate(bar_year_indices):
        # Get the current year.
        if type(con) == list:
            year = data[con[0]]["year"][i,0]
        else:
            year = data[con]["year"][i,0]
        # Set the label.
        if ci == 0:
            if year < 1000:
                lbl_year = str(int(round(year, ndigits=0)))
            else:
                lbl_year = str(int(round(year / 1000, ndigits=0))) + "k"
            lbl = "Year {}".format(lbl_year)
        else:
            lbl = None
        # Compute mean and standard error of the mean across runs.
        if type(con) == list:
            y = numpy.hstack([data[c]["p_low_allele"][i,:] for c in con])
        else:
            y = data[con]["p_low_allele"][i,:]
        m = numpy.mean(y)
        sem = numpy.std(y) / numpy.sqrt(y.shape[0])
        # Draw the bar and the errorbar.
        ax.bar(bar_x[ci]+bi*bar_w, m, bar_w, color=cmap(norm(year)), label=lbl)
        ax.errorbar(bar_x[ci]+bi*bar_w, m, yerr=sem, ecolor="#000000", \
            elinewidth=3, capsize=3)
# Set the x axis ticks and limits.
ax.set_xlim(bar_x[0]-bar_w, bar_x[-1]+len(bar_year_indices)*bar_w+bar_w)
xticks = numpy.array(bar_x) + 0.5*len(bar_year_indices)*bar_w - bar_w/2.0
xticklabels = ["Polygenic\n(control)", "Polygenic", \
    "Monogenic\n(high dominant)", "Monogenic\n(low dominant)"]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=16)
# Set the y axis label and limits.
ax.set_ylabel("Proportion low allele in population", fontsize=20)
ax.set_ylim(0, 1.05)
# Add the legend.
ax.legend(loc="upper left", fontsize=14)
# SAVE the figure.
fig.savefig(os.path.join(FIGDIR, "fig-01.jpg"))
pyplot.close(fig)


# FIGURE 2: SIMULATION STATS
# Create a new figure.
fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(18,10), dpi=300)
fig.subplots_adjust(left=0.07, right=0.97, top=0.99, bottom=0.07, \
    hspace=0.1, wspace=0.2)
axs = [axes[0]]
divider = make_axes_locatable(axs[0])
axs.append(divider.append_axes("bottom", size="100%", pad=0.5))
# Create a matrix of years.
years = data[data.keys()[0]]["year"][:,0]
max_year = numpy.max(years)
# PANEL A: Culture and polygenic p values
ax = axs[0]
for ci, comp_name in enumerate(["combined", "combined-restricted"]):
    ax.set_yscale("log")
    ls = {0.05:":", 0.005:"--", 0.0005:"-"}
    for alpha in [0.05, 0.005, 0.0005]:
        ax.axhline(y=alpha, lw=1, ls=ls[alpha], color="#000000", alpha=0.3)
        ax.annotate(r"$\alpha$={}".format(alpha), (500,alpha+alpha*0.1), \
            fontsize=12, alpha=0.3)
    if comp_name == "combined":
        linestyle = "-"
        draw_lbl = True
    elif comp_name == "combined-restricted":
        linestyle = ":"
        draw_lbl = False
    for vi, var in enumerate(["m_culture", "m_pheno"]):
        if var == "m_culture":
            col = COLS["orange"]
            lbl = "Cultural trait"
        elif var == "m_pheno":
            col = COLS["green"]
            lbl = "Phenotype (polygenic)"
        else:
            col = COLS["pink"]
        if not draw_lbl:
            lbl = None
        ax.plot(years, stats[comp_name][var]["sim_welch_p"], ls=linestyle, \
            lw=3, color=col, alpha=0.5, label=lbl)
    if comp_name == "combined":
        linestyle = "-"
        ax.plot([-10,-10], [-10,-10], lw=3, ls=linestyle, alpha=0.5, \
            color="#000000", label="Realistic cultural evolution")
    elif comp_name == "combined-restricted":
        linestyle = ":"
        ax.plot([-10,-10], [-10,-10], lw=3, ls=linestyle, alpha=0.5, \
            color="#000000", label="Restricted cultural evolution")
ax.set_ylabel( \
    "Real vs. control simulation\n(Welch's t-test $p$ value)", \
    fontsize=20)
#ax.set_xlabel("Years since population onset", fontsize=20)
xticks = range(0, int(max_year+1), int(max_year//5))
xticklabels = [str(int(x//1000))+"k" for x in xticks]
xticklabels[0] = "0"
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=14)
ax.set_xlim(0, max_year)
ax.set_ylim(1e-6, 1)
ax.legend(loc="lower right", fontsize=14)

# PANEL B: Genetic inheritance p values.
# Set up the plot.
ax = axs[1]
ax.set_yscale("log")
ls = {0.05:":", 0.005:"--", 0.0005:"-"}
for alpha in [0.05, 0.005, 0.0005]:
    ax.axhline(y=alpha, lw=1, ls=ls[alpha], color="#000000", alpha=0.3)
    ax.annotate(r"$\alpha$={}".format(alpha), (6.2e4,alpha+alpha*0.1), \
        fontsize=12, alpha=0.3)
ls = {"p_low_allele":"-", "p_lower_allele":":"}
col = {"polygenic":COLS["green"], "monogenic_high":COLS["pink"], \
    "monogenic_low":COLS["vermillion"]}
lbl = {"polygenic":"Polygenic", \
    "monogenic_high":"Monogenic (high dominant)", \
    "monogenic_low":"Monogenic (low dominant)", \
    "p_low_allele": "Without mutations", \
    "p_lower_allele": "With mutations", \
    }
# Plot the different gene inheritances.
for vi, var in enumerate(["p_low_allele", "p_lower_allele"]):
    for genotype in ["polygenic", "monogenic_high", "monogenic_low"]:
        if vi == 0:
            l = lbl[genotype]
        else:
            l = None
        ax.plot(years, stats[var][genotype]["p"], lw=3, ls=ls[var], \
            color=col[genotype], alpha=0.5, label=l)
# Create an out-of-view labels.
for vi, var in enumerate(["p_low_allele", "p_lower_allele"]):
    ax.plot([-10,-10], [-10,-10], lw=3, ls=ls[var], color="#000000", \
        alpha=0.5, label=lbl[var])
# Finish the plot.
ax.set_xlabel("Years since population onset", fontsize=20)
ax.set_ylabel("P(low allele) > 0.5\n(One-sample t-test $p$ value)", \
    fontsize=20)
xticks = range(0, int(max_year+1), int(max_year//5))
xticklabels = [str(int(x//1000))+"k" for x in xticks]
xticklabels[0] = "0"
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=14)
ax.set_xlim(0, max_year)
ax.set_ylim(1e-20, 1)
ax.legend(loc="lower left", fontsize=14)

# PANEL C: Proportions of each allele under genetic mutation.
# Grab the axis, and split it into 11.
ax = [axes[1]]
divider = make_axes_locatable(ax[0])
for i in range(10):
    ax.append(divider.append_axes("bottom", size="100%", pad=0.05))
# Plot all the alleles.
geno_vars = ["p_geno_70", "p_geno_71", "p_geno_72", "p_geno_73", "p_geno_74", \
    "p_geno_75", "p_geno_76", "p_geno_77", "p_geno_78", "p_geno_79", \
    "p_geno_80"]
for genotype in ["polygenic", "monogenic_high", "monogenic_low"]:
    # Set the condition.
    if genotype == "polygenic":
        con = "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-polygenic_mutation-1"
    elif genotype == "monogenic_high":
        con = "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-monogenic_highest_mutation-1"
    elif genotype == "monogenic_low":
        con = "disgust-1_culture-0_cultrestrict-1_genes-1_genetype-monogenic_lowest_mutation-1"
    # Loop through all conditions.
    for i, var in enumerate(geno_vars):
        # Compute the mean and standard error of the mean.
        m = numpy.mean(data[con][var], axis=1)
        sd = numpy.mean(data[con][var], axis=1)
        sem = sd / float(data[con][var].shape[1])
        # Plot the mean and standard error.
        ax[i].plot(years, m, "-", lw=3, color=col[genotype], alpha=0.5)
        ax[i].fill_between(years, m-sem, m+sem, color=col[genotype], alpha=0.3)
        # Remove plot axis labels.
        ax[i].annotate("Allele 0.{}".format(geno_vars[i][-2:]), (1000, 0.25), \
            fontsize=14)
        ax[i].set_yticks([0.0, 0.15, 0.3])
        ax[i].set_ylim(0,0.4)
        ax[i].set_xticks([])
        ax[i].set_xlim(0, 1e5)
# Finish the panel.
ax[-1].set_xlabel("Years since population onset", fontsize=20)
xticks = range(0, int(max_year+1), int(max_year//5))
xticklabels = [str(int(x//1000))+"k" for x in xticks]
xticklabels[0] = "0"
ax[-1].set_xticks(xticks)
ax[-1].set_xticklabels(xticklabels, fontsize=14)
ax[-1].set_xlim(0, max_year)
ax[5].set_ylabel("Proportion of allele in population", fontsize=20)
    
# SAVE THE FIGURE
fig.savefig(os.path.join(FIGDIR, "fig-02.jpg"))
pyplot.close(fig)


# SUPPLEMENTARY FIGURES
# Supplementary figures showing each of the control conditions, and the
# simulations they matched. One figure is for the cultural, and another for
# the polygenic trait.
for vi, var in enumerate(["culture_hist", "phenotype_hist"]):
    
    # Create a new 4x2 figure.
    fig, axes = pyplot.subplots(nrows=2, ncols=4, figsize=(24,16), dpi=300)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, \
        hspace=0.4, wspace=0.3)

    # Create a normed colour map to indicate years. Usage: cmap(norm(year))
    cmap = matplotlib.cm.get_cmap(YEAR_CMAP)
    min_year = data[data.keys()[0]]["year"][0,0]
    max_year = data[data.keys()[0]]["year"][-1,0]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_year)
    # Find the indices of all years to plot in histogram plots.
    year_indices = numpy.where(numpy.isin(data[data.keys()[0]]["year"][:,0], \
        PLOT_YEARS))[0]
    
    # Loop through all conditions.
    conditions = { \
        "control":[ \
            "disgust-0_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-0", \
            "disgust-0_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1", \
            "disgust-0_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-0", \
            "disgust-0_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-1", \
            ], \
        "real":[ \
            "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-0", \
            "disgust-1_culture-1_cultrestrict-0_genes-1_genetype-polygenic_mutation-1", \
            "disgust-1_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-0", \
            "disgust-1_culture-1_cultrestrict-1_genes-1_genetype-polygenic_mutation-1", \
            ], \
        }
    con_ax_titles = { \
        "control":[ \
            "Control condition,\nrealistic culture,\nwithout gene mutations", \
            "Control condition,\nrealistic culture,\nwith gene mutations", \
            "Control condition,\nrestricted culture,\nwithout gene mutations", \
            "Control condition,\nrestricted culture,\nwith gene mutations", \
            ], \
        "real":[ \
            "Natural selection,\nrealistic culture,\nwithout gene mutations", \
            "Natural selection,\nrealistic culture,\nwith gene mutations", \
            "Natural selection,\nrestricted culture,\nwithout gene mutations", \
            "Natural selection,\nrestricted culture,\nwith gene mutations", \
            ], \
        }
    for row, con_type in enumerate(["control", "real"]):
        for col, con in enumerate(conditions[con_type]):
            
            # Get the handle to the axis.
            ax = axes[row,col]
            # Check if we're dealing with restricted culture.
            if (var == "culture_hist") and (data[con]["settings"]["cultrestrict"]):
                plot_var = "culture_hist_restricted"
            else:
                plot_var = var
            # Set the title.
            ax.set_title(con_ax_titles[con_type][col], fontsize=24)

            # Draw a vertical line across x=0.75 to indicate the strating position.
            ax.axvline(x=0.75, lw=3, ls="--", color="#000000", alpha=0.5)
            # Shade the area that indicates less-than-average disgust avoidance.
            ax.axvspan(0.75, 1.00, color="#000000", alpha=0.1)
            # Loop through all years that should be plotted.
            for i in year_indices:
                # Get the current year.
                year = data[data.keys()[0]]["year"][i,0]
                # Compute the average and standard error across runs for both the mutation
                # and non-mutation runs.
                y = data[con][plot_var][i,:,:]
                m = numpy.mean(y, axis=1)
                sem = numpy.std(y, axis=1) / numpy.sqrt(y.shape[1])
                # Plot the average and shade the error.
                ax.plot(MULTIVAR_BINS[plot_var]["x"], m, "-", lw=1, \
                    color=cmap(norm(year)), alpha=0.8)
                ax.fill_between(MULTIVAR_BINS[plot_var]["x"], m-sem, m+sem, \
                    color=cmap(norm(year)), alpha=0.3)
            # Only add colour bar on right-most column.
            if col == axes.shape[1]-1:
                # Set colour bar ticks.
                xticks = range(0, int(max_year+1), int(max_year//5))
                xticklabels = [str(int(x//1000))+"k" for x in xticks]
                xticklabels[0] = "0"
                # Add the colour bar.
                divider = make_axes_locatable(ax)
                bax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, norm=norm, \
                    ticks=xticks, orientation='vertical')
                cbar.set_ticklabels(xticklabels)
                cbar.ax.tick_params(labelsize=12)
                cbar.set_label("Years since population onset", fontsize=24)
            # Finish this plot.
            if plot_var == "culture_hist_restricted":
                xticks = numpy.round(numpy.arange(0.74, 0.76, 0.01), decimals=2)
                ax.set_xlim(0.74, 0.76)
            elif plot_var == "phenotype_hist":
                xticks = numpy.round(numpy.arange(0.74, 0.761, 0.01), decimals=2)
                ax.set_xlim(0.735, 0.765)
            else:
                xticks = numpy.round(numpy.arange(0.5, 1.01, 0.1), decimals=1)
                ax.set_xlim(0.5, 1.0)
            xticklabels = list(map(str, xticks))
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontsize=18, rotation=80)
            ax.set_ylim(bottom=0)
            if row == axes.shape[0]-1:
                ax.set_xlabel(MULTIVAR_BINS[plot_var]["xlabel"], fontsize=24)
            if col == 0:
                ax.set_ylabel(MULTIVAR_BINS[plot_var]["ylabel"], fontsize=24)
    
    fig.savefig(os.path.join(FIGDIR, "fig-s{}_{}.jpg".format(vi+1,var)))
    pyplot.close(fig)
    

raise Exception()


# # # # #
# CHECK PLOTS

# Plot the following variables for every condition, so that we can compare
# them. These should be similar, so these plots serve as sanity checks.

# TODO: Plot var against time.


# Plot these variables for every condition.
for var in MULTIVARS.keys():
    # Loop through all conditions
    for con in data.keys():
        
        # Skip if this variable does not exist for this condition.
        if var not in data[con].keys():
            continue
        
        # Create a new figure.
        fig, ax = pyplot.subplots(figsize=(8.0, 6.0), dpi=300.0)
        fig.subplots_adjust(left=0.1, top=0.90, bottom=0.2, right=0.9)

        # Create a normed colour map to indicate years.
        # Usage: cmap(norm(year))
        cmap = matplotlib.cm.get_cmap(YEAR_CMAP)
        min_year = data[con]["year"][0,0]
        max_year = data[con]["year"][-1,0]
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_year)
        
        # Find the indices of all years to plot.
        year_indices = numpy.where(numpy.isin(data[con]["year"][:,0], \
            PLOT_YEARS))[0]
        # Loop through all years that should be plotted.
        for i in year_indices:
            # Get the current year.
            year = data[con]["year"][i,0]
            # Compute the average and standard error across runs.
            m = numpy.mean(data[con][var][i,:,:], axis=1)
            sem = numpy.std(data[con][var][i,:,:], axis=1) \
                / numpy.sqrt(data[con][var].shape[2])
            # Plot the average and shade the error.
            ax.plot(MULTIVAR_BINS[var]["x"], m, "-", lw=1, \
                color=cmap(norm(year)), alpha=0.3)
            ax.fill_between(MULTIVAR_BINS[var]["x"], m-sem, m+sem, \
                color=cmap(norm(year)), alpha=0.1)
        # Add the colour bar.
        divider = make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, norm=norm, \
            ticks=[0, max_year], orientation='vertical')
        cbar.set_ticklabels(list(map(int, [0, max_year//1000])))
        cbar.set_label("Year (x1000)", fontsize=14)
        # Finish the plot.
        ax.set_title(con)
        ax.set_xticks(MULTIVAR_BINS[var]["x"])
        ax.set_xticklabels(MULTIVAR_BINS[var]["xticks"], rotation=80)
        ax.set_xlabel(MULTIVAR_BINS[var]["xlabel"], fontsize=18)
        ax.set_ylabel(MULTIVAR_BINS[var]["ylabel"], fontsize=18)
        # Save the figure.
        fig.savefig(os.path.join(CHECKDIR, "{}-{}.png".format(var, con)))
        pyplot.close(fig)

