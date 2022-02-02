import os
import sys
import copy
import time
from multiprocessing import cpu_count, Process

import numpy
import scipy.stats

from simulate_population import Population
from email_notification import email_message

# Ignore NumPy warnings. NOTE: DANGEROUS! Only doing so here after testing
# thoroughly, to surpress the harmless warnings that would otherwise obscure
# our own processing messages.
import warnings
warnings.filterwarnings('ignore')


# # # # #
# FUNCTION TO RUN

def run_simulations(environment_danger, environment_nutrition, \
    environment_contamination, disgust, mutation, genes, gene_type, culture, \
    culture_restriction, n_runs):
    
    # Reseed NumPy's pseudo-random number generator. (This was seeded upon
    # importing NumPy, so parallel processes will have the same seed if they
    # are not reseeded.)
    numpy.random.seed()


    # # # # #
    # CONSTANTS
    
    # BIRTH PROBABILITY
    # Set the initial birth probability. This does not directly translate into a
    # birth rate, as this only applies to women who are currently fertile, which
    # is also impacted by the timing of the previous birth and available nutrition.
    BIRTH_PROB_INIT = 0.42
    # Set the minimum and maximum birth probability. The birth rate is used to 
    # limit fluctuations in the population size (to prevent it from infinite 
    # growth or complete collapse) by incrementing or decrementing it. The
    # following values are the boundaries on the birth probability.
    BIRTH_PROB_MIN = 0.32
    BIRTH_PROB_MAX = 0.52
    # Set the population size above which the birth probability cannot grow.
    MAX_POPULATION_SIZE = 5000
    
    # GENETICS
    # Set to True for gene mutations being randomly sampled from [0, 1]; or to
    # False for them to be sampled from a normal distribution of (current+
    # N(0,GENE_MUTATION_ERROR_SD)).
    GENE_MUTATION_DISCRETE = False
    # The alleles determine the underlying genotype. The beginning values are
    # randomly sampled with replacement from [low, high], and thereafter genotype
    # is determined by natural selection. We're aiming to start at 60% avoidance,
    # so a value of 0.4 here. (Based on data from chimpanzees and bonobos by 
    # Sarabian et al. 2017, 2018).
    LOW_ALLELE = 0.3
    HIGH_ALLELE = 0.5
    INITIAL_GENOTYPE_SD = 0.0132
    # Optionally start with randomly distributed gene values with the mean being
    # the average of the low and the high allele, and SD being the mutation SD.
    ALLELE_RANDOM_INIT = True
    # The gene mutation rate is the probability of an allele becoming a new value.
    GENE_MUTATION_RATE = 6.6e-05
    # Gene mutation error standard deviation. (Errors are sampled from a normal
    # distribution with mean 0 and the defined SD, and added to the current value
    # of the gene.)
    GENE_MUTATION_ERROR_SD = 0.025
    
    # CULTURE
    # The starting value determines the mean of the cultural trait of disgust
    # avoidance.
    CULTURE_STARTING_VALUE = 0.95
    # The error rate is HALF the error rate, and reflects the imperfect
    # reproduction between generations. Normally set at 0.025 (5%), but set to
    # 0.0025 (0.5%) if the SD needs to be matched to the typical phenotype SD.
    CULTURE_ERROR_RATE = 0.025
    # The conformist bias is the proportion of individuals who conform to the
    # population average rather than to one of their parents. Eerkens & Lipo
    # (2005) point out that 10% conformity is enough to hold variance quite
    # stable. On the other hand, Hamilton & Buchanan (2009) find a bias of 0.38
    # along with an error rate of 5%.
    CULTURE_CONFORMIST_BIAS = 0.38
    # Set the cultural mutation rate to 0 for now. Cultural mutations would be
    # "new" ideas, but that mechanism might be a bit controversial.
    CULTURE_MUTATION_RATE = 0
    # Mutations are sampled from a uniform distribution between (min,max). Note
    # that this range is also used to determine the histogram range in the data
    # file, so set it to something sensible even if mutation==False, or if the
    # mutation rate is set to 0.
    if CULTURE_ERROR_RATE == 0.0025:
        CULTURE_MUTATION_RANGE = (0.7, 0.8)
    else:
        CULTURE_MUTATION_RANGE = (0.5, 1.0)
    
    # SIMULATION SPECS
    # This setting determines whether the number of existing files will be 
    # taken into account when running n_runs. If set to False, n_runs will be
    # run. When set to True, n_runs - the number of existing runs.
    SKIP_EXISTING_RUNS = False
    # The following number will be added to file names for each run. For 
    # example, if you set the RUN_FILENR_OFFSET to 3, the first run will be 
    # named "run-00004.csv" instead of "run-00001.csv".
    RUN_FILENR_OFFSET = 0
    # Starting population size.
    N = 1000
    # Number of years to simulate.
    N_YEARS = 100000
    # Write-to-file frequency in years.
    WRITE_FREQ = 100
    # Details of the bins that are to be written to file.
    N_BINS_CULTURE = 31
    N_BINS_GENOTYPE = 102
    N_BINS_PHENOTYPE = 31
    BIN_RANGE_CULTURE = (0.0, 1.0)
    BIN_RANGE_GENOTYPE = (0.0, 1.0)
    BIN_RANGE_PHENOTYPE = (0.0, 1.0)
    
    # FILES AND FOLDERS
    DIR = os.path.dirname(os.path.abspath(__file__))
    DATADIR = os.path.join(DIR, "data")
    if not os.path.isdir(DATADIR):
        os.mkdir(DATADIR)


    # # # # #
    # PROCESS ARGUMENTS
    
    # Restrict the culture variance.
    if culture_restriction:
        CULTURE_ERROR_RATE = 0.0025
        CULTURE_CONFORMIST_BIAS = 0.1
    
    # ENVIRONMENT
    # Attribute a proportion of the total mortality rate to disgusting 
    # elements in the environment. This proportion is an estimate based on 
    # Table 5 in Gurven &  Kaplan (2007), showing that 5.5% of deaths in 
    # hunter-gatherer population the Ache (forest) were due to gastrointestinal
    # illness; and 13.2% among settled Ache. Fever is also a potential outcome
    # of eating contaminated food, and is the cause of 8.1% and 21.7% in the
    # forest and settled Ache, respectively. What proportion of illnes and 
    # death could have been prevented by disgust avoidance is hard to estimate.
    # Here, we assume 2.75% (half of 5.5%), 6.6% (half of 13.2%), and then 
    # 10.45%, 14.3%, and 18.15% (several steps of the same size). In sum, the
    # total preventable death by disgust avoidance is:
    # low danger:       0.0275
    # medium danger:    0.0660
    # high danger:      0.1045
    # higher danger:    0.1430
    # highest danger:   0.1815
    # These values are over the course of a lifetime. We thus divide this further
    # by the average lifespan of about 33 years (for those who reach age 5): 
    # danger-per-year = 1 - (1-lifetime_danger)**(1/lifespan)
    # low danger:       1 - (1-0.0275)**(1.0/33) = 0.000845
    # medium danger:    1 - (1-0.0660)**(1.0/33) = 0.002067
    # high danger:      1 - (1-0.1045)**(1.0/33) = 0.003339
    # higher danger:    1 - (1-0.1430)**(1.0/33) = 0.004665
    # highest danger:   1 - (1-0.1815)**(1.0/33) = 0.006051
    # The above figures are the annual probability of death. If we assume that
    # this annual probability is the outcome of a probability of death at each
    # contaminated kJ of input energy, and we know what annual intake of 
    # contaminated input energy is, we can compute the probability of death
    # per contaminated kJ. Assuming 11083 kJ/day and a 10% contamination rate,
    # this would mean a yearly contaminated energy intake of:
    # contaminated energy per day: 11083 kJ/day * 0.1 contamination rate
    # contaminated energy per year: daily contaminated energy * 365.25 days
    # annual contaminated kJ = 11083 * 365.25 * 0.1 = 404807
    # Chance of living despite contamination = (1-p_per_kJ)**contaminated_kJ
    # Solve for the computed annual rates:
    # low danger:   0.000845 = 1 - ((1-p_per_kJ)**404807)
    #               1 - 0.000845 = (1-p_per_kJ)**404807
    #               (1 - 0.000845)**(1/404807) = 1 - p_per_kJ
    #               1 - (1 - 0.000845)**(1/404807) = p_per_kJ
    #               = 4.0882968687629955e-09
    # To recompute the annual probability of death (for low danger):
    #   p_death_year = 1 - p_survival_year
    #   p_survival_year = p_survival_kJ**yearly_kJ
    #   p_survival_kJ = 1 - 2.0883e-9
    #   => p_death_year = 1 - (1 - 2.0883e-9)**404807 = 0.00845
    # In sum, for all danger levels:
    # low danger:     1 - (1 - 0.000845)**(1.0/404807) = 2.0883e-9
    # medium danger:  1 - (1 - 0.002067)**(1.0/404807) = 5.1114e-9
    # high danger:    1 - (1 - 0.003339)**(1.0/404807) = 8.2622e-9
    # higher danger:  1 - (1 - 0.004665)**(1.0/404807) = 1.1551e-8
    # highest danger: 1 - (1 - 0.006051)**(1.0/404807) = 1.4993e-8
    if environment_danger == "none":
        P_ENVIRONMENT_DEATH = 0.0
    elif environment_danger == "low":
        P_ENVIRONMENT_DEATH = 2.0883e-9
    elif environment_danger == "mid":
        P_ENVIRONMENT_DEATH = 5.1114e-9
    elif environment_danger == "high":
        P_ENVIRONMENT_DEATH = 8.2622e-9
    elif environment_danger == "higher":
        P_ENVIRONMENT_DEATH = 1.1551e-8
    elif environment_danger == "highest":
        P_ENVIRONMENT_DEATH = 1.4993e-8
    # Available nutritional energy (kJ/day) in the environment. The default of 
    # 11083 kJ is based on Pontzer et al. (2012, Table 1), who describe this as
    # the average total energy expenditure per day for men among the Hadza. We use
    # 60%, 80%, 100%, 120%, and 140% of that number.
    ENVIRONMENTAL_NUTRITION = 11083 * (float(environment_nutrition)/100.0)
    # Proportion of contaminated food in the environment. This will determine the
    # maximum extent to which disgust avoidance impacts loss of nutrition. It is
    # decoupled from the extent to which disgust avoidance impacts death for two
    # main reasons: 1) Death can occur after ingestion of a single piece of
    # contaminated food, and even at a 1% contamination, it is quite likely that
    # an individual will come into contact with at least one piece of contaminated
    # food; and 2) Decoupling allows for modelling the impact of disgust avoidance
    # through both illness probability and food reduction separately, thereby
    # making for a more informative simulation. We use 0%, 2%, 5%, 10%, 20%, and 50%
    P_ENVIRONMENT_CONTAMINATION = float(environment_contamination)/100.0
    
    # Construct the name for the output directory.
    outname = \
        "disgust-{}_culture-{}_cultrestrict-{}_genes-{}_genetype-{}_mutation-{}".format( \
        int(disgust), int(culture), int(culture_restriction), int(genes), \
        gene_type, int(mutation))
    envname = \
        "environment_danger-{}_environment_nutrition-{}_environment_contamination-{}".format( \
        environment_danger, environment_nutrition, environment_contamination)
    if not os.path.isdir(os.path.join(DATADIR, outname)):
        os.mkdir(os.path.join(DATADIR, outname))
    outdir = os.path.join(DATADIR, outname, envname)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    
    # # # # #
    # VARIABLE-DEPENDENT CLASS
    
    # Class for genetic AND cultural evolution.
    class DisgustPopulation(Population):
        
        def init_environmental_mortality_probability(self):
            
            """This sets the maximum proportion of mortality that is determined
            by how the traits that we are modelling impact mortality.
            """
            
            # Attribute a proportion of the total mortality rate to disgusting
            # elements in the environment. This proportion is an estimate based on
            # Table 5 in Gurven & Kaplan (2007), showing that 5.4% of deaths in
            # hunter-gatherer population the Ache (forest) were due to 
            # gastrointestinal illness. The exact proportion of those deaths, and
            # its distribution over the lifespan, is explained at the top of this
            # file.
            self._p_environmental_mortality_per_kJ = P_ENVIRONMENT_DEATH
            # Compute the maximum annual probability of death. This is used to
            # compute the total of "regular" mortality to carve out for all
            # mortality based on environment*trait interactions.
            nutrition = min(self._environmental_nutrition, \
                self._nutritional_intake_male)
            self._p_environmental_mortality = 1.0 - \
                (1.0 - self._p_environmental_mortality_per_kJ) ** \
                (nutrition * 365.25 * P_ENVIRONMENT_CONTAMINATION)
    
            
        def init_culture(self):
            
            """This function sets disgust avoidance as a cultural trait.
            """
    
            # Start with empty dicts for genes and their rules.
            self._culture = {}
            self._culture_error_rate = {}
            self._culture_conformist_bias = {}
            # Start with empty dicts for mutation rules.
            self._culture_mutation_rates = {}
            self._culture_mutation_ranges = {}
            
            # Count the number of individuals in the population.
            n = self._specs["age"].shape[0]
            
            # Disgust avoidance as a cultural trait, starting with an average of
            # 0.1 and a standard deviation based on the error rate.
            self._culture_error_rate["disgust"] = CULTURE_ERROR_RATE
            self._culture["disgust"] = numpy.zeros(n, dtype=numpy.float64)
            self._culture["disgust"] = CULTURE_STARTING_VALUE + \
                CULTURE_STARTING_VALUE * numpy.random.randn(n) \
                * self._culture_error_rate["disgust"]
            # The conformist bias is the proportion of the population that will
            # not inherit from their parents, but rather conform to the norm set
            # in society, reflected as the average cultural trait.
            self._culture_conformist_bias["disgust"] = CULTURE_CONFORMIST_BIAS
            # New ideas can be introduced by people randomly thinking of something
            # new throughout their lives. For example, an individual can suddenly
            # decide to eat a particular food, or to display new behaviour. The
            # "mutation rate" indicates the probability of a single person coming
            # up with a completely new idea in a year.
            self._culture_mutation_rates["disgust"] = CULTURE_MUTATION_RATE
            # The range limits where potential ideas can occur. As cultural traits
            # are represented by values that weigh environmental risk, the range
            # of new ideas limits the values that new ideas can take within a
            # trait. New ideas are randomly sampled from a uniform distribution
            # between the min and max indicated by this range.
            self._culture_mutation_ranges["disgust"] = CULTURE_MUTATION_RANGE
    
    
        def init_genotypes(self):
            
            """This function sets disgust avoidance as a genetic trait. Four
            traits are prepared:
                - monogenic, high values dominant for phenotype
                - monogenic, low values dominant for phenotype
                - monogenic, intermediate phenotype
                - polygenic (phenotype is average of all alleles)
            """
    
            # Start with empty dicts for genes and their rules.
            self._genotype = {}
            self._gene_rules = {}
            # Start with empty dicts for mutation rules.
            self._gene_mutation_rates = {}
            self._gene_mutation_discrete = {}
            self._gene_mutation_ranges = {}
            self._gene_mutation_decimals = {}
            self._gene_mutation_error_sd = {}
            
            # Count the number of individuals in the population.
            n = self._specs["age"].shape[0]
            
            # MONOGENIC TRAITS
            for mono_type in ["highest", "lowest", "average"]:
                # Set the trait name.
                trait = "disgust_monogenic_{}".format(mono_type)
                # Set the trait to continuous error rather than discrete within a
                # predefined range.
                self._gene_mutation_discrete[trait] = False
                # Create an Nx2 matrix to store genotype in. The second dimension
                # separates the mother's (n,0) and father's (n,1) contribution.
                self._genotype[trait] = numpy.zeros((n,2), dtype=numpy.float16)
                # Set the rule for how phenotypes come to be from this gene.
                self._gene_rules[trait] = mono_type
                # Randomly assign the high and low value (these are the genotypes),
                # and set the probability of each occurring.
                alleles = [LOW_ALLELE, HIGH_ALLELE]
                p_alleles = [0.5, 0.5]
                # Randomly set mother's contribution.
                self._genotype[trait][:,0] = numpy.random.choice(alleles, \
                    size=n, replace=True, p=p_alleles)
                # Randomly set father's contribution.
                self._genotype[trait][:,1] = numpy.random.choice(alleles, \
                    size=n, replace=True, p=p_alleles)
                # Set the gene mutation rate.
                self._gene_mutation_rates[trait] = GENE_MUTATION_RATE
                # Set the gene mutation error SD.
                self._gene_mutation_error_sd[trait] = GENE_MUTATION_ERROR_SD
                # Set the (min,max) range in which mutations can randomly occur.
                # Mutations are sampled from a uniform distribution between min
                # and max.
                self._gene_mutation_ranges[trait] = None
                # Set the maximum number of decimals for an allele.
                self._gene_mutation_decimals[trait] = None
    
            # POLYGENIC TRAIT        
            # Set the trait name.
            trait = "disgust_polygenic"
            # Set the trait to continuous error rather than discrete within a
            # predefined range.
            self._gene_mutation_discrete[trait] = GENE_MUTATION_DISCRETE
            # Set the number of alleles that make up the polygenic trait.
            m = 10
            # Create an Nx2 matrix to store genotype in. The second dimension
            # separates the mother's (n,0) and father's (n,1) contribution.
            self._genotype[trait] = numpy.zeros((n,2,m), dtype=numpy.float16)
            # Set the rule for how phenotypes come to be from this gene.
            self._gene_rules[trait] = "average"
            # Continuous alleles, sampled from normal distribution.
            if ALLELE_RANDOM_INIT:
                allele_m = (LOW_ALLELE + HIGH_ALLELE) / 2.0
                self._genotype[trait][:,0,:] = allele_m + \
                    numpy.random.randn(n,m) * INITIAL_GENOTYPE_SD
                self._genotype[trait][:,1,:] = allele_m + \
                    numpy.random.randn(n,m) * INITIAL_GENOTYPE_SD
                # Remove all <0 and >1 occurrences.
                self._genotype[trait][self._genotype[trait]<0] = 0.0
                self._genotype[trait][self._genotype[trait]>1] = 1.0
            # Discrete alleles, sampled from pre-defined high and low.
            else:
                # Randomly assign the high and low value (these are the genotypes),
                # and set the probability of each occurring.
                alleles = [LOW_ALLELE, HIGH_ALLELE]
                p_alleles = [0.5, 0.5]
                # Randomly set mother's contribution.
                self._genotype[trait][:,0,:] = numpy.random.choice(alleles, \
                    size=(n,m), replace=True, p=p_alleles)
                # Randomly set father's contribution.
                self._genotype[trait][:,1,:] = numpy.random.choice(alleles, \
                    size=(n,m), replace=True, p=p_alleles)
    
            # Set the gene mutation rate.
            self._gene_mutation_rates[trait] = GENE_MUTATION_RATE
            # Set the gene mutation error SD.
            self._gene_mutation_error_sd[trait] = GENE_MUTATION_ERROR_SD
            # Set the (min,max) range in which mutations can randomly occur.
            # Mutations are sampled from a uniform distribution between min
            # and max.
            self._gene_mutation_ranges[trait] = (0.0, 1.0)
            # Set the maximum number of decimals for this allele.
            self._gene_mutation_decimals[trait] = None
    
    
        def init_nutrition(self, n, is_male, culture, phenotype):
            
            """This function is called for newborns, passing the newborns' cultural
            (`culture` variable) and genetic (`phenotype` variable) traits. It sets
            the energy intake for individuals. We overwrite the function here to
            couple disgust avoidance and energy intake.
            """
            
            # Don't take more than the necessary nutrition.
            nutrition = numpy.ones(n, dtype=numpy.int32)
            nutrition[is_male] = min(self._nutritional_intake_male, \
                self._environmental_nutrition)
            nutrition[is_male==False] = min(self._nutritional_intake_female, \
                self._environmental_nutrition)
    
            # Compute the average disgust avoidance, given both culture and
            # genetics.
            p = numpy.zeros(n, dtype=numpy.float64)
            i = 0
            # Compute the culture*evironment interaction, provided it is set.
            if "disgust" in culture.keys():
                p = p + culture["disgust"]
                i += 1
            # Compute the genetics*environment interaction. Note that this is done
            # on the basis of the phenotype, which in turn is determined by
            # genetic make-up.
            if "disgust_monogenic_highest" in phenotype.keys():
                p = p + phenotype["disgust_monogenic_highest"]
                i += 1
            if "disgust_monogenic_lowest" in phenotype.keys():
                p = p + phenotype["disgust_monogenic_lowest"]
                i += 1
            if "disgust_monogenic_average" in phenotype.keys():
                p = p + phenotype["disgust_monogenic_average"]
                i += 1
            if "disgust_polygenic" in phenotype.keys():
                p = p + phenotype["disgust_polygenic"]
                i += 1
            
            # If there is a disgust trait, compute its impact on energy intake.
            if i > 0:
                # Compute the average disgust trait. Note that this is the 
                # proportion of disgusting food that an individual would still eat.
                p = p / float(i)
                # We would magically create extra food with p>1, so we should
                # limit those values. Similarly, negative food does not exist, so
                # we should limit p<0 values too.
                p[p<0] = 0.0
                p[p>1] = 1.0
                # Compute the amount of contaminated environmental nutrition.
                contaminated = int(round(self._environmental_nutrition * \
                    P_ENVIRONMENT_CONTAMINATION, 0))
                contaminated_consumed = contaminated * p
                uncontaminated = self._environmental_nutrition - contaminated
                # The energy intake is the consumed contaminated food plus the
                # uncontaminated food.
                energy_intake = contaminated_consumed + uncontaminated
                # Individuals don't take more than they need.
                too_high_male = is_male & \
                    (energy_intake > self._nutritional_intake_male)
                too_high_female = (is_male==False) & \
                    (energy_intake > self._nutritional_intake_female)
                energy_intake[too_high_male] = self._nutritional_intake_male
                energy_intake[too_high_female] = self._nutritional_intake_female
    
            # Without any impact on the energy intake, it's simply the 
            # environmental nutrition or the maximum amount of nutrition needed.
            else:
                energy_intake = numpy.ones(n, dtype=numpy.int32)
                energy_intake[is_male] *= min( \
                    self._nutritional_intake_male, self._environmental_nutrition)
                energy_intake[is_male==False] *= min( \
                    self._nutritional_intake_female, self._environmental_nutrition)
            
            return energy_intake
    
    
        def compute_environmental_mortality_probability(self):
            
            """This function describes the interaction between the environment
            and cultural or genetic traits to determine the probability of an
            individual dying because of a specific environmental harm. Note that
            this is counted on top of the existing age-dependent mortality.
            
            The environmental risk increases with early age, and plateaus around
            7-12 years of age, when children in hunter-gatherer societies can
            start helping out with a variety of tasks that include cooking,
            cleaning, carrying water, minding lifestock, and harvesting (Hames & 
            Draper, 2004). Disgust avoidance can help in these tasks, and hence 
            children from these ages accumulate more environmental risk than 
            younger children who are not yet able to engage in behaviours for 
            which disgust avoidance would limit mortality.
            
            REFERENCES
            Hames, R., & Draper, P. (2004). Women's work, child care, and 
                helpers-at-the-nest in a hunter-gatherer society. Human Nature, 
                15(4), p. 319-341. doi:10.1007/s12110-004-1012-x
            """
            
            # Count the population size.
            n = self._specs["age"].shape[0]
            
            # Shortcut if the external probability is 0.
            if self._p_environmental_mortality_per_kJ == 0:
                return numpy.zeros(n, dtype=numpy.float64)
            
            # The baseline disgust-related mortality is set according to age,
            # under the assumption that very young children do not die because
            # of a lack of disgust avoidance, because they are minded by their
            # parents. Instead, we use a cumulative normal distribution with a
            # mean and standard deviation set so that children between 7 and 12
            # are increasingly likely to face the consequences of a lack of
            # disgust avoidance.
            h = scipy.stats.norm.cdf(self._specs["age"], loc=5, scale=3)
            
            # Compute the average disgust avoidance, given both culture and
            # genetics.
            p = numpy.zeros(n, dtype=numpy.float64)
            i = 0
            # Compute the culture*evironment interaction, provided it is set.
            if "disgust" in self._culture.keys():
                p = p + self._culture["disgust"]
                i += 1
            
            # Compute the genetics*environment interaction. Note that this is done
            # on the basis of the phenotype, which in turn is determined by
            # genetic make-up.
            if "disgust_monogenic_highest" in self._phenotype.keys():
                p = p + self._phenotype["disgust_monogenic_highest"]
                i += 1
            if "disgust_monogenic_lowest" in self._phenotype.keys():
                p = p + self._phenotype["disgust_monogenic_lowest"]
                i += 1
            if "disgust_monogenic_average" in self._phenotype.keys():
                p = p + self._phenotype["disgust_monogenic_average"]
                i += 1
            if "disgust_polygenic" in self._phenotype.keys():
                p = p + self._phenotype["disgust_polygenic"]
                i += 1
                    
            # Compute the average disgust avoidance. (NOTE: This is actually 
            # 1-avoidance, as it's the probability of still eating "disgusting"
            # contaminated food.)
            p = p / float(i)
            
            # Compute the yearly number of contaminated kJ for each individual.
            # We do so using the individual consumed energy, which is either the
            # full environmentally available nutrition, or a random subset if more
            # nutrition is available than needed by an individual. The proportion
            # of consumed energy that was contaminated is determined by both the
            # environmental contamination level, and the proportion of this 
            # contaminated nutrition that an individual would still eat (p).
            # The multiplication with 365.25 is to compute the yearly intake, as
            # self._specs["energy_intake"] is the daily intake.
            yearly_contaminated_energy = P_ENVIRONMENT_CONTAMINATION * \
                self._specs["energy_intake"] * 365.25
            # Use disgust avoidance to compute the contaminated intake per
            # individual. This is the contaminated energy * p, because p
            # quantifies the proportion of disgusting food that an individual
            # will still consume.
            yearly_contaminated_energy_intake = p * yearly_contaminated_energy
            # Use the probability per contaminated kJ to compute the annual
            # probability of death.
            p = 1.0 - (1.0 - self._p_environmental_mortality_per_kJ) ** \
                yearly_contaminated_energy_intake
            
            # Compute the environmental mortality.
            h = h * p
            
            return h


        def limit_trait_ranges(self):

            """This function exists to limit the trait values, so that they do
            not surpass 1 or dip under 0. This is to prevent nutrition from 
            magically appearing with >1 trait values, and potential other
            funkiness in the mortality data.
            """

            if "disgust" in self._culture.keys():
                self._culture["disgust"][self._culture["disgust"] < 0] = 0.0
                self._culture["disgust"][self._culture["disgust"] > 1] = 1.0
            
            for name in ["disgust_monogenic_highest", \
                "disgust_monogenic_lowest", "disgust_monogenic_average", \
                "disgust_polygenic"]:
                if name in self._phenotype.keys():
                    # Find those under 0 or over 1.
                    lo = self._genotype[name] < 0
                    hi = self._genotype[name] > 1
                    sel = (numpy.sum(numpy.sum(lo, axis=1), axis=1) > 0) \
                        | (numpy.sum(numpy.sum(hi, axis=1), axis=1) > 0)
                    self._genotype[name][lo] = 0.0
                    self._genotype[name][hi] = 1.0
                    # Recompute phenotype.
                    self._phenotype[name][sel] = self.set_phenotype( \
                        self._genotype[name][sel], self._gene_rules[name])
                    self._phenotype[name][self._phenotype[name] < 0] = 0.0
                    self._phenotype[name][self._phenotype[name] > 1] = 1.0
    
    
    # # # # #
    # SIMULATION
    
    for run in range(n_runs):
    
        # Count the number of existing runs.
        n_existing_runs = len(os.listdir(outdir))
        # Skip if we have enough runs already (only if this option is set).
        if SKIP_EXISTING_RUNS and (n_existing_runs >= n_runs):
            break

        # Construct the name for the output file.
        run_nr_str = str(n_existing_runs+RUN_FILENR_OFFSET+1).rjust(5,"0")
        fname = "run-{}.csv".format(run_nr_str)
        fpath = os.path.join(outdir, fname)
        
        # Initialise a new population.
        pop = DisgustPopulation(n=N, yearly_newborn_probability=BIRTH_PROB_INIT, \
            max_pair_age_diff_at_onset=None, max_pair_age_diff=None, \
            repair_after_partner_death=True, \
            environmental_nutrition=ENVIRONMENTAL_NUTRITION)
        # Set the population's mortality profile, and then reinitialise the ages to
        # match the new profile.
        pop.set_mortality_by_type("hunter-gatherer")
        pop.init_ages()
        
        # If disgust should not be modelled, set the environmental mortality
        # probability to 0.
        if not disgust:
            pop._p_environmental_mortality = 0.0
        
        # Delete the cultural traits that we will NOT be dealing with.
        del_traits = ["disgust"]
        if culture:
            del_traits.remove("disgust")
        for trait in del_traits:
            del(pop._culture[trait])
            del(pop._culture_error_rate[trait])
            del(pop._culture_conformist_bias[trait])
            del(pop._culture_mutation_rates[trait])
            del(pop._culture_mutation_ranges[trait])
        
        # Delete the genetic traits that we will NOT be dealing with.
        del_traits = ["disgust_monogenic_highest", "disgust_monogenic_lowest", \
            "disgust_monogenic_average", "disgust_polygenic"]
        if genes:
            del_traits.remove("disgust_{}".format(gene_type))
        for trait in del_traits:
            del(pop._phenotype[trait])
            del(pop._genotype[trait])
            del(pop._gene_mutation_discrete[trait])
            del(pop._gene_rules[trait])
            del(pop._gene_mutation_rates[trait])
            del(pop._gene_mutation_ranges[trait])
            del(pop._gene_mutation_decimals[trait])
            del(pop._gene_mutation_error_sd[trait])
        
        # Turn mutation off.
        if not mutation:
            for trait in pop._gene_mutation_rates.keys():
                pop._gene_mutation_rates[trait] = 0
            for trait in pop._culture_mutation_rates.keys():
                pop._gene_mutation_rates[trait] = 0
        
        # Create bins to use in the counts.
        age_bins = list(range(0,81,5)) + [100]
        culture_bins = numpy.linspace(BIN_RANGE_CULTURE[0], \
            BIN_RANGE_CULTURE[1], num=N_BINS_CULTURE)
        pheno_bins = numpy.linspace(BIN_RANGE_PHENOTYPE[0], \
            BIN_RANGE_PHENOTYPE[1], num=N_BINS_PHENOTYPE)
        geno_bins = numpy.linspace(BIN_RANGE_GENOTYPE[0], \
            BIN_RANGE_GENOTYPE[1], num=N_BINS_GENOTYPE)
        # Artificially limit the allele varieties by rounding to two decimals.
        # (This might or might not reflect the true genetic situation, as the
        # option for rounding genotype values might or might not be set.)
        allele_varieties = numpy.round(geno_bins[:-1]+numpy.diff(geno_bins)/2.0, 2)
        
        # Write a header to the file.
        with open(fpath, "w") as f:
            # Construct the line.
            header = ["year", "n", "n_male", "p_male", "n_male_fertile", \
                "n_female_fertile", "p_fertile", "n_pairs", "p_pair", \
                "birth_rate", "m_children", "sd_children", \
                "m_children_over_40", "sd_children_over_40", "m_energy", \
                "sd_energy", "min_energy", "max_energy", "m_energy_male", \
                "sd_energy_male", "m_energy_female", "sd_energy_female", 
                "m_age", "sd_age", "m_age_over_5", "sd_age_over_5"]
            for i in range(0, len(age_bins)-1):
                header.append("p_age_{}-{}".format(age_bins[i],age_bins[i+1]))
            # Add culture to the header, if modelled.
            if culture:
                header += ["m_culture", "sd_culture"]
                for i in range(0, culture_bins.shape[0]-1):
                    header.append("pdf_culture_{}-{}".format( \
                        int(round(1000*culture_bins[i],0)), \
                        int(round(1000*culture_bins[i+1],0))))
                for midpoint in ["750", "median"]:
                    for pos in ["over", "under"]:
                        header.append("m_children_culture_{}_{}".format(pos,midpoint))
                        header.append("sd_children_culture_{}_{}".format(pos,midpoint))
                    header.append("ratio_children_culture_under-over_{}".format( \
                        midpoint))
            # Add genes to the header, if modelled.
            if genes:
                # Phenotype.
                header += ["m_pheno", "sd_pheno"]
                for i in range(0, pheno_bins.shape[0]-1):
                    header.append("pdf_pheno_{}-{}".format( \
                        int(round(1000*pheno_bins[i],0)), \
                        int(round(1000*pheno_bins[i+1],0))))
                for pos in ["over", "under"]:
                    header.append("m_children_pheno_{}_750".format(pos))
                    header.append("sd_children_pheno_{}_750".format(pos))
                header.append("ratio_children_pheno_under-over")
                # Genotype.
                header += ["m_geno", "sd_geno"]
                if mutation:
                    for i in range(0, allele_varieties.shape[0]):
                        header.append("p_geno_{}".format( \
                            int(round(100*allele_varieties[i],0))))
                else:
                    header += ["p_high_allele", "p_low_allele"]
                for pos in ["over", "under"]:
                    header.append("m_children_geno_{}_750".format(pos))
                    header.append("sd_children_geno_{}_750".format(pos))
                header.append("ratio_children_geno_under-over")

            # Add male/female specific traits.
            if culture:
                for sex in ["male", "female"]:
                    header += ["{}_m_culture".format(sex), \
                        "{}_sd_culture".format(sex)]
                    for i in range(0, culture_bins.shape[0]-1):
                        header.append("{}_pdf_culture_{}-{}".format(sex, \
                            int(round(1000*culture_bins[i],0)), \
                            int(round(1000*culture_bins[i+1],0))))
            if genes:
                for sex in ["male", "female"]:
                    header += ["{}_m_pheno".format(sex), \
                        "{}_sd_pheno".format(sex)]
                    for i in range(0, culture_bins.shape[0]-1):
                        header.append("{}_pdf_pheno_{}-{}".format(sex, \
                            int(round(1000*pheno_bins[i],0)), \
                            int(round(1000*pheno_bins[i+1],0))))

            # Write the header to file.
            f.write(",".join(map(str, header)))
        
        
        # Loop through all the years.
#        print("Starting simulation {}\nSaving to {}".format(outname,fname))
        t0 = time.time()
        for year in range(1, N_YEARS+1):
            
            # Count the number of individuals in the population in the pervious
            # year.
            n_prev = pop._specs["age"].shape[0]
            # Advance the population by one year.
            population_alive = pop.next_year()
            # Bind the traits' ranges (between 0 and 1).
            pop.limit_trait_ranges()
            # Count the number of individuals in the population in the current
            # year.
            n_next = pop._specs["age"].shape[0]
            # Copy the current birth rate.
            birth_rate = copy.deepcopy(pop._p_newborn_per_year)
            # Adjust the birth rate by 0.2%, depending on whether population is
            # increasing (birth rate should go down) or decreasing (birth rate
            # should go up).
            if (n_next > n_prev) or (n_next > MAX_POPULATION_SIZE):
                pop._p_newborn_per_year -= 0.002
            else:
                pop._p_newborn_per_year += 0.002
            pop._p_newborn_per_year = min(max(BIRTH_PROB_MIN, \
                pop._p_newborn_per_year), BIRTH_PROB_MAX)
            # Stop the simulation if all individuals in the population died.
            if not population_alive:
#                print("\tPopulation went extinct")
                break
            
            # Write to file.
            if year % WRITE_FREQ == 0:
                secs_passed = time.time() - t0
                hh = int(secs_passed // 3600)
                mm = int((secs_passed - (hh*3600)) // 60)
                ss = int(round(secs_passed - (hh*3600 + mm*60), 0))
#                print("\tRun {}/{}, year {}, N={} ({}:{}:{})".format(run+1, \
#                    n_runs, year, n_next, hh, str(mm).rjust(2,"0"), \
#                    str(ss).rjust(2,"0")))
                # COMPUTE
                # Number of individuals in the population.
                n = pop._specs["age"].shape[0]
                # Compute the number of males.
                n_male = numpy.sum(pop._specs["is_male"].astype(int))
                p_male = float(n_male) / float(n)
                # Compute the number of fertile individuals for each sex.
                n_male_fertile = numpy.sum((pop._specs["is_male"] \
                    & pop._specs["is_fertile"]).astype(int))
                is_female = pop._specs["is_male"]==False
                n_female_fertile = numpy.sum((is_female \
                    & pop._specs["is_fertile"]).astype(int))
                p_fertile = float(n_male_fertile+n_female_fertile) / float(n)
                # Compute the number of pairs.
                n_pairs = numpy.sum((pop._specs["pair_nr"]>0).astype(int)) // 2
                p_pair = (n_pairs*2.0) / float(n)
                # Compute the number of children per mother.
                m_children = numpy.mean(pop._specs["n_children"][is_female])
                sd_children = numpy.std(pop._specs["n_children"][is_female])
                # Compute the number of children of mothers over 40.
                is_female_over_40 = is_female & (pop._specs["age"]>40)
                m_children_over_40 = numpy.mean( \
                    pop._specs["n_children"][is_female_over_40])
                sd_children_over_40 = numpy.std( \
                    pop._specs["n_children"][is_female_over_40])
                # Compute the average, SD, min, and max energy of all individuals.
                m_energy = numpy.mean(pop._specs["energy_intake"])
                sd_energy = numpy.std(pop._specs["energy_intake"])
                min_energy = numpy.min(pop._specs["energy_intake"])
                max_energy = numpy.max(pop._specs["energy_intake"])
                # Compute the average and SD for male and female individuals 
                # separately.
                m_energy_male = numpy.mean( \
                    pop._specs["energy_intake"][pop._specs["is_male"]])
                sd_energy_male = numpy.std( \
                    pop._specs["energy_intake"][pop._specs["is_male"]])
                m_energy_female = numpy.mean( \
                    pop._specs["energy_intake"][is_female])
                sd_energy_female = numpy.std( \
                    pop._specs["energy_intake"][is_female])
                # Compute the average age of all individuals.
                m_age = numpy.mean(pop._specs["age"])
                sd_age = numpy.std(pop._specs["age"])
                # Compute the average age of all individuals over 5.
                over_5 = pop._specs["age"][pop._specs["age"]>5]
                m_age_over_5 = numpy.mean(over_5)
                sd_age_over_5 = numpy.std(over_5)
                # Create a histogram of ages, as proportion of population total.
                age_hist, bin_edges = numpy.histogram(pop._specs["age"], \
                    bins=age_bins, density=False)
                age_hist = age_hist / float(n)
                # Compute the average and standard deviation of the cultural trait.
                if culture:
                    # Compute the average and standard deviation of the cultural
                    # trait.
                    m_culture = numpy.mean(pop._culture["disgust"])
                    sd_culture = numpy.std(pop._culture["disgust"])
                    # Compute a histogram of the culture.
                    hist_culture, bin_edges = numpy.histogram( \
                        pop._culture["disgust"], bins=culture_bins, density=True)
                    # Create a Boolean mask to identify those who are or were 
                    # fertile.
                    fert = pop._specs["is_fertile"] | pop._specs["was_fertile"]
                    # Compute the competitive advantage of culture.
                    m_children_culture = {}
                    sd_children_culture = {}
                    p_children_culture = {}
                    for midpoint in ["75", "median"]:
                        m_children_culture[midpoint] = {}
                        sd_children_culture[midpoint] = {}
                        p_children_culture[midpoint] = {}
                        if midpoint == "75":
                            mid = 0.75
                        elif midpoint == "median":
                            mid = numpy.median(pop._culture["disgust"])
                        else:
                            raise Exception("Unknown midpoint {}".format(midpoint))
                        for pos in ["high", "low"]:
                            if pos == "high":
                                sel = pop._culture["disgust"] > mid
                            elif pos == "low":
                                sel = pop._culture["disgust"] < mid
                            else:
                                raise Exception("Unknown pos {}".format(pos))
                            m_children_culture[midpoint][pos] = numpy.mean( \
                                pop._specs["n_children"][fert & sel])
                            sd_children_culture[midpoint][pos] = numpy.std( \
                                pop._specs["n_children"][fert & sel])
                        p_children_culture[midpoint] = \
                            m_children_culture[midpoint]["low"] \
                            / m_children_culture[midpoint]["high"]
                # Compute the average and standard deviation of the g/phenotype.
                if genes:
                    # PHENOTYPE
                    # Compute average and standard deviation of the phenotype.
                    m_pheno = numpy.mean( \
                        pop._phenotype["disgust_{}".format(gene_type)])
                    sd_pheno = numpy.std( \
                        pop._phenotype["disgust_{}".format(gene_type)])
                    # Compute a histogram for the phenotype.
                    hist_pheno, bin_edges = numpy.histogram( \
                        pop._phenotype["disgust_{}".format(gene_type)], \
                        bins=pheno_bins, density=True)
                    # Create a Boolean mask to identify those who are or were
                    # fertile.
                    fert = pop._specs["is_fertile"] | pop._specs["was_fertile"]
                    # Compute the competitive advantage of phenotype.
                    m_children_pheno = {}
                    sd_children_pheno = {}
                    for pos in ["high", "low"]:
                        if pos == "high":
                            sel = pop._phenotype["disgust_{}".format(gene_type)] \
                                > 0.75
                        elif pos == "low":
                            sel = pop._phenotype["disgust_{}".format(gene_type)] \
                                < 0.75
                        else:
                            raise Exception("Unknown pos {}".format(pos))
                        m_children_pheno[pos] = numpy.mean( \
                            pop._specs["n_children"][fert & sel])
                        sd_children_pheno[pos] = numpy.std( \
                            pop._specs["n_children"][fert & sel])
                    p_children_pheno = m_children_pheno["low"] \
                        / m_children_pheno["high"]
                    # GENOTYPE
                    # Compute average and standard deviation of the genotype.
                    m_geno = numpy.mean( \
                        pop._genotype["disgust_{}".format(gene_type)])
                    sd_geno = numpy.std( \
                        pop._genotype["disgust_{}".format(gene_type)])
                    # Compute a histogram for the genotype.
                    if mutation:
                        hist_geno, bin_edges = numpy.histogram( \
                            pop._genotype["disgust_{}".format(gene_type)], \
                            bins=geno_bins, density=False)
                        hist_geno = hist_geno / numpy.sum(hist_geno)
                    # Alternatively, if no mutations occur, it makes more sense
                    # to simply compute the proportions of high and low alleles
                    # because no other values will occur.
                    else:
                        p_high_allele = numpy.sum( \
                            (pop._genotype["disgust_{}".format(gene_type)] \
                            == HIGH_ALLELE).astype(int)) \
                            / float(pop._genotype["disgust_{}".format(gene_type)].size)
                        p_low_allele = numpy.sum( \
                            (pop._genotype["disgust_{}".format(gene_type)] \
                            == LOW_ALLELE).astype(int)) \
                            / float(pop._genotype["disgust_{}".format(gene_type)].size)
                    # Create a Boolean mask to identify those who are or were
                    # fertile.
                    fert = pop._specs["is_fertile"] | pop._specs["was_fertile"]
                    # Compute the competitive advantage of phenotype.
                    m_children_geno = {}
                    sd_children_geno = {}
                    if len(pop._genotype["disgust_{}".format(gene_type)].shape) == 2:
                        gen_values = pop._genotype["disgust_{}".format(gene_type)]
                    else:
                        gen_values = numpy.mean( \
                            pop._genotype["disgust_{}".format(gene_type)], axis=2)
                    for pos in ["high", "low"]:
                        if pos == "high":
                            sel = numpy.mean(gen_values, axis=1) > 0.75
                        elif pos == "low":
                            sel = numpy.mean(gen_values, axis=1) < 0.75
                        else:
                            raise Exception("Unknown pos {}".format(pos))
                        m_children_geno[pos] = numpy.mean( \
                            pop._specs["n_children"][fert & sel])
                        sd_children_geno[pos] = numpy.std( \
                            pop._specs["n_children"][fert & sel])
                    p_children_geno = m_children_geno["low"] \
                        / m_children_geno["high"]
        
                # WRITE TO FILE
                # Open the file to append a line.
                with open(fpath, "a") as f:
                    # Construct the line.
                    line = [year, n, n_male, p_male, n_male_fertile, \
                        n_female_fertile, p_fertile, n_pairs, p_pair, \
                        birth_rate, m_children, sd_children, \
                        m_children_over_40, sd_children_over_40, m_energy, \
                        sd_energy, min_energy, max_energy, m_energy_male, \
                        sd_energy_male, m_energy_female, sd_energy_female, \
                        m_age, sd_age, m_age_over_5, sd_age_over_5]
                    for val in age_hist:
                        line.append(val)
                    # Add culture to the line, if modelled.
                    if culture:
                        line += [m_culture, sd_culture]
                        for val in hist_culture:
                            line.append(val)
                        for midpoint in ["75", "median"]:
                            for pos in ["high", "low"]:
                                line.append(m_children_culture[midpoint][pos])
                                line.append(sd_children_culture[midpoint][pos])
                            line.append(p_children_culture[midpoint])
                    # Add genes to the line, if modelled.
                    if genes:
                        # Phenotypes.
                        line += [m_pheno, sd_pheno]
                        for val in hist_pheno:
                            line.append(val)
                        for pos in ["high", "low"]:
                            line.append(m_children_pheno[pos])
                            line.append(sd_children_pheno[pos])
                        line.append(p_children_pheno)
                        # Genotypes.
                        line += [m_geno, sd_geno]
                        if mutation:
                            for val in hist_geno:
                                line.append(val)
                        else:
                            line += [p_high_allele, p_low_allele]
                        for pos in ["high", "low"]:
                            line.append(m_children_geno[pos])
                            line.append(sd_children_geno[pos])
                        line.append(p_children_geno)

                    # Add male/female specific traits.
                    if culture:
                        for male_sex in [True, False]:
                            sel = pop._specs["is_male"] == male_sex
                            line += [ \
                                numpy.mean(pop._culture["disgust"][sel]), \
                                numpy.std(pop._culture["disgust"][sel])]
                            hist_culture, bin_edges = numpy.histogram( \
                                pop._culture["disgust"][sel], \
                                bins=culture_bins, density=True)
                            for val in hist_culture:
                                line.append(val)
                    if genes:
                        for male_sex in [True, False]:
                            sel = pop._specs["is_male"] == male_sex
                            line += [ \
                                numpy.mean(pop._phenotype[ \
                                "disgust_{}".format(gene_type)][sel]), \
                                numpy.std(pop._phenotype[ \
                                "disgust_{}".format(gene_type)][sel]), \
                                ]
                            hist_pheno, bin_edges = numpy.histogram( \
                                pop._phenotype["disgust_{}".format( \
                                gene_type)][sel], bins=pheno_bins, \
                                density=True)
                            for val in hist_pheno:
                                line.append(val)

                    # Write the line to file.
                    f.write("\n" + ",".join(map(str, line)))

    return 0


# # # # #
# RUN PARALLEL SIMULATIONS

if __name__ == "__main__":
    
    # Check how many cores we have, and subtract one (for the current).
    available_cpus = cpu_count() - 1
    
    # Construct a list of all simulation specs.
    simulations = []
    # Add all environmental conditions.
    for env_danger in ["none", "low", "mid", "high", "higher", "highest"]:
        for env_nutrition in [60, 80, 100, 120, 140]:
            for env_contamination in [0, 2, 5, 10, 20, 50]:
                simulations.append({ \
                    "environment_danger":env_danger, \
                    "environment_nutrition":env_nutrition, \
                    "environment_contamination":env_contamination, \
                    "disgust":True, \
                    "mutation":True, \
                    "genes":True, \
                    "gene_type":"polygenic", \
                    "culture":True, \
                    "culture_restriction":False, \
                    "n_runs":3, \
                    })

#    for env_danger in ["highest"]:
#        for env_nutrition in [60, 80, 100, 120, 140]:
#            for env_contamination in [0, 2, 5, 10, 20, 50]:
#                simulations.append({ \
#                    "environment_danger":env_danger, \
#                    "environment_nutrition":env_nutrition, \
#                    "environment_contamination":env_contamination, \
#                    "disgust":True, \
#                    "mutation":True, \
#                    "genes":True, \
#                    "gene_type":"polygenic", \
#                    "culture":True, \
#                    "culture_restriction":False, \
#                    "n_runs":10, \
#                    })

#    # DEBUG
#    simulations.append({ \
#        "environment_danger":"high", \
#        "environment_nutrition":100, \
#        "environment_contamination":20, \
#        "disgust":True, \
#        "mutation":True, \
#        "genes":True, \
#        "gene_type":"polygenic", \
#        "culture":True, \
#        "culture_restriction":False, \
#        "n_runs":1, \
#        })

#    # Add gene-only runs.
#    simulations.append({ \
#        "environment_danger":"mid", \
#        "environment_nutrition":100, \
#        "environment_contamination":3, \
#        "disgust":True, \
#        "mutation":True, \
#        "genes":True, \
#        "gene_type":"polygenic", \
#        "culture":False, \
#        "culture_restriction":False, \
#        "n_runs":1, \
#        })
#    simulations.append({ \
#        "environment_danger":"mid", \
#        "environment_nutrition":100, \
#        "environment_contamination":3, \
#        "disgust":True, \
#        "mutation":True, \
#        "genes":True, \
#        "gene_type":"monogenic_lowest", \
#        "culture":False, \
#        "culture_restriction":False, \
#        "n_runs":1, \
#        })
#    simulations.append({ \
#        "environment_danger":"mid", \
#        "environment_nutrition":100, \
#        "environment_contamination":3, \
#        "disgust":True, \
#        "mutation":True, \
#        "genes":True, \
#        "gene_type":"monogenic_highest", \
#        "culture":False, \
#        "culture_restriction":False, \
#        "n_runs":1, \
#        })

    # Run through all simulations.
    currently_running_processes = []
    sim_nr = 0
    start_times = {}
    sim_details = {}
    n_sims = len(simulations)
    while len(simulations) > 0 or len(currently_running_processes) > 0:
        
        # Check if any of the processes have finished yet.
        for p in currently_running_processes:
            if not p.is_alive():
                # Compute how long this process ran.
                secs_passed = time.time() - start_times[p.name]
                hh = int(secs_passed // 3600)
                mm = int((secs_passed - (hh*3600)) // 60)
                ss = int(round(secs_passed - (hh*3600 + mm*60), 0))
                time_string = "{}:{}:{}".format(hh, str(mm).rjust(2,"0"), \
                    str(ss).rjust(2,"0"))
                # Report in the terminal.
                print("Process '{}' finished in {}".format( \
                    p.name, time_string))
                # Email home.
                subject = "Simulation '{}' finished ({})".format(p.name, \
                    time_string)
                message = "Simulation details\n"
                keys = list(sim_details[p.name].keys())
                keys.sort()
                for key in keys:
                    message += "{}: {}\n".format(key, \
                        sim_details[p.name][key])
                try:
                    email_message("edwin@dalmaijer.org", subject, message)
                except Exception as e:
                    print("Failed to email! Error:\n\t{}".format(str(e)))
                # Join the process, and remove it from the list and dicts.
                p.join()
                currently_running_processes.remove(p)
                del(start_times[p.name])
                del(sim_details[p.name])

        # Count the number of available CPUs.
        currently_available_cpus = available_cpus - \
            len(currently_running_processes)
        
        # Start a new simulation if a CPU is available.
        if currently_available_cpus > 0 and len(simulations) > 0:
            # Remove the next simulation from the list.
            sim  = simulations.pop(0)
            sim_nr += 1
            # Create a new Process for the next simulation.
            p = Process( \
                target=run_simulations, \
                args = [sim["environment_danger"], \
                    sim["environment_nutrition"], \
                    sim["environment_contamination"], sim["disgust"], \
                    sim["mutation"], sim["genes"], sim["gene_type"], \
                    sim["culture"], sim["culture_restriction"], \
                    sim["n_runs"]], \
                )
            p.name = "simulation_{}/{}".format(sim_nr, n_sims)
            p.daemon = True
            # Start the process.
            p.start()
            start_times[p.name] = time.time()
            sim_details[p.name] = sim
            currently_running_processes.append(p)
            print("Started process '{}'".format(p.name))
            
        time.sleep(2.0)
