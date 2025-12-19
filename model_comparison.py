# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import math
import random

from pyfaidx import Fasta
from json import dump
from numpy.random import RandomState
from pybedtools import BedTool
from collections import defaultdict
from scipy.stats import norm
from scipy.stats import mannwhitneyu
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from TFFM_run_scan_batch import occupancy_scanning
from TFFM_run_scan_batch import thermodynamic_scanning
from TFFM_run_scan_batch import tffm_scanning
from TFFM_run_scan_batch import parse_fasta_header
from TFFM_permutations_batch import permute_sequence
from TFFM_Scanning_Final_Model_27 import thermodynamic_scanning_two_state

this_dir = os.path.dirname(os.path.abspath(__file__)) 

roc_comparison = os.path.abspath(os.path.join(this_dir, "../roc_comparison"))
tffm_source = os.path.abspath(os.path.join(this_dir, "../TFFM"))

sys.path.append(tffm_source) 
sys.path.append(roc_comparison) 

from tffm_module import tffm_from_xml
from constants import TFFM_KIND # TFFM_Kind defines the three kinds of TFFMs (1st-order, detailed, 0-order)
from compare_auc_delong_xu import delong_roc_variance
from compare_auc_delong_xu import delong_roc_test
from compare_auc_delong_xu import compute_ground_truth_statistics
from compare_auc_delong_xu import fastDeLong

# __ Step 4: Prepare FASTA file of High-Stringency HIF-1 Binding Sites from Chip-Seq ___________________

def ChipSeq_Fasta(
        chip_seq_path, 
        genome_file, 
        foreground_path_name="ChipSeq_Foreground.fa", 
        background_path_name="ChipSeq_Background.fa", 
        temp_path_name="temp.bed",
        window_size = None, 
        random_seed=None, 
        foreground_bed_override=None,
        negative_method = "dinucleotide shuffling",
        neg_bed_path=None,
        use_bed_file=True, bed_columns=None, num_peaks=1000):
    """
    Extracts genomic sequences from ChIP-Seq peak coordinates and generates two FASTA files using:

        1) Foreground Set: Sequences corresponding to peak regions with a standardized sequence length
        2) Background Set: Sequences corresponding to dinucleotide-shuffled versions of foreground sequences

    This function supports both:

        - Standard proccessed BED input files from Chip-Seq analysis (BED6+4 format; only significant peaks are considered for these files)
        - Tabular CSV files from Chip-Seq analysis pf stringent sites (Must include Chrom, Start, and End columns; all given peaks are considered for these files)

    Assumes all peaks are on the positive strand and the genome reference file is globally defined. 
    The assumption of the strand peak is irrelevant as an equal length around the center is extracted in both directions and model scanning occurs on both strands.

    ----------
    Description:
    ----------

    This function takes peak coordinates from a table of stringent transcription factor binding sites or a BEDFILE of normalized proccessed peaks derived from ChIP-Seq, and generates a foreground (centered on peak) and background set of uniform length by:
        1) Normalizing size of extracted sequences using the maximum window length across all entries or using a provided fixed window length. This prevents biases during motif scanning, which can be sensitive to varying sequence lengths.
        2) Converting ChIP-Seq coordinates to a six-column BED-formatted entry, then using getfasta to retrieve corresponding DNA sequences from a reference genome file (hg19 or hg38 in this case). This generates the foreground FASTA set.
        3) Shuffling (sampling without replacement) each foreground sequence using a dinucleotide-preserving algorithm. This shuffling maintains same local structure and GC content while disrupting true motif occurrence. This generates the background FASTA set (negative control)
    
    ----------
    Bed File Format:
    ----------

    A six-column BED file has the following format for its columns:
        Column   | Name         | Description
        ---------|--------------|------------------------------------------------------
        1        | chrom        | Chromosome name (e.g., "chr1", "chrX", etc.)
        2        | chromStart   | Start coordinate on the chromosome for the sequence considered (0-based, inclusive)
        3        | chromEnd     | End coordinate on the chromosome for the sequence considered (1-based, exclusive)
        4        | name         | Name of the line in the BED file (often nearest gene name)
        5        | score        | Score between 0 and 100 (optional)
        6        | strand       | Strand orientation (positive ["+"] or negative ["-"] or "." if no strand)
    
    In a BED(6+4) format - There are additional columns for: 

        - signalValue (float);  "Measurement of average enrichment for the region"
        - pValue (float);       "Statistical significance of signal value (-log10). Set to -1 if not used."
        - qValue (float);       "Statistical significance with multiple-test correction applied (FDR -log10). Set to -1 if not used."
        - peak (int);           "Point-source called for this peak; 0-based offset from chromStart. Set to -1 if no point-source called."

    ----------
    Args:

        chip_seq_path (str): Path to the ChIP-Seq input file (BED or CSV). Must contain at least 3 columns: chromosome, start, end.

        foreground_path_name (str): Filename for the output FASTA of foreground (true ChIP-Seq) sequences

        background_path_name (str): Filename for the output FASTA of background (shuffled) sequences

        window_size (int, optional): Length of the sequence window to extract around each peak center (default: max peak width in input)

        random_seed (int, optional): Seed for reproducible permutations of background sequences

        use_bed_file (bool, default=True): Whether to treat the input file as a BED-formatted file. If False, assumes a raw tabular CSV file

        num_peaks (int, default=2000): Number of peaks to sample from the input file if it contains more than 2000 rows. This is used to limit the number of peaks in the foreground FASTA file.

        bed_columns (list of str, optional): Custom column names for BED file input. If None, uses a standard BED6+4 column set:
            ["Chromosome", "Start", "End", "Name", "Score", "Strand", "Signal Value", "p-Value", "Peak"]

        negative_method (bool): The method to use for generating negative samples (dinucleotide shuffling or genome matched).
            If genome matched is used, a separate bed file containing the genome-matched negative samples must be provided.

        neg_bed_path = The path to the bed file for the genome-matched negative samples (required if negative_method == "genome matched")

    ----------
    Returns:

        tuple(str, str): Paths to generated FASTA files
            1) ChipSeq_Fasta_Foreground_Path (path): The path to the foreground FASTA file 
            2) ChipSeq_Fasta_Background_Path (path): The path to the background FASTA file 

    """

    # Check if the neg_bed_path is provided if genome matched method is used
    if negative_method == "genome matched":
        if neg_bed_path is None:
            raise ValueError("Negative method 'genome matched' requires a path to the negative BED file (neg_bed_path).")

    # Initialize a new random number generator if one is not provided.
    if random_seed is None:
        rng = RandomState()

    # Initialize the random number generator using the provided seed 
    elif isinstance(random_seed, int):
        rng = RandomState(random_seed)

    # Initialize a list to hold promoter entries 
    foreground_sequences_list = []
    background_sequences_list = []

    # Define output file path for FASTAs
    ChipSeq_Foreground_Path = os.path.join(plots_dir, foreground_path_name)
    ChipSeq_Background_Path = os.path.join(plots_dir, background_path_name)
    
    # ------------------ Step 1: Generate FASTA Sequences for Foreground ---------------------

    if use_bed_file:

        if bed_columns is None:

            # Set the bed column names as standard format for Chip-Seq proccessed format (BED6+4)
            column_names = [
                "Chromosome", 
                "Start", 
                "End", 
                "Name", 
                "Score", 
                "Strand",
                "Signal Value", 
                "-log10(pvalue)",
                "-log10(qvalue)",
                "Peak"
            ]
        
        else:
            
            # Set the column names based on the provided name list
            column_names = bed_columns

        # Load the Chip-Seq Bedfile as a Dataframe 
        input_bed_file = pd.read_csv(
            chip_seq_path, 
            sep="\t",                # Use each tab in file as the delimiter
            header=None,             # Prevent pandas from interpreting any row as header
            names=column_names)

        # Filter the dataframe to include only significant peaks
        ChipSeq_peaks = input_bed_file[input_bed_file["-log10(qvalue)"] > 2]

        # Filter the dataframe to include a random number significant peaks based on user input
        if len(ChipSeq_peaks.index) > num_peaks:
            ChipSeq_peaks = ChipSeq_peaks.sample(num_peaks, random_state=random_seed, replace=False)

    else:

        # Load the Chip-Seq rows as a pandas DataFrame (each row corresponds to a single binding site)
        ChipSeq_peaks = pd.read_csv(chip_seq_path)

    # Clean up columns in Chip-Seq pandas dataframe (remove whitespaces and lowercase all strings)
    ChipSeq_peaks.columns = [col.strip().lower() for col in ChipSeq_peaks.columns]

    # Raise ValueError if required columns do not exist in pandas dataframe
    required_cols = ['chromosome', 'start', 'end']
    for col in required_cols:
        if col not in ChipSeq_peaks.columns:
            raise ValueError("Missing required column: %s in Chip-Seq file" % col)
            
    if window_size is None:

        # Compute the range of the all binding sites (the df"end" - df"start" returns a pandas series object of numerical differences for each row; .abs() is applied on the series object as a precaution for incorrect strand orientation)
        all_window_sizes = (ChipSeq_peaks["end"] - ChipSeq_peaks["start"]).abs()

        # Compute the maximum binding site range (.max() is applied on the pandas series object of numerical differences)
        window_size = all_window_sizes.max()
    
    else:

        # Convert input to integer
        window_size = int(window_size)

    # Load the genome FASTA file once before iterating
    genome = Fasta(genome_file)

    # Iterate through each row of the Chip-Seq dataframe (see note below on .iterrows())
    for i, row in ChipSeq_peaks.iterrows():

        # Extract the relevant values for the specific row from the corresponding column in the dataframe
        chrom = str(row["chromosome"])
        start = int(row["start"])
        end = int(row["end"])

        # Define the length for each chromosome in genome
        chrom_length = len(genome[chrom])

        # Compute the size of half the window for identification of new start and end coordinates (result is an integer; floor division of an integer gives an integer)
        half_window = window_size // 2

        if use_bed_file is True:
            
            # Define header and Peak value (Point-source called for the peak)
            header = i
            Peak = int(row["peak"])

            # Define the point-source as the center
            center = Peak + start

        else:
            
            # Use "nearest gene locus" if available, otherwise fallback to row index
            if "nearest gene locus" in row.index:
                header = str(row["nearest gene locus"])
            else:
                header = str(i)  

            # Compute the center of the region in which the Chip-Seq peak was found using floor division (result is casted as an integer value)
            center = ((start + end) // 2)

        # Compute the start and end of the region that will be extracted based on maximum window size (all coordinates are casted as integer strings since this is required by bedtools)
        new_start = str(int(max(center - half_window, 0))) # max() is used to prevent negative coordinates 
        new_end = str(int(min(chrom_length, center + half_window))) # min() is used to prevent coordinates beyond chromosome length

        # Skip any binding site with malformed chromosome coordinates
        if new_end <= new_start:
            print("Skipping %s due to malformed BED interval: start=%s, end=%s" % (header, new_start, new_end))
            continue

        # Append values for the specific row to promoter_list in BED file format (assume all binding sites are on positive strand)
        foreground_sequences_list.append((chrom, new_start, new_end, header, ".", "+"))

    # Determine whether to use re-sampled bed file or an existing file
    if foreground_bed_override and os.path.isfile(foreground_bed_override):
        # Reuse the exact same sampled positives if it already exists
        bed_path = foreground_bed_override
    else:
        # Convert the promoter list to BED file object and sort (the default sort() method sorts a BED file in the following order: (i) Chromosome: Lexicographically (chr1, chr10, chr2, etc.), (ii) Start Position: Numerically in ascending order, (iii) End Position: Numerically in ascending order (if start positions are identical)
        foreground_bed_file = BedTool(foreground_sequences_list).sort()

        # Save the BED file to the output directory
        bed_path = os.path.join(plots_dir, temp_path_name)
        foreground_bed_file.saveas(bed_path)

    # Create a FASTA file using the provided BED file (the getfasta function works only in shell environment; the os.system function allows execution of shell commands directly from the script)
    os.system('bedtools getfasta -fi "%s" -bed "%s" -name -fo "%s"' % (genome_file, bed_path, ChipSeq_Foreground_Path))

    if not os.path.isfile(ChipSeq_Foreground_Path):
        raise IOError("FASTA file was not created; bedtools may have failed.")
    
    # ------------------ Step 2: Filter FASTA produced by bedtools ---------------------

    # Load the foreground FASTA that bedtools just created
    foreground_records_raw = list(SeqIO.parse(ChipSeq_Foreground_Path, "fasta"))

    # Filter the sequences with only ambigous bases (by creating set which retains all unique characters and filtering sequences with only the "N" character)
    Foreground_SeqRecord = [rec for rec in foreground_records_raw if set(str(rec.seq.upper())) != {"N"}]

    # Over-write the FASTA produced by bedtools (on disk) with the filtered records
    SeqIO.write(Foreground_SeqRecord, ChipSeq_Foreground_Path, "fasta")

    # # ------------------ Step 3: Generate FASTA Sequences for Background ---------------------
 
    if negative_method == "dinucleotide shuffling":

        # Create a iterator of SeqRecord objects for each fasta entry in the fasta file and convert it to a list of SeqRecord objects
        ChipSeq_SeqRecord = list(SeqIO.parse(ChipSeq_Foreground_Path, "fasta"))

        # Iterate over each promoter sequence in the foreground to generate shuffled background sequences (i = index; seq = SeqRecord object)
        for i, seq in enumerate(ChipSeq_SeqRecord):

            # Extract the DNA sequence from the SeqRecord (as a Biopython Seq object)
            sequence = seq.seq

            # Convert the sequence to string and perform di-nucleotide-preserving permutation
            permuted_str = permute_sequence(seqstr=str(sequence), rng=rng)

            # Create a new SeqRecord for the shuffled sequence with a unique identifier
            background_seq_record = SeqRecord(
                Seq(permuted_str),                            # Convert string back to Seq object
                id="shuffled_seq_%d" % i,                     # FASTA header
                description="dinuc_shuffled"
            )

            # Append the background sequence to the list
            background_sequences_list.append(background_seq_record)

        # Write the list of shuffled sequences to a FASTA file
        SeqIO.write(background_sequences_list, ChipSeq_Background_Path, "fasta")

    elif negative_method == "genome matched":

        # Convert bedfile to dataframe
        neg_bed_df = pd.read_csv(neg_bed_path, sep="\t", header=None, names=["chrom", "start", "end", "name", "score"])

        # Retrieve a series for length of each interval
        lens = neg_bed_df["end"].astype(int) - neg_bed_df["start"].astype(int)

        # Ensure all values in length series are equal to window length
        if not (lens == int(window_size)).all():
            bad = (lens != int(window_size)).sum() 
            raise ValueError("Negative intervals not equal to window_size (bad rows: %d)." % bad)

        # Ensure at least 4 columns in BED file
        if neg_bed_df.shape[1] < 4:
            raise ValueError("Negative BED must have at least 4 columns (chrom, start, end, name).")

        # Randomly sample foreground sequences to match number of negatives
        if neg_bed_df.shape[0] < num_peaks:

            # Determine number of peaks for negative sample
            neg_peaks = int(neg_bed_df.shape[0])

            Foreground_SeqRecord_Filtered = random.sample(Foreground_SeqRecord, neg_peaks)

            # Over-write the current FASTA  with the filtered records
            SeqIO.write(Foreground_SeqRecord_Filtered, ChipSeq_Foreground_Path, "fasta")

            print("Number of Positives Adjusted to Match Negatives for TF: %s. New Total Samples: %s" % (tf, neg_peaks))

        # Randomly sample negative BED to match number of peaks 
        if neg_bed_df.shape[0] > num_peaks:
            neg_bed_df = neg_bed_df.sample(num_peaks, random_state=random_seed, replace=False)

            # Create a BedTool object from the sampled dataframe
            neg_bt = BedTool.from_dataframe(neg_bed_df[["chrom","start","end"]]).sort()

            # Save the new BED file to the output directory
            neg_bed_path = os.path.join(plots_dir, "sampled_%s_neg_bed_temp.bed" % tf)
            neg_bt.saveas(neg_bed_path)

        # Create a FASTA file using the provided BED file 
        os.system('bedtools getfasta -fi "%s" -bed "%s" -name -fo "%s"' % (genome_file, neg_bed_path, ChipSeq_Background_Path))

    else:

        raise ValueError("Invalid negative_method: %s. Choose 'dinucleotide shuffling' or 'genome matched'." % negative_method)

    print ("Foreground and Background Sequences saved to %s and %s (Negative set method: %s)" % (ChipSeq_Foreground_Path, ChipSeq_Background_Path, negative_method))
    
    return ChipSeq_Foreground_Path, ChipSeq_Background_Path

# __ Step 1: Examine and Visualize Paramater Space for Stringent Binding Sites ___________________

def parameter_space(foreground_fasta_path, background_fasta_path, xml_path, gene="HIG2", variable="cluster scores", label_name="Local Cluster", has_strand=False):
    """
    Computes a 3D thermodynamic landscape for a list of high-stringency foreground sites against background sites by scanning over a grid 
    of (λ, S₀) parameters and visualizing the resulting site probabilities or local cluster scores.

    Args:
        foreground_fasta_path (str): Path to foreground FASTA file (normalized sequences centered on ChIP-Seq HIF-1A peaks).
       
        background_fasta_path (str): Path to background FASTA file (shuffled or matched control sequences).

        xml_path (str): Path to trained TFFM model XML.

        variable (str): Score to be evaluated ("site probabilities" or "cluster scores")

        gene (str): A specific gene in the foreground fasta file that will be used for generating the 3D landscape

        label_name (str): Label for max variable score (z-axis) in 3D landscape for each paramater combination (x- and y-axis)
      
    Returns:
        None. Saves 3D surface and heatmap to output directory.

    -----------------
    Note:
    -----------------
    This function defaults to HIG2 (hypoxia inducible gene 2) and extracts a 1300 bp region surrounding all Chip-Seq Peaks. 
    """

    # ------------------ Step 1: Perform Scanning over Paramater Space ---------------------
        
    # Define float ranges (grid) of each parameter (hard-coded for now)
    lambda_values = np.arange(0.4, 3.0 + 0.01, 0.2)   # 0.0 to 3.0 inclusive
    s0_values = np.arange(1.0, 7.5 + 0.01, 0.5)       # 0.0 to 7.5 inclusive

    def run_scanning(fasta_path, label):
        """
        Internal Helper: Runs Scanning over parameter space
        
        """

        # Parse the extracted FASTA and retrieve a SeqRecord object
        promoter_seqs = list(SeqIO.parse(fasta_path, "fasta"))

        # Define a container to hold results for each promoter and each (lambda, S0) pair
        probability_surface = []

        # Iterate through all lambda values (outer loop)
        for lam in lambda_values:

            # Iterate through all S0 values (inner loop)
            for s0 in s0_values:

                print (u"Iteration lambda: %s and S₀: %s for variable %s" % (lam, s0, variable))

                # Run thermodynamic scoring
                _, _, summary = thermodynamic_scanning(
                    promoter_seqs=promoter_seqs,
                    xml_path=xml_path,
                    lam=lam,
                    S0=s0
                )

                # Pair the parameter values with results from each promoter (summary is a list of dictionaries)
                for i, promoter_result in enumerate(summary):

                    # Extract relevant value - Each summary dictionary contains list of all values for sequence (see note below on .get())
                    relevant_list = promoter_result.get(variable, [])

                    # Skip promoter entry if it does not contain relevant result
                    if not relevant_list:
                        print("Empty relevant_list for promoter #%d (%s)" % (i, promoter_seqs[i].id))
                        continue

                    # Compute the maximum of the list of scores  
                    relevant_value = max(relevant_list)
                    
                    # Retrieve header from list of SeqRecord (by indexing ith promoter)
                    header = promoter_seqs[i].id

                    # Retrieve gene name from SeqRecord header (if none exists, keep header as reference)
                    gene_id, _, _, _, _ = parse_fasta_header(header, has_strand=has_strand)
                    if gene_id is None:
                        gene_id = header
                    
                    # Append full dictionary of result
                    probability_surface.append({
                        "lambda": lam,
                        "S0": s0,
                        "promoter": gene_id,
                        "value": relevant_value,
                        "group": label
                    })

        return probability_surface

    # Run scanning for both foreground and background
    fg_results = run_scanning(foreground_fasta_path, "foreground")
    bg_results = run_scanning(background_fasta_path, "background")

    # Combine results into a single list
    combined_results = fg_results + bg_results

    # ------------------ Step 2: Create a 3D Plot Matrix for a Specific Gene ------------------------
    
    # Define a matrix of shape (14, 14) to hold maximum scores
    scores = np.zeros((len(lambda_values), len(s0_values)))

    # Track whether any matching entry was found
    match_found = False

    # Iterate through list (of dictionaries) of foreground results (to populate 3D plot matrix for specified gene in foreground)
    for entry in fg_results:

        # Filter for dictionaries containing specified gene 
        if entry["promoter"] == gene:

            # Index the grid for lambda and S₀ to identify row (i) and column (j) position in the 2D matrix (scores[i,j]) that corresponds to a given pair of the parameters (abs(lambda_values - entry["lambda"]) creates a new array where each lambda_value is subtracted from the entry value and we take the absolute value; argmin() returns the index in the array with the lowest value - This essentially finds the index i of the value in lambda_values that equals (or is closest) to entry["lambda"] which should correspond to when lambda_values=entry["lambda"])
            i = np.argmin(abs(lambda_values - entry["lambda"]))
            j = np.argmin(abs(s0_values - entry["S0"]))

            # Save the value to the score matrix
            scores[i, j] = entry["value"]

            match_found = True

    if not match_found:
        print("Warning: No entries found for gene '%s' in foreground results." % gene)
        
    # Define a meshgrid for plotting (a meshgrid creates coordinate matrices based on vectors [parameters])
    lambda_grid, S0_grid = np.meshgrid(lambda_values, s0_values)

    # Initialize a figure for 3D plot
    fig = plt.figure(figsize=(10, 7))

    # Create a subplot with a 3D projection
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot coordinate matrices and scores onto subplot
    surf = ax.plot_surface(lambda_grid, S0_grid, scores, cmap='viridis', edgecolor='k')
    
    # Define axes labels and colorbar
    ax.set_xlabel(u"λ (Sigmoid slope)")
    ax.set_ylabel(u"S₀ (Cutoff midpoint)")
    ax.set_zlabel("Max %s Score" % label_name)
    ax.set_title("Thermodynamic Parameter Space for %s" % gene)
    fig.colorbar(surf, shrink=0.5, aspect=8)

    # Save 3D plot
    surface_path = os.path.join(plots_dir, "%s_%s_3D_Thermo_Surface.png" % (gene, variable))
    plt.savefig(surface_path)
    plt.close()
    
    # ------------------ Step 4: Round Float Values in Dataframe -----------------------
    
    # Convert list of results into a DataFrame
    df = pd.DataFrame.from_records(combined_results)

    # Ensure 'value' column is numeric (convert invalid entries to NaN)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Drop rows where value is NaN (caused by non-numeric entries)
    df = df.dropna(subset=['value'])

    # Convert S₀ and λ float values to one decimal place
    df["lambda"] = df["lambda"].round(1)
    df["S0"] = df["S0"].round(1)
    
    print("The columns are %s" % df.columns)

    # ------------------ Step 4: Compute statistical difference in scores between foreground and background -----------------------

    def pooled_mad(x, y):
        """
        Computes the Pooled Median Absolute Deviation (PMAD) of two samples.

        ----------
        Parameters
        
        x, y (array): Two samples (Foreground and background values)

        ----------
        Returns

        p_mad (float): Pooled MAD of the two samples.

        --------------------------------------
        Background:
        --------------------------------------

        Typically effect size (Cohen's d) normalizes the difference between two group means by their pooled standard deviation
            
            Equation: d = (mean(y) - mean(x)) / s-pooled where s-pooled is the pooled standard deviation.

        However this approach assumes data are approximately normal - and thus the means and variances are robust.
        For non-normal distributions (skewed motif scores), both means and standard deviations can be misleading due to outliers.
    
        --------------------------------------
        Median Absolute Devatition
        --------------------------------------

        For non-parametric distributions, we often use the median instead of the mean and the Median Absolute Deviation (MAD) instead of standard deviation
        
        The Median Absolute Deviation (MAD) is computed as: MAD = median(|X - median(X)|). This provides a measure of variability that uses medians instead of means.
        
            This is robust since it uses a 50% breakdown point (half the data can be extreme outliers and the MAD still works).

        --------------------------------------
        PMAD Derivation
        --------------------------------------
        
        The standard deviations are pooled using the root-mean-square (RMS) formula. Similarly, we can pool MAD values from two groups in the same way

            Equation assuming equal sample size: PMAD_xy = sqrt( (MADx^2 + MADy^2) / 2 )

        The PMAD is used to standardize differences in location (differences in quantiles) in gammap) - See below.
        """

        # Calculate MAD (creates a new array in which all values are shifted by the group median; we can take the median of this array to compute MAD))
        mad_x = np.median(np.abs(x - np.median(x)))
        mad_y = np.median(np.abs(y - np.median(y)))

        # Pool MAD using the same logic as for standard deviations 
        p_mad = np.sqrt((mad_x**2 + mad_y**2) / 2.0)

        return p_mad

    def gamma_p(x, y, p=0.5):
        """
        Compute an effect size (difference between the groups) for a specific quantile (γₚ)

        ----------
        Args:
            
            x, y (array-like): Two samples (foreground, background).
            p (float): Quantile level (between 0 and 1). Default = 0.5 (median).
        
        ----------
        Returns:
        
            gammap (float): Quantile-based robust effect size. NaN if PMADxy = 0.

        -------------------------
        Gammap
        -------------------------
        
        The Cohen's d test measures effect size as a standardized difference in means. However means are not robust to skewed data and outliers.

        To address this, we will use quantiles and MAD. Instead of comparing means, we compare quantiles (Qp) and instead of SD we normalize by PMAD.

            The resulting formula is: γₚ = ( Qp(X) - Qp(Y) ) / PMADxy

            Where:
                Qp(X or Y) = p-th quantile of X and Y
                PMADxy = pooled MAD of X and Y

        -------------------------
        Conceptual Meaning
        -------------------------

        The decomposition of the comparison into quantiles allows us to investigate the difference in distribution (how far apart groups are at different segments (quantiles) in units of pooled MAD. For example
            - p = 0.5 (median) investigates median-based effect size
            - p = 0.25 (lower tail) investigates difference in lower quartiles
            - p = 0.75 (upper tail) investigates difference in upper quartiles

            If γₚ is large at the median but small at the tails - this indicates a strong difference in the center but not the tail
                This allows for a profile of effect sizes over the whole distribution.

        The γₚ can be either positive or negative 
            Positive γₚ: Foreground quantiles (x) are higher than background.
            Negative γₚ: Background quantiles (y) are higher than foreground.

            Note: The sign is dependent on whether the specific group we assign to X and Y respectively 
        
        """

        # Compute p-th quantiles of both groups
        qx = np.quantile(x, p)
        qy = np.quantile(y, p)

        # Compute pooled MAD for scaling
        pmad = pooled_mad(x, y)

        # Compute gammap value (since we're investigating )
        gammap = (qx - qy) / pmad if pmad != 0 else np.nan

        return gammap

    def gamma_profile(x, y, p_low=0.2, p_high=0.8, step=0.05):
        """
        Compute a condensed γₚ profile between two samples.

        This function calculates γₚ for multiple quantiles between [p_low] and [p_high]. Then it summarizes shape differences with min, max, and mean γₚ.

        ----------
        Parameters:
            
            x, y (array-like): Two samples (foreground, background).
            p_low, p_high (float): Quantile range to consider (avoid extremes near 0 and 1)
            step (float): Step size for quantiles

        ----------
        Returns:

            profile : A dictionary containing:
                - gamma_values (a list of all gamma values with their corresponding quantile - {p: gamma_p_value})
                - min gamma (minimum gamma value across quantiles)
                - max gamma (maximum gamma value across quantiles)
                - mean gamma (mean gamma value across quantiles)
        """

        # Create an array float values for all quantiles (from lowest specified quantile to highest specified quantile at specified step value)
        ps = np.arange(p_low, p_high + 0.01, step)

        # Create a list of gammap values for each quantile 
        gammas = [gamma_p(x, y, p) for p in ps]

        # Convert list of gammap values to an array
        gammas = np.array(gammas)

        # Return a dictionary with relevant metrics of gammap values (min, max, mean)
        return {
            'gamma_values': dict(zip(ps, gammas)),
            'min gamma': np.nanmin(gammas),
            'max gamma': np.nanmax(gammas),
            'mean gamma': np.nanmean(gammas)
        }

    def gamma_matrix(df):
        """
        For every (lambda, S0) parameter combination, this function compares the distribution of foreground scores to background scores using γₚ effect sizes across multiple quantiles (0.2 to 0.8) and Mann-Whitney statistic
        
        This is used to evaluate how far apart the foreground scores are from the background scores for a given parameter setting across the quantiles.

        ---------------------------
        Args:
        
        df (pd.DataFrame): A dataframe with the following columns for each promoter evaluated:
            'lambda'  (float): parameter λ for thermodynamic scoring
            'S0'      (float): cutoff midpoint parameter
            'group'   (str)  : "foreground" or "background"
            'value'   (float): score for a single promoter

        ---------------------------
        Returns:
        
        results_pd (pd.DataFrame): A dataframe with one row per (lambda, S0) combination containing:
            - lambda value
            - S0 value
            - p value (p-value derived from U statistic)
            - min gamma (minimum gamma value across quantiles)
            - max gamma (maximum gamma value across quantiles)
            - mean gamma (mean gamma value across quantiles)
            
        """

        # Initialize lists to hold all gamma values and overall results
        gamma_values = []
        results = []

        # Loop over unique λ values in data (outer loop of ascending array of unique λ values; the pd.Series (column of df) is transformed to a NumPy array of unique values by .unqiue())
        for lam in sorted(df['lambda'].unique()):

            # Loop over unique S0 values in data (inner loop of ascending array of unique S₀ values for specific λ and S₀ combination)
            for s0 in sorted(df['S0'].unique()):

                # Subset dataframe to current λ and S₀ (the dataframe is filtered for rows where lambda column equals current lam value and S0 column equals current S0 value; Both conditions ((df['lambda'] == lam) & (df['S0'] == s0)) return a boolean series/mask (True/False) that is combined with logical AND)
                sub_df = df[(df['lambda'] == lam) & (df['S0'] == s0)]

                # Split dataframe into foreground and background score distributions (the dataframe is filtered for rows where group column is either foreground or background; Both conditions (sub_df['group'] == 'foreground' or sub_df['group'] == 'background') returns boolean series/mask (True/False)
                fg = sub_df[sub_df['group'] == 'foreground']
                bg = sub_df[sub_df['group'] == 'background']

                # Convert value column in filtered dataframe to a numpy array
                fg_scores = fg["value"].to_numpy()
                bg_scores = bg["value"].to_numpy()

                # Compute gamma_p profile for quantiles [0.2, 0.8]
                gamma_stats = gamma_profile(fg_scores, bg_scores, p_low=0.2, p_high=0.8, step=0.05)

                # Perform Mann-Whitney U Test
                U, p_value = mannwhitneyu(fg_scores, bg_scores)
                
                # Extract relevant metrics from gamma stats
                min_gamma = gamma_stats['min gamma']
                max_gamma = gamma_stats['max gamma']
                mean_gamma = gamma_stats['mean gamma']
                gamma_list = gamma_stats['gamma_values']
            
                # Store quantile and associated value for combination (.items() returns a view object displaying list of key-value pairs for gamma (formed during zipping) as tuples that we can iterate over
                for q, val in gamma_list.items():
                    gamma_values.append({
                        "lambda": lam,
                        "S0": s0,
                        "quantile": q,
                        "gamma": val
                    })
                
                # Collect results
                results.append({
                    "lambda": lam,
                    "S0": s0,
                    "p value": p_value,
                    "min gamma": min_gamma,
                    "max gamma": max_gamma,
                    "mean gamma": mean_gamma,
                    "foreground_median": np.median(fg_scores),
                    "background_median": np.median(bg_scores)
                })
                
        # Convert final list of dictionaries and gamma results to a dataframe
        results_pd = pd.DataFrame.from_records(results)
        gamma_pd = pd.DataFrame.from_records(gamma_values)

        return results_pd, gamma_pd

    # ------------------ Step 4: Create a BoxPlot Comparing Foreground and Background Across Paramater Space -----------------------

    def faceted_boxplot(data):
        """
        Creates a faceted boxplot: one facet per λ value where the x-axis is S₀, where boxplots show foreground vs background motif score distributions.
     
        This visualizes the interaction between λ and S₀ on motif score separation.

        Note: 
            The seaborn.catplot class returns a high-level FacetGrid object that manages the figure and axes internally. Therefore, there is no need to initialize the plot with plt.figure() beforehand.
            All axes and legends for plot are set by calling appropriate methods directly on the FacetGrid object. The figure size and layout can be controlled directly via height and aspect arguments passed to sns.catplot()
        """
        # Compute gamma values across quantiles 
        test_df, _ = gamma_matrix(df=data)

        # Create the FacetGrid Categorical Plot across Parameter Space 
        facet_boxplot = sns.catplot(
            data=data,
            x="S0",
            y="value",
            hue="group",           # Compare foreground vs background
            col="lambda",          # Facet by lambda
            col_wrap=4,            # Arrange in 4 columns
            kind="box",            # Draw a box-plot for each facet
            height=4,              # Height of each facet
            aspect=1.2,            # Width-to-Height ratio of each facet (1.2 * Height) 
            palette="mako",
            sharey=False           # Allow each facet to scale independently
        )

        # Set the lambda value used in the facet as the title of each subplot (catplot uses str.format() syntax and internally calls .format() method to assign col_value to col_name)
        facet_boxplot.set_titles(u"λ = {col_name}") 

        # Define axes labels and legend
        facet_boxplot.set_axis_labels(u"S₀ (Cutoff midpoint)", "Max Motif Score")
        facet_boxplot._legend.set_title("Group")

        # Flatten the 2D array of AxesSubPlot objects (returned by facet_boxplot.axes) into a 1D list (in order to iterate over all subplots)
        for ax in facet_boxplot.axes.flatten():
            
            # Extract λ value from subplot title and convert to float (splits title into a list of [λ, λ-value] using '=' in title as delimiter, and then extracts λ ([1] from list))
            title = ax.get_title()  
            lam_val = float(title.split('=')[1].strip())

            # Extract tick position and tick labels 
            xticks = ax.get_xticks()

            # Extract tick labels and convert to float S₀ (.get_xticklabels() returns tick labels on x-axis as list of text objects; .get_text() extracts the text content of the text object)
            xticklabels = [float(label.get_text()) for label in ax.get_xticklabels()]

            # Iterate over each tick position on x-axis (xtick) and corresponding S0 value (xticklabel)
            for tick, s0_label in zip(xticks, xticklabels):

                # Extract row in test dataframe that match the specific combination (λ, S₀); Returns a new dataframe with only one row corresponding to combination (using same boolean mask as earlier; each combination only has one specific row in test_df)
                row = test_df[(test_df["lambda"] == lam_val) & (test_df["S0"] == s0_label)]
                if row.empty:
                    continue
                
                # Extract scalar p-value and gamm values from the matching row dataframe (row[_] returns a pandas series with a single value which is converted into a NumPy array with .values() - [0] extracts the first (and ideally only) element from that array)
                p_value = row["p value"].values[0]
                mean_gamma = row["mean gamma"].values[0]

                # For cluster scores - annotate box plot with p-values from Mann-Whitney U-Test (since sigmoid-transformed scores would cause non-linear contribution's doing clustering  - the relative ordering of cluster probabilities will change and thus Mann-Whitney test adds interpretative value)
                if variable == "cluster scores":

                    # Define text annotation as significance value
                    if p_value < 0.001:
                        text = "***"
                        color = 'lightcoral'
                    elif p_value < 0.01:
                        text = "**"
                        color = 'coral'
                    elif p_value < 0.05:
                        text = "*"
                        color = 'seashell'
                    else:
                        text = ""
                        color = 'lightgrey'

                # For site probabilities - annotate box plots with mean gamma value (since the sigmoid transformation is monotonic - the relative ordering of site probabilities will not change thus Mann-Whitney test adds no interpretative value)
                else:
                    
                    # Define text annotation as range of gamma_p values
                    text = "%.1f" % mean_gamma
        
                    # Determine color of text based on mean gamma value (absolute value is taken since mean gamma can be either positive [if group 1 is larger] or negative [if group 2 is larger])
                    abs_gamma = abs(mean_gamma)
                    if abs_gamma < 0.2:
                        color = 'lightgrey'
                    elif abs_gamma < 0.5:
                        color = 'seashell'
                    elif abs_gamma < 0.8:
                        color = 'coral'
                    else:
                        color = 'lightcoral'

                # Find max y value in the current facet for placement of stars and significance line (ax.get_ylim() returns (ymin, ymax) from which we access ymax [1] (top of y-axis))
                y_max = ax.get_ylim()[1]

                # Assign the base-y position for bracket line (starting vertical level slighly below top of the plot)
                base_y = y_max * 0.95    

                # Assign the height of the "bump" for bracket line 
                height = 0.05 * y_max 

                # Determine offset x position between boxes (Foreground and background boxes are side-by-side)
                x1 = tick - 0.2      # foreground (left)
                x2 = tick + 0.2      # background (right)  

                # Draw line (bracket) between foreground and background (corresponding to derived x and y points above boxplots)
                ax.plot(
                    [x1, x1, x2, x2],                                       # x-values of line
                    [base_y, base_y + height, base_y + height, base_y],     # y-values of line (base level determines bottom of bracket; the height + base level determines top of the bracket should be)
                    lw=1.3, 
                    c=color) 
                
                # Add text above bracket
                ax.text(
                        (x1 + x2) / 2,              # x-value of text
                        base_y + height + 0.03,     # y-value of text
                        text,                       # Specified stars or mean gamma value
                        ha='center', 
                        va='bottom',
                        color=color,                # specified color based on mean gamma value
                        fontsize=6, 
                        weight='bold')
                    
            # Iterate over list of x-axis tick labels (text objects) for each subplot 
            for label in ax.get_xticklabels():

                # Rotate the x-axis labels by 45° in each subplot (to prevent overlap in S₀ values)
                label.set_rotation(45)

        # Save faceted plot
        plt.tight_layout()
        catplot_path = os.path.join(plots_dir, "Faceted_BoxPlots_%s.png" % variable)
        plt.savefig(catplot_path)
        plt.close()
    
        # Merge significance test p-values into main dataframe (on= specifies the columns to match between the dataframe for merging; left= specifies to include all rows from left dataframe (data) and matched rows from right (test_df))
        merged = pd.merge(data, test_df, on=["lambda", "S0"], how="left")

        return merged

    overall_df = faceted_boxplot(data=df)
    df_path = os.path.join(plots_dir, "parameter_space_test_results_%s.csv" % variable)
    overall_df.to_csv(df_path)

    def plot_gamma_heatmaps_combined(data):
        """
        Creates a single figure with three subplots (heatmaps) corresponding to quantiles 0.25, 0.5, and 0.75. 
        Each heatmap shows γₚ as a function of λ and S₀.
        """

        # Compute gamma values across quantiles
        _, gamma_df = gamma_matrix(df=data)

        # Round the float values in the quantile column for the gamma dataframe to two decimal places 
        gamma_df['quantile'] = gamma_df['quantile'].round(2)

        # Define quantiles of interest (two decimal point floats)
        quantiles_to_plot = [0.25, 0.50, 0.75]

        # Filter to only quantiles of interest (The isin() method checks whether elements in a Series are contained in a specified set of values and returns a corresponding boolean mask which is used for filtering)
        gamma_filtered = gamma_df[gamma_df['quantile'].isin(quantiles_to_plot)]

        # Compute global min/max for consistent color scaling
        global_vmin = gamma_filtered['gamma'].min()
        global_vmax = gamma_filtered['gamma'].max()

        # Initialize a figure with 1 row and 3 columns on subplot grid (the subplots() functions is a convience wrapper that returns a figure object and an array of axes objects (one for each subplot); it is equivalent to calling plt.figure() and then adding subplots to it)
        fig, axes = plt.subplots(
            1, 3, 
            figsize=(24, 7), 
            sharex=True, 
            sharey=True
            )

        # Iterate through each axis object and the corresponding quantile that will be plotted on the axes
        for ax, q in zip(axes, quantiles_to_plot):

            # Filter data for the specific quantile (using same Boolean mask as earlier)
            sub_df = gamma_filtered[gamma_filtered['quantile'].round(2) == q]

            # Create a pivot table for a heatmap 
            heatmap_data = sub_df.pivot(
                index='S0',                 # Rows = S0
                columns='lambda',           # Columns = lambda
                values='gamma')             # Values = gamma

            # Plot heatmap on the given axis (subplot)
            sns.heatmap(
                heatmap_data,
                cmap="coolwarm",
                center=0,
                vmin=global_vmin, vmax=global_vmax,         # Anchor the colormap across all heatmaps consistently; not anchoring results in each quantile's heatmap having seperate range of colors
                annot=True, fmt=".2f",                      # Write the data value in each cell with two decimal places
                ax=ax,
                cbar=(ax is axes[-1])                       # show colorbar only on the last subplot (returns true on last axes object in array)
            )

            ax.set_title(u"γₚ Heatmap at p = %.2f" % q)
            ax.set_xlabel(u"λ")
            ax.set_ylabel(u"S₀")

        plt.suptitle(u"γₚ Heatmaps Across Quantiles", fontsize=18, y=1.05)
        plt.tight_layout()

        # Save combined heatmap
        combined_path = os.path.join(plots_dir, "Gamma_Heatmaps_Combined_%s.png" % variable)
        plt.savefig(combined_path)
        plt.close()

    plot_gamma_heatmaps_combined(data=df)

    return overall_df

# __ Step 4: Scan FASTA file of Binding Sites derived from Chip-Seq Peaks using all Defined Scanning Models ___________________

def Scan_ChipSeq(foreground_set, background_set, xml_path, tffm_object, tf_name, parse_header_strand=False, window_size=150, lam=1.4, S0=4.0, min_dist=5, tau = 0.5, cluster_constraint=True, negative_sequences=None):
    """
    Perform motif scanning on foreground (ChIP-Seq derived) and background (dinucleotide-shuffled or GC-matched) FASTA sequences.
    
    The scanning is conducted using three independent methods:
        1. Pocc (occupancy-based probability)
        2. Thermodynamic model (Boltzmann-weighted total binding probability)
        3. TFFM (Transcription Factor Flexible Model) Best-Hit

    ----------
    Args:
        foreground_set (str): Path to foreground FASTA file (normalized sequences centered on ChIP-Seq HIF-1A peaks).
        background_set (str): Path to background FASTA file (shuffled or matched control sequences).
        xml_path (str): Path to the XML file containing the detailed TFFM for the transcription factor of interest.
        tffm_object (TFFM): A pre-loaded TFFM object representing the transcription factor binding model.
        tf_name (str): Name of the transcription factor (used for labeling outputs).
        parse_header_strand (bool): Whether to parse strand information from FASTA headers. Default is False.
        window_size (int): Size of the scanning window for thermodynamic model. Default is 150.
        lam (float): Sigmoid slope parameter for thermodynamic model. Default is 1.4.
        S0 (float): Cutoff midpoint parameter for thermodynamic model. Default is 4.0.
        min_dist (int): Minimum distance between binding sites for clustering in thermodynamic model. Default is 5.
        cluster_constraint (bool): Whether to apply clustering constraints in thermodynamic model. Default is True.

    ----------
    Returns:
        models_performance (dict[str, dict[str, any]]): A nested dictionary mapping each parent key (foreground or background string) to their child keys (model name). The model names map to the model results:
            - pocc_values (List[dict]): Concatenated list of occupancy results (foreground + background).
            - binding_values (List[dict]): Concatenated list of thermodynamic results (foreground + background).
            - real_best_hits (List[dict]): Concatenated list of best motif results from TFFM scanning (foreground + background).
    """
    
    # Convert FASTA files to list of SeqRecord objects (SeqIO.parse reads each sequence in a file and creates a iterator of SeqRecord objects for each entry - this is consumed to form a list) 
    Foreground_SeqRecord = list(SeqIO.parse(foreground_set, "fasta"))
    Background_SeqRecord = list(SeqIO.parse(background_set, "fasta"))
    
    # Perform POCC Scanning on foreground and background FASTA files 
    pocc_values_foreground = occupancy_scanning(
        fasta_file=foreground_set, 
        tffm_object=tffm_object
    )
    pocc_values_background = occupancy_scanning(
        fasta_file=background_set, 
        tffm_object=tffm_object
    )
    
    # Perform Thermodynamic Scanning and Two-State Thermodynamic Scanning on foreground and background list of SeqRecords
    binding_values_foreground, two_state_model_foreground, _ = thermodynamic_scanning_two_state(
        promoter_seqs=Foreground_SeqRecord,
        xml_path=xml_path,
        lamR = lam,
        S0=S0,
        cluster_window_size=window_size,
        min_dist=min_dist,
        cluster_constraint=cluster_constraint,
        parse_header_strand=parse_header_strand,
        both_models=True,
        negative_sequences=negative_sequences
    )

    binding_values_background, two_state_model_background, _ = thermodynamic_scanning_two_state(
        promoter_seqs=Background_SeqRecord,
        xml_path=xml_path,
        lamR = lam,
        S0=S0,
        cluster_window_size=window_size,
        min_dist=min_dist,
        cluster_constraint=cluster_constraint,
        parse_header_strand=parse_header_strand,
        both_models=True,
        negative_sequences=negative_sequences
    )
    
    # Perform TFFM hit Scanning on foreground and background list of SeqRecords
    _, real_best_hits_foreground = tffm_scanning(
        promoter_seqs=Foreground_SeqRecord,
        tffm_object=tffm_object,
        parse_header_strand=parse_header_strand
    )
    _, real_best_hits_background = tffm_scanning(
        promoter_seqs=Background_SeqRecord,
        tffm_object=tffm_object,
        parse_header_strand=parse_header_strand
    )
    

    # Define nested dictionary of foreground and background values
    models_performance = {

        "foreground": {
            "probability of occupancy": pocc_values_foreground,
            "maximum cluster score": binding_values_foreground,
            "maximum site probabilities": binding_values_foreground,
            "two state model": two_state_model_foreground,
    #        "two state cluster score": two_state_cluster_scores_foreground,
            "best hits": real_best_hits_foreground
        },

        "background": {
            "probability of occupancy": pocc_values_background,
            "maximum cluster score": binding_values_background,
            "maximum site probabilities": binding_values_background,
            "two state model": two_state_model_background,
    #        "two state cluster score": two_state_cluster_scores_background,
            "best hits": real_best_hits_background,
        },
    }
    
    # Define output path for models_performance nested dictionary
    models_performance_path = os.path.join(plots_dir, "model_performance_scores_%s.json" % tf_name)

    # Save best hits: Open the file at 'model performance scores" in write mode and assign the file handle to 'f'
    with open(models_performance_path, "w") as f:

        # Take the nested dictionary and serialize it into a JSON-formatted string with 2-space indendation and send it into the open file handle (f)
        dump(models_performance, fp=f, indent=2)
    
    return models_performance

# __ Step 6: Evaluate Accuracy of all Models used to Scan Chip-Seq Results ___________________

def ROCAUC_Comparison(model_performance, tf_name, roc_plot_name="Comparative ROC Curve for TFBS Models", delong_results_name="DeLong comparison for Models"):
    """
    Perform ROC-AUC analysis and DeLong's test to evaluate the discriminative performance of the different transcription factor binding site (TFBS) scanning models. 
    This function compares their ability to distinguish true ChIP-Seq-derived binding sites (foreground) from dinucleotide-shuffled sequences (background).

    ----------
    Args:
        model_performance (dict[str, dict[str, any]]): A nested dictionary mapping each parent key (foreground or background string) to their child keys (model name). The model names map to the model results
            - pocc_values (List[dict]): Concatenated list of occupancy results (foreground + background).
            - binding_values (List[dict]): Concatenated list of thermodynamic results (foreground + background).
            - real_best_hits (List[dict]): Concatenated list of best motif results from TFFM scanning (foreground + background).

    Each score list's dict must contain numerical predictions (float) for each sequence (in the dictionary) representing the strength of evidence for TF binding (e.g. occupancy, thermodynamic binding, motif match).

    ----------
    Returns:
        roc_auc_results (dict[str, dict[str, float]]): A nested dictionary mapping each scanning method (parent key) to its ROC-AUC resulated metrics (child key) including
                                                            1. FPR, TPR, Threshold
                                                            2. Standard AUC (estimated from ROC curve using the trapezoidal rule)
                                                            3. DeLong AUC Estimate (estimated from Mann-Whitney U-statistic (mid-ranks))
                                                            4. AUC Variance
                                                            5. AUC Confidence Interval (95% CI) ; Derived using standard formula assuming AUC is the mean and the standard deviation is estimated from DeLong's AUC variance
    
        delong_comparison_df (pd.DataFrame): A dataFrame of pairwise DeLong's test results comparing ROC-AUCs between each model pair
    
    ----------
    Note:
    ----------
    The area under the ROC curve (AUC) can be estimated using two distinct approaches:
        1) The trapezoidal rule: A geometric method that computes the AUC directly from the empirical ROC curve by summing the area under the stepwise function defined by true positive rate (sensitivity) and false positive rate (1 - specificity) across thresholds.
        2) The Mann-Whitney U-statistic: A rank-based estimate of AUC, defined as the probability that a randomly chosen positive sample will have a higher score than a randomly chosen negative sample. Formally this given by:
    
    The empirical AUC is mathematically equivalent to the Mann-Whitney U-statistic.

    The confidence intervals for the AUCs is computed using the formula: CI = AUC ± z * σ where:
        - z is the critical value (1.96 for 95% confidence)
        - σ is the standard deviation estimated via DeLong's variance output (√variance)
    """

    # Initialize nested dictionary for ROC outputs
    roc_auc_results = defaultdict(dict)

    # Initialize dictionary to hold combined prediction scores for each model (foreground + background)
    method_predictions = {}

    # Initialize list to hold results for deLong's test
    delong_results = []

    # Initialize the figure and define its size (this will be the global axis for all curves)
    plt.figure(figsize=(10, 7))
    sns.set_style("ticks")
    sns.set_context("talk")

    # Define a dictionary to map each method's results to its associated score value key and color for ROC plot (note: each of the model's function return a list of dicts where each dict has a specific key for the score value)
    method_score_map = {
        "probability of occupancy": ("pocc"),
        "maximum site probabilities": ("maximum site probability"),
        "best hits": ("score"),
        "two state model": ("best hit effective probability"),
    #   "two state cluster score": ("two state"),
        "maximum cluster score": ("maximum local score"),
    }

    # Define a list of colors for the ROC plot
    colors = ["rebeccapurple", "mediumvioletred", "thistle", "crimson", "cornflowerblue", "darkorange"]

    # ------------------ Step 1: Compute ROC Values and and AUC Score for Each Model ---------------------

    # Iterate through each method and its corresponding score value and color (since .items() returns a view object that displays dictionary key-value pairs as tuple - the zip returns (method, score), color))
    for (method, score), color in zip(method_score_map.items(), colors):

        # Create list of results for each sequence in foreground (label = 1) and background (label = 0); There are three levels for the results dictionary i) Set key, ii) Method key, iii) Score key for each dict in list returned by model 
        fg_results = [f[score] for f in model_performance["foreground"][method]]
        bg_results = [b[score] for b in model_performance["background"][method]]

        # Combine all scores into a single list
        all_scores = fg_results + bg_results

        # Create a list of labels (ground truths) by concatenation of two seperate lists: i) list of ones with length equal to fg_scores and ii) list of zeroes with length equal to bg_scores (both fg_scores and bg_scores should have same length)
        all_labels = [1] * len(fg_results) + [0] * len(bg_results)

        # Compute datapoints for ROC curve (roc_curve returns an ndarray for FPR, TPR, and threshold)
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

        # Compute AUC for ROC Curve
        roc_auc = auc(fpr, tpr)

        # Convert ground truths and scores to numpy array (required for Delong Variance Estimation)
        ground_truths = np.array(all_labels)
        predicted_scores = np.array(all_scores)
        
        # Compute ROC AUC variance (delong_roc_variance returns the float value of the AUC and the covariance estimate)
        delong_auc, auc_variance = delong_roc_variance(ground_truths, predicted_scores)

        # Compute standard deviation of AUC by taking square root of variance
        auc_std = np.sqrt(auc_variance)

        # Compute the 5th and 95th percentile value for the AUC distribution (the percent-pointfunction (ppf) finds the value of a random variable at a given cumulative probability - is it essentially the inverse of the CDF)
        ci_95_low  = norm.ppf(0.05, delong_auc, auc_std)
        ci_95_high = norm.ppf(0.95, delong_auc, auc_std)

        # Add all ROC-AUC results to the nested dictionary 
        roc_auc_results[method] = {
            "tf name": tf_name,
            "FPR": fpr,
            "TPR": tpr,
            "Thresholds": thresholds,
            "AUC": roc_auc,
            "Delong AUC estimate": delong_auc,
            "AUC Variance": auc_variance,
            "AUC Confidence Interval": (ci_95_low, ci_95_high)
        }

        # Append the complete list of ground truths and predicted scores to the method predictions dictionary (nested)
        method_predictions[method] = {
            "scores": predicted_scores,
            "labels": ground_truths
        }

    # ------------------ Step 2: Plot the ROC Curve using ROC Values  -------------------------

        # Plot the ROC curve onto the global axis to overlay curves (using the data points computed above) - FPR (x), TPR (Y), and Method (hue - category)
        sns.lineplot(
            x=fpr, 
            y=tpr, 
            linewidth=2, 
            color=color,
            label="%s (AUC = %s)" % (method, round(roc_auc, 3))
        )

    # Plot a random classifier line 
    plt.plot(
        [0, 1], 
        [0, 1],     
        linestyle="--", 
        color="gray", 
        label="Random (AUC = 0.5)")

    # Set figure axis and legend
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()

    model_comparison_roc = os.path.join(plots_dir, roc_plot_name)
    plt.savefig(model_comparison_roc)
    plt.close()
    
    # ------------------ Step 2: Perform Delong's Test for Comparison of ROC-AUCs ---------------------

    # Iterate through all possible combinations of model comparisons (combinations [from itertools] generates all possible combinations of a specified length [2] from an iterable [keys from method_prediction])
    for model_1, model_2 in combinations(method_predictions.keys(), 2):

        # Extract ground truths for each model
        ground_truth_1 = method_predictions[model_1]["labels"]
        ground_truth_2 = method_predictions[model_2]["labels"]

        # Raise error if ground truths are not equal for both models
        if not np.array_equal(ground_truth_1, ground_truth_2):
            raise ValueError("Mismatch in ground truth labels between models: %s and %s" % (model_1, model_2))

        # Extract scores for each model
        scores_1 = method_predictions[model_1]["scores"]
        scores_2 = method_predictions[model_2]["scores"]
        
        # Raise error if scores are not equal for both models
        if len(scores_1) != len(scores_2):
            raise ValueError("Score vector length mismatch: %s has length: (%d) vs %s has length: (%d)" % (model_1, len(scores_1), model_2, len(scores_2)))

        # Builds an index that puts all positives (1) before negatives (0)
        order, label_1_count = compute_ground_truth_statistics(ground_truth_1)

        # Stack scores for both models (to prepare predictions for fastDeLong: shape (n_classifiers, n_examples))
        predictions = np.vstack([scores_1, scores_2])

        # Sort the predictions and ground truths based on the order index array (Required for fastDeLong; uses advanced Numpy indexing where [:,...] is a slice that takes all items along the axis (the part before the comma indexes rows; the part after indexes columns) and where [..., index] specifies how to re-order all columns (together they take all rows and re-order the columns))
        predictions_sorted = predictions[:, order]

        # Compute AUCs and DeLong covariance for the two models 
        _, delongcov = fastDeLong(predictions_sorted, label_1_count)

        # Compute p-value for DeLong's test for the specific model comparisons (DeLong's test computes log(p-value) for hypothesis that two ROC AUCs are different)
        log_p_val = delong_roc_test(ground_truths, scores_1, scores_2)

        # Append a dictionary comprising model's being compared and corresponding DeLong's p-value to list
        delong_results.append({
            "Model 1": model_1,
            "Model 2": model_2,
            "log(p-value)": log_p_val,
            "covariance": delongcov 
        })

    # Convert the list of dictionaries for DeLong test to a dataframe (column names = keys; rows = record/models)
    delong_comparison_df = pd.DataFrame.from_records(delong_results)

    # Save the CSV
    delong_output = os.path.join(plots_dir, delong_results_name)
    delong_comparison_df.to_csv(delong_output, index=False)

    return roc_auc_results, delong_comparison_df

def plot_auc_ratios(*roc_auc_dicts):
    """
    Plots the ratio of AUCs (model AUC / best AUC per dataset) for each model across multiple datasets.
    Each dataset is shown on the x-axis, and each model's ratio is a dot with error bars (CI).
    
    Args:
        *auc_results_dict (dict): An argument list of variable length, where each argument is a dictionary mapping dataset to ROC-AUC results.
    
    Returns:
        None: The function saves the plot to the specified directory (plots_dir).

    ----------
    Note:
    ----------

        The input dictionaries should be structured such that each key is a model name and the value is a dictionary containing the AUC score for that model on a specific dataset.

            (dict[str, dict[str, float]]) - Parent Key: scanning method and Child Key: ROC-AUC resulted metrics 

    """
    # ---------------- Step 0: Prepare Data ----------------

    # Create a dictionary for priority list of scores
    priority = {
    #   "two state cluster score"       : 0,
        'two state model'               : 1,
        "maximum cluster score"         : 2,
        "maximum site probabilities"    : 3,
        "best hits"                     : 4,
        "probability of occupancy"      : 5
    }

    # Extract all unique model names from the input dictionaries (set comprehension iterates through each result dictionary (outer loop) and then iterates through each key which are model names (inner loop) to create a set of unique model names)
    models = set(model for result in roc_auc_dicts for model in result)

    # Sort models for deterministic ordering using priority dictionary (if model not in priority, assign it infinity to sort it last)
    models = sorted(models, key=lambda m: priority.get(m, float('inf'))) 

    # ---------------- Step 1: Extract AUC Scores and Confidence Intervals from Input Dictionaries ----------------

    # Initialize list to hold AUC scores 
    AUC_scores = []

    # Iterate through each result dictionary in the input list (outer loop)
    for i, result in enumerate(roc_auc_dicts):

        # Initialize list to hold AUC scores for specific result dictionary
        aucs_across_models = []

        # Ensure input is a dictionary mapping dataset names to ROC-AUC results
        if not isinstance(result, defaultdict):
            raise ValueError("Input must be a dictionary mapping dataset names to ROC-AUC results.")
        
        # Iterate through each model parent key in the result dictionary (inner loop)
        for model in models:

            # Extract nested dictionary for the current model (if the model is not present, it will return an empty dictionary)
            model_results = result.get(model, {})

            # Extract AUC from the nested model's dictionary (the key of the current result dictionary)
            auc = model_results.get("AUC", 0.0)

            # Append AUC score to the list for this model
            aucs_across_models.append(auc)

        # Compute the maximum AUC score
        max_auc = max(aucs_across_models) if aucs_across_models else 0.0

        # Iterate through each model and its AUC pair (to compute the AUC ratio for each model (AUC / max AUC))
        for model, auc in zip(models, aucs_across_models):

            # Calculate the AUC ratio (AUC / max AUC) for the current model (if max_auc is greater than 0 to avoid division by zero)
            if max_auc > 0:
                auc_ratio = auc / max_auc
            else:
                auc_ratio = 0.0
            
            # Append the AUC ratio to the AUC_scores list
            AUC_scores.append({
                "index": i,
                "model": model,
                "AUC Ratio": round(auc_ratio, 2)
            })

    # Convert data to pandas dataframe
    AUC_scores_df = pd.DataFrame.from_records(AUC_scores)

    # ---------------- Step 2: Identify winning model per data-set and order data ----------------

    # Find the index labels of rows with the highest AUC Ratio in each group (the groupby() method divides the data into groups based on unique values; ["AUC Ratio"].idxmax() returns the index of the maximum element of each group from the ["AUC Ratio"] column)
    max_auc_indices = AUC_scores_df.groupby("index")["AUC Ratio"].idxmax()

    # Create a dataframe with only max AUC Ratio for each datset (the loc() method can be used to filter dataframe by selecting rows based on given indices)
    winning_models = AUC_scores_df.loc[max_auc_indices]
    
    # Rename columns of winning models for clarity
    winning_models = winning_models.rename(columns={"index": "dataset", "model": "best_model", "AUC Ratio": "best_AUC"})

    # Filter only the "maximum cluster score" rows and sort by descending AUC Ratio
    cluster_score_df = AUC_scores_df[AUC_scores_df["model"] == "maximum cluster score"].copy()
    cluster_score_sorted = cluster_score_df.sort_values(by="AUC Ratio", ascending=False)

    # Assign a rank column to the cluster score dataframe (this is used to order the datasets by their cluster score AUCs)
    cluster_score_sorted["rank"] = range(len(cluster_score_sorted))

    # Create a dictionary mapping dataset to its ordered cluster score rank
    cluster_ordered_score_dict = dict(zip(cluster_score_sorted["index"], cluster_score_sorted["rank"]))

    if len(cluster_score_sorted) != len(winning_models):
        raise ValueError("Mismatch: number of cluster score indices does not match number of cluster score winners. A dataset may not have a 'maximum cluster scores' model. Length of cluster score indices: %d, number of datasets: %d" % (len(cluster_score_sorted), len(winning_models)))

    # Create a new column of priority (by mapping best model (key) to associated value in priority dictionary above)
    winning_models["priority"] = winning_models["best_model"].map(priority)

    # Create a new column of cluster scores indices by mapping dataset to its ordered index in the cluster score indices (this is used to order the datasets by their cluster score AUCs)
    winning_models["cluster_score_rank"] = winning_models["dataset"].map(cluster_ordered_score_dict)

    # Group the winning models by priority only (best_AUC is always 1, so sorting by it is unnecessary)
    winning_models = winning_models.sort_values(["priority", "cluster_score_rank"], ascending=[True, True])

    # Convert column of datasets in winners dataframe to a list (dataset_order is a list of datasets in the order of their priority and best AUC scores)
    dataset_order = winning_models["dataset"].tolist()

    # Create a dictionary mapping each dataset to its ordered index in the list
    dataset_to_order = {ds: i for i, ds in enumerate(dataset_order)}

    # Add a column for x-axis positions (x) in dataframe (by mapping each dataset index to its ordered index)
    AUC_scores_df["x"]  = AUC_scores_df["index"].map(dataset_to_order) 

    # ---------------- Step 2: Plot AUC Scores of each Model for all Results ----------------

    # Create a figure and axis object (fig is the overall figure container; ax is the specific plot area within the figure); see note above on plt.subplots() - allows us to use object oriented approach to plotting below
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create a seaborn scatterplot of AUC scores 
    sns.scatterplot(
        data=AUC_scores_df,
        x="x",
        y="AUC Ratio",
        hue="model",
        palette='rocket',
        s=30,
        ax=ax,          
        linewidth=1.5,
    )

    # Set axis limits and labels
    ax.set_xlim(-1, len(dataset_order))
    ax.set_ylim(0.55, 1.05)

    # Set y-ticks to correspond to AUC ratio values with padding (0.55 to 1.05 with step of 0.05)
    ax.set_yticks(np.arange(0.55, 1.06, 0.05))

    # Add title and labels
    ax.set_ylabel("AUC ratio", fontsize=13)
    ax.set_xlabel("Dataset", fontsize=13)

    # Set x-ticks to correspond to dataset indices with small length 
    ax.set_xticks(range(len(dataset_order)))
    ax.tick_params(axis='x', length=3)

    # Set tick label size for both axes
    ax.tick_params(axis='both', labelsize=10)          

    # Remove x-axis labels (to avoid cluttering with dataset names); set_xticklabels() sets an empty list as the tick labels
    ax.set_xticklabels([])  

    # Retrieve the current legend handles/labels from seaborn
    handles, labels = ax.get_legend_handles_labels()

    # Create a dictionary mapping each label to its handle (zip creates a generator pair of labels and handles which is then consumed to form a dictionary; otherwise it would be consumed in the first loop)
    unique = dict(zip(labels, handles))

    # Filter handles to only include those corresponding to the models (models is a set of unique model names extracted earlier)
    handles_filtered = [handle for label, handle in unique.items() if label in models]
    labels_filtered = [label for label in unique if label in models]

    # Set legend for plot (from filtered hue labels)
    ax.legend(
        handles=handles_filtered,
        labels=labels_filtered,
        title=None, 
        frameon=False, 
        loc="lower right", 
        fontsize=8
    )

    # Remove top/right spines of the figure
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ---------------- Step 4: Create Horizontal Lines for Winning Models ----------------

    # Add a threshold line at AUC ratio of 0.95
    ax.axhline(0.95, color="black", linestyle="--", linewidth=0.5)

    # Create a dictionary mapping each model name to its label for the horizontal line
    model_labels = {
        "maximum cluster score": "Cluster Scores",
        "maximum site probabilities": "Site Probabilities",
        "two state model": "Two State Model",
    #    "two state cluster score": "Two State Cluster Scores",
        "best hits": "Hits",
        "probability of occupancy": "Pocc"
    }

    # Iterate over each unique model in the winning models dataframe
    for model in winning_models["best_model"].unique():

        # Filter the winning models dataframe to get only the rows corresponding to the current model
        winning_subset = winning_models[winning_models["best_model"] == model]

        # Get x position value for the winning model (by mapping the dataset to its order index; same as above for AUC_scores_df)
        x_vals = winning_subset['dataset'].map(dataset_to_order)  

        # Compute maximum and minimum x position values for best dataset
        x_start = x_vals.min()
        x_end = x_vals.max()

        # Define y position for the horizontal line and text
        y_line = 1.025
        text_y = 1.035

        # Get the label for the model or use the model name if not found
        model_name = model_labels.get(model, model)  

        # Plot a horizontal line for each winning model
        ax.hlines(
            y=y_line, 
            xmin=x_start - 0.2, 
            xmax=x_end   + 0.2, 
            color="black", 
            lw=1.0, 
            label=None,  # No label for the line itself
        )

        ax.text(
            x=((x_start - 0.2) + (x_end + 0.2)) / 2,    # Center the text between start and end of the line
            y=text_y,                                   # Position above the line
            s=model_name,                               # Text to display (model name)
            ha='center',                                # Horizontal alignment
            va='bottom',                                # Vertical alignment
            fontsize=8,                                 # Font size
            color="black",                              # Color of the text
        )

    # Save figure
    output_path = os.path.join(plots_dir, "AUC_Comparison_Plot.png")
    plt.savefig(output_path)
    plt.close()

    print("AUC ratio plot saved to %s" % output_path)
    
    return
    
if __name__ == "__main__":

    # Initialize a list to hold the ROC AUC results for each TF
    roc_auc_results = []
    
    # _____ Step 0: Define the File Directory Path _______________________________

    # Define the directory to store all outputs generated from the analyses
    plots_dir = os.path.join(this_dir, "Model_Comparison_dinuc_window300_cluster10")

    # Create an output folder of the specified name in the case that one does not already exist
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir) 

    # _____ Step 2: Define Paths to Relevant Input Files _______________________________

    hg_18_genome = os.path.join(this_dir, "Inputs/hg18.fa")
    hg_19_genome = os.path.join(this_dir, "Inputs/hg19.fa")
    hg_38_genome = os.path.join(this_dir, "../Extraction of Fasta Sequences from DEGs/input_data/hg38.fa")
    hif_chip_data_csv = os.path.join(plots_dir, "Chip-Seq Stringent Peaks.csv")

    negative_sequences = os.path.join(this_dir, "fit_evaluation_outputs/subcatalog/negative_samples_random_uniform_gc.bed")
   
    # _____ Step 2: Examine Parameter Space of Thermodynamic Model for HIF1A _______________________________
    """
    # Define paths for input TFFM model and Create the detailed TFFM Object (instance of TFFM class in TFFM module)
    hif_model_file = os.path.join(this_dir, "Inputs/TFFM_detailed_trained_HIF1A.xml")
    tf_tffm_model = tffm_from_xml(hif_model_file, TFFM_KIND.DETAILED)

    # Define path of bed file for negative samples (if genome matched method is used below)
    hif_neg_bed_file = os.path.join(plots_dir, "hif1a_neg_temp.bed")
    
    # Create foreground and background set for stringent binding sites from HIF-1A Chip-Seq
    ChipSeq_Foreground_Path, ChipSeq_Background_Path = ChipSeq_Fasta(
            chip_seq_path=hif_chip_data_csv,
            genome_file=hg_18_genome,
            foreground_path_name="hif1a_ChipSeq_Foreground.fa",
            background_path_name="hif1a_ChipSeq_Background.fa",
            temp_path_name="hif1a_temp.bed",
            random_seed=9246,
            window_size=1500,
            num_peaks=500,
            use_bed_file=False,
            negative_method="dinucleotide shuffling",
            neg_bed_path="hif1a_neg_temp.bed"
        )
   
    
    # Visualize parameter space for site probabilities
    parameter_space(
        foreground_fasta_path=ChipSeq_Foreground_Path,
        background_fasta_path=ChipSeq_Background_Path,
        xml_path=hif_model_file,
        variable="site probabilities",
        label_name="Site Probability"
    )
   
    # Visualize parameter space for cluster scores
    parameter_space(
        foreground_fasta_path=(os.path.join(plots_dir, "hif1a_ChipSeq_Foreground.fa")),
        background_fasta_path=(os.path.join(plots_dir, "hif1a_ChipSeq_Background.fa")),
        xml_path=hif_model_file,
        variable="cluster scores",
        label_name="Local Cluster"
    )
    
    # _____ Step 2: Run ROC-AUC Analysis for HIF1A Factor _______________________________
    
    print("Analyzing TF: %s..." % "HIF1A")

    # Run model scan
    models_performance = Scan_ChipSeq(
        foreground_set=ChipSeq_Foreground_Path,
        background_set=ChipSeq_Background_Path,
        tffm_object=tf_tffm_model,
        xml_path=hif_model_file,
        tf_name="HIF1A",
        parse_header_strand=False,
        window_size=150,
        lam=1.4,
        S0=4.0,
        tau=0.25,
        min_dist=5,
        cluster_constraint=False,
        negative_sequences=negative_sequences
    )

    # Run ROC AUC + DeLong test
    roc_auc_result, _ = ROCAUC_Comparison(
        model_performance=models_performance,
        tf_name="HIF1A",
        roc_plot_name="hif1_Comparative_ROC_Curve",
        delong_results_name="hif1_DeLong_Comparison",
    )
    # Append the ROC AUC results for HIF1A to the list
    roc_auc_results.append(roc_auc_result)
    """
    # _____ Step 2: Run ROC-AUC Analysis for all Factors _______________________________

    # Define list of TFs for analysis (total: 11)
    tf_list = [  
        "MYC", "TCF12", "NFIC", "EGR1", "MAX", "RBPJ", "JUN", "CEBPB",
        "BHLHE40", "USF1", "RFX5", 
    ]

    for tf in tf_list:

        print("Analyzing TF: %s..." % tf)

        # Define paths for input TFFM model and Chip-Seq Proccessed BED file
        tf_chip_data = os.path.join(this_dir, "Inputs/%s.bed" % tf)
        tf_model_file = os.path.join(this_dir, "Inputs/TFFM_detailed_trained_%s.xml" % tf)
        tf_neg_bed_file = os.path.join(this_dir, "fit_evaluation_outputs/subcatalog/%s_neg_temp_3000.bed" % tf)
        tf_bed_file = os.path.join(plots_dir, "%s_temp.bed"% tf)

        # Create the detailed TFFM Object (instance of TFFM class in TFFM module)
        tf_tffm_model = tffm_from_xml(tf_model_file, TFFM_KIND.DETAILED)
        
        # Extract foreground and background FASTA
        ChipSeq_Foreground_Path, ChipSeq_Background_Path = ChipSeq_Fasta(
            chip_seq_path=tf_chip_data,
            genome_file=hg_38_genome,
            temp_path_name="%s_temp.bed" % tf,
            foreground_path_name="%s_ChipSeq_Foreground.fa" % tf,
            background_path_name="%s_ChipSeq_Background.fa" % tf,
            random_seed=9246,
            window_size=500,
            num_peaks=500,
            foreground_bed_override=tf_bed_file, 
            negative_method="dinucleotide shuffling",
            neg_bed_path=tf_neg_bed_file
        )
        
        # Run model scan
        models_performance = Scan_ChipSeq(
            foreground_set=ChipSeq_Foreground_Path,
            background_set=ChipSeq_Background_Path,
            tffm_object=tf_tffm_model,
            xml_path=tf_model_file,
            tf_name=tf,
            parse_header_strand=False,
            lam=1.4,
            S0=4.0,
            tau=0.25,
            window_size=18,
            min_dist=0,
            cluster_constraint=False,
            negative_sequences=negative_sequences
        )

        # Run ROC AUC + DeLong test
        roc_auc_result, _ = ROCAUC_Comparison(
            model_performance=models_performance,
            tf_name=tf,
            roc_plot_name="%s_Comparative_ROC_Curve" % tf,
            delong_results_name="%s_DeLong_Comparison" % tf,
        )

        # Append the ROC AUC results for the current TF to the list
        roc_auc_results.append(roc_auc_result)

    # Plot the AUC ratios for all TFs (* unpacks the list into positional arguments)
    plot_auc_ratios(*roc_auc_results)
    
    print("All TFs Analyzed - Project Completed")


