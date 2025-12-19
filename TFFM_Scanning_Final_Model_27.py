# -*- coding: utf-8 -*-

import logging
import re  
import math
import numpy as np  
import os
import gzip
import xml.etree.ElementTree as ET  
            
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
from collections import namedtuple


# ==============================================================================
#                           Logging Configuration
# ==============================================================================

def setup_logging():
    """
    Sets up logging configuration for the script. This configures the logging level, format, and date format.

    This allows us to view time-stamped, leveled messages rather than print statements, which is useful for long-running scripts and multi-step pipelines.
    .
    Once configured, calls such as logging.info(), logging.warning(), and logging.error()
    will print to the terminal (stdout/stderr by default).

    Logging Configuration:
    
        1) level=logging.INFO:
        
            Only messages at INFO level or higher (INFO, WARNING, ERROR, CRITICAL) will be displayed. 
            DEBUG messages are suppressed unless the level is lowered.

        2) format="%(asctime)s | %(levelname)s | %(message)s":

            Specifies the format of the log messages:
                - %(asctime)s: Timestamp of the log entry
                - %(levelname)s: Severity level of the message (e.g., INFO, WARNING, ERROR)
                - %(message)s: The actual text passed to the logging call

        3) datefmt="%H:%M:%S":
            
            Specifies the format of the timestamp in the log messages. Here, it shows only hours, minutes, and seconds (e.g., "14:32:10").
          
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    return

# ==============================================================================
#                      Internal Helpers to Parse Fasta Header
# ==============================================================================

def parse_fasta_header(header, has_strand = True):
    """
    Parse a FASTA record header and retrieve its individual components.

    This function supports two header formats:
        1. If strand is included (has_strand=True): gene_name(strand)::chr:start-end ; Example: GENE(+)::chr1:100-200
        2. If strand is not included (has_strand=False): gene_name::chr:start-end ; Example: GENE::chr1:100-200

    ----------
    Args:

        header (str): The FASTA header string (record.id for record in SeqRecord object format)
        has_strand (bool, default=True): Indicates whether the header contains strand information in parentheses.

    ----------
    Returns:
    
        Tuple [str, str, int, int, Optional[str]]: Returns a 5-tuple containing
                - gene_name (str): Parsed gene name or original full header if parsing fails.
                - chrom (str): Chromosome name (e.g. 'chr1') or None if parsing fails.
                - start (int): Start coordinate of the region (1-based). -1 if parsing fails.
                - end (int): End coordinate of the region (1-based). -1 if parsing fails.
                - strand (str or None): '+' or '-' if present; None if parsing fails or has_strand=False.

    ----------
    Notes:
    ----------

    The expected regex patterns are:
            
        1. With strand: r"^([^()]+)\(([+-])\)::([^:]+):(\d+)-(\d+)"
        2. Without strand: r"^([^()]+)::([^:]+):(\d+)-(\d+)"

        ^            : Match start of the string
        ([^()]+)     : Capture any characters except parentheses (gene name)
        \(           : Literal opening parenthesis '(' (escaped with backslash)
        ([+-])       : Capture strand symbol, either '+' or '-'
        \)           : Literal closing parenthesis ')'
        ::           : Two literal colons
        ([^:]+)      : Capture chromosome name (any chars until next colon)
        :            : Literal colon
        (\d+)        : Capture one or more digits (start position)
        -            : Literal dash
        (\d+)        : Capture one or more digits (end position)

        The 'r' before the string denotes a raw string literal. This tells python not to treat backslashes as escape characters.

    The expected groups/sub-strings (captured by parentheses () in the regex) are:

        1. gene_name
        2. strand (+ or - or None if has_strand=False)
        3. chromosome
        4. start
        5. end
            
    """
    
    # Define the regex patterns
    pattern_with_strand = r"^([^()]+)\(([+-])\)::([^:]+):(\d+)-(\d+)"
    pattern_without_strand = r"^([^()]+)::([^:]+):(\d+)-(\d+)"

    # Compile the appropriate regex pattern (the re.compile() compiles the regular expression into a regex object which can be used multiple times to match against strings)
    pattern = re.compile(pattern_with_strand if has_strand else pattern_without_strand)

    # Attempt to match the pattern to the header string (the .match() method tries to match the pattern from the start of the string; If the string matches, it returns a Match object. If not, it returns None).
    match = pattern.match(header)

    if not match:
        # If the header does not match the expected format: return a fallback tuple
        return (header, None, -1, -1, None)

    # Extract matched groups from the Match object (the .group(n) method retrieves the substring matched by the nth capturing group (parentheses in the regex) for a match object)
    if has_strand:
        gene_name = match.group(1)
        strand_symbol = match.group(2)
        chrom = match.group(3)
        start = int(match.group(4))     # convert string digits to integer
        end = int(match.group(5))       # convert string digits to integer
    else:
        gene_name = match.group(1)
        chrom = match.group(2)
        start = int(match.group(3))     # convert string digits to integer
        end = int(match.group(4))       # convert string digits to integer
        strand_symbol = None            # No strand information present

    return (gene_name, chrom, start, end, strand_symbol)

def compute_upstream_distance(promoter_s, hit_start, hit_end, strand):
    """
    Given the promoter region (region_s, region_e) and strand, plus a hit's genomic
    start/end, return the distance (in bp) upstream from TSS to the center of the hit:
      • For + strand: TSS = region_s + 1000.  Distance = (TSS) - (hit_midpoint).
      • For - strand: TSS = region_s + 300.   Distance = (hit_midpoint) - (TSS).
    (Because on - strand, “upstream” in genomic coordinates is larger than TSS.)
    """

    # Convert all inputs to correct type if they are not correct
    if not isinstance(promoter_s, int):
        promoter_s = int(promoter_s)
    if not isinstance(hit_start, int):
        hit_start = int(hit_start)
    if not isinstance(hit_end, int):
        hit_end = int(hit_end)
    if not isinstance(strand, str):
        strand = str(strand)

    # Compute the upstream distance based on strand
    if strand == "+":
        tss = promoter_s + 1000
        mid = float(hit_start + hit_end) * 0.5
        distance = tss - mid
    
    else:
        tss = promoter_s + 300
        mid = float(hit_start + hit_end) * 0.5
        distance = mid - tss

    return distance

def transform_absolute_coordinates(promoter_s, promoter_e, hit_strand, hit_start, hit_end):
    """
    Convert strand-relative hit coordinates given by TFFM framework into absolute genomic coordinates in + strand coordinate system

    For hits on + strand: A match found on forward strand already matches FASTA sequence orientation (genomic start = seq[0]) so promoter_s must be offset by leftmost motif base (region_s + i) to find "genomic start"

    For hits on - strand: A match found on reverse strand reverse-compliments FASTA sequence orentiation (genomic end = seq [0]) so promoter_e must be offset by rightmost motif base (region_e - i) to find "genomic start"

    """
    if hit_strand == "+":
        genomic_hit_start = promoter_s + hit_start
        genomic_hit_end   = promoter_s + hit_end

    else:
        genomic_hit_start = promoter_e - hit_end
        genomic_hit_end   = promoter_e - hit_start

    return (genomic_hit_start, genomic_hit_end)

# ==============================================================================
#           Extract Probabilities and Emissions for all States from HMM
# ==============================================================================

def extract_probabilities(xml_path):
    """
    Parse a detailed TFFM XML and pull out its HMM parameters - including the initial probabilities and emission vector for each state, and transition probabilities for each transition between states.

    Args:

        xml_path (Path): Path to the TFFM detailed XML file.
    
    -----------------------------
    Returns
    -----------------------------

        tuple (List, Dict, Dict, DefaultDict): A tuple consisting of:

            state_ids (List[int]): A sorted list of all state IDs in the detailed TFFM.

            initial_vals (Dict[int, float]): A dictionary mapping from each state ID (key; int) to its initial probability (value; float).
                                            initial_probs[state] = P(initial state)

            emission_vectors (Dict[int, List[int]]): A dictionary mapping each state ID (key; int) to its 1-hot emission vector (value; list of ints)
                                                    emissions_identity[state] = [0.0,1.0,0.0,0.0]
            
            transition_vals (DefaultDict[int, Dict[int, float]]): A nested dictionary mapping each source state (parent key; int) to its target states (child key; int) with associated log-transition probabilities (value; float)
                                                                transitions[src][tgt] = log P(tgt | src)
        
    -----------------------------
    Overview:
    -----------------------------

        The TFFM (Transcription Factor Flexible Model) is an HMM that encodes transcription factor binding motifs with previous context-dependent base preferences. The model consists of:
            (i) States 
                    a) Each <state> represents one position in the motif conditioned on a specific previous state. This <state> contains both:
                        1) Initial probability (likelihood of a given state starting the TFBS [transcription factor binding site])
                        2) Discrete emission vector (the nucleotide [A, C, G, T] being emitted at the state; encoded as a 1-hot vector (e.g., [0,0,1,0] for G)
            (ii) Transitions
                    a) Each <transition> represents the probability of moving from one state (source state ID) to a specific next state conditioned on the previous state (target state ID)
                        1) This is encoded by a transition probability (likelihoods of A, C, G, T being emitted at the next state)
    
    -----------------------------
    Model Design:
    -----------------------------

        The possible states are expanded in detailed TFFMs: each motif position is decomposed into 4 context-specific states (i.e., for A, C, G, T at the previous position). Hence for an 10-bp motif we expect:
            40 foreground states for state transitions (4 states per position x 10 positions).
            4 background-conditioned entry states for motif initiation (corresponding to each background nucleotide)

            
    -----------------------------
    Model Structure:
    -----------------------------

        The structure of the detailed TFFM:

            a) The initial probability is assigned only to the first four states - the background-conditioned entry states - which correspond to motif initiation depending on the nucleotide emitted from the background 
                - All other states (positions 1 through n) have ~0 initial probability, as they are only reachable through explicit transitions from the preceding state.

            b) Emission vectors are deterministic (i.e. 1-hot encoded); each state emits a fixed nucleotide with probability 1.0. 
                - Emissions do not contribute to probabilistic variability or scoring - the emission identity is hard-coded into the state.

            c) All stochasticity in the model lies in the transition probabilities, which encode the conditional preferences for which nucleotide should follow a given preceding nucleotide.
                - Dinucleotide dependencies are thus modeled entirely through the state transitions: from a given state emitting base X at position *p*, the model probabilistically transitions to one of four states at position (p+1), each corresponding to a specific nucleotide Y. The transition probability reflects P(Y | X), mimicking a first-order Markov chain over motif bases.

    """

    # Parse the XML tree and construct the ElementTree (an XML tree starts at the root element and branches from root to child elements)
    tree = ET.parse(xml_path)

    # Retrieve the root element from the XML tree (<mixture>)
    root = tree.getroot()

    # Retrieve the <HMM> element under the <mixture> element
    hmm_root = root.find("HMM")

    print ("The XML root: %s" % root)

    # ------------------ Step 1: Extract Emissions and Initial Probabilities ---------------------

    # Initialize a list to hold state IDs for each state
    state_ids = []

    # Initialize a dictionary to hold initial probabilities for each state (float)
    initial_vals = {}

    # Initialize a dictionary to hold emission vectors for each state (list of floats)
    emissions_vectors = {}

    # Loop over all <state> elements in the HMM root
    for state in hmm_root.findall(".//state"):

        # Extract the ID attribute from <state> element (id ranges from 0-43 for HIF-1A) and store as an integer (.attrib returns a dictionary of all attributes for element and we access it with the key ['id'] to extract the state ID value)
        state_id = int(state.attrib['id'])

        # Extract initial probability attribute from <state> element and default to zero if it does not exist (.attrib.get returns 0 (fallback) if the attribute key [initial] is missing)
        initial_prob_value = float(state.attrib.get('initial', '0.0'))

        # Extract 1-hot encoded emission vector from the nested "discrete" element (.find("discrete") returns the <discrete> child element inside the specific <state> element, and .text extracts the inner string (e.g., <discrete>0.0, 1.0, 0.0, 0.0</discrete> --> "0.0, 1.0, 0.0, 0.0") from which we remove whitespaces (.strip()), split the string into a list based on commas (.split(",")), and then convert each value (x) in list into a float
        emission_vector = [int(x) for x in state.find("discrete").text.strip().split(',')]

        # Store the state ID in the state IDs list
        state_ids.append(state_id)

        # Store the initial probability (value) for the state ID (key)
        initial_vals[state_id] = initial_prob_value

        # Store the emission vector (list) for the state ID (key)
        emissions_vectors[state_id] = emission_vector
        
    # Sort the State IDs in-place in ascending order 
    state_ids.sort()

    if not emissions_vectors or not initial_vals:
        raise ValueError("initial probabilities and/or emission vectors not found")

    # ------------------ Step 2: Extract Transition Probabilities ---------------------

    # Initialize a nested dictionary where a source state points to a transition state that contains the transition probability (transitions[src][tgt] = probability of moving from src -> tgt)
    transition_vals = defaultdict(dict) 

    # Loop over every <transition> elements in the HMM root 
    for trans in hmm_root.findall(".//transition"):

        # Retrieve the source state attribute for the transition
        src = int(trans.attrib["source"])

        # Retrieve the target state attribute for the transition 
        tgt = int(trans.attrib["target"])

        # Extract the transition probability from the nested <probability> element (.find("probability") returns the <probability> child element inside the <transition> element, and .text extracts its inner value (e.g., <probability>0.93</probability> --> "0.93").
        prob = float(trans.find("probability").text)

        # Store the log-adjusted values in nested dictionary using the source as the higher key and target as the lower key (transitions[0][1] = 0.95; if transition probability is zero and thereby its log is undefined, store it as negative infinity)
        transition_vals[src][tgt] = math.log(prob) if prob > 0 else float("-inf")

    if not transition_vals:
        raise ValueError("transition values not found")

    return (state_ids, initial_vals, emissions_vectors, transition_vals)

# ==============================================================================
#               Compute Maximum Achievable Score Through HMM
# ==============================================================================

def trace_back(state_ids, motif_length, dp, path):
    """
    Reconstructs the highest-scoring state sequence (“Viterbi path”) from dynamic-programming tables.

    Args:

        state_ids (List[int]): A sorted list of all state IDs in the detailed TFFM.

        motif_length (int): Number of HMM steps (time-points) over which dynamic programming was run.

        dp (list[defaultdicts]): List of defaultdicts (length motif_length) mapping state-id to cumulative log-score of the best path ending in that state at each time step; 
                where dp[t][s] is the maximum log-likelihood of reaching state s at time t.

        path (list[dicts]): List of dicts (length motif_length) mapping state-id to previous state-id along that best path at each time step
                where path[t][s] is the state at time t-1 that gave dp[t][s] when extending the best path.

    Returns:

        tuple (float, list): A tuple consisting of
            best_score: The maximum log-likelihood over all end states at t = motif_length-1.
            best_path: The list of state IDs (length motif_length) giving that maximum path,
                    ordered from t = 0 -> motif_length-1.

    Raises:
        ValueError: If dp or path are not of length motif_length.
    """

    # Raise error if length of dp and path lists do not equal the number of time steps
    if len(dp) != motif_length or len(path) != motif_length:
        raise ValueError("dp and path must each have length == motif_length")
    
    # Define the final states by creating a shallow copy slice of the last 4 states 
    final_states = state_ids[-4:]

    # Retrieve dictionary of all scores at final time-step
    last_scores = dp[-1]              

    # Initialize a dictionary to hold only valid final states in last time step
    valid_final_scores = {}

    for state in final_states:

        # Retrieve the value for the final state if it exists in the final time-step; otherwise return -∞
        score = last_scores.get(state, float("-inf"))

        # Create a new entry in dictionary for state (key) that have associated score (value) in final time step
        if score > float("-inf"):
            valid_final_scores[state] = score

    # If no final state was reachable, return null result
    if not valid_final_scores:
        return float("-inf"), []

    # Select the state ID (key) with highest final score (value); when max() is applied on an iterable - the key= parameter for max specifies a function (.get in this case) that returns a value for each element (key) of the iterable (.get retrieves values associated with given keys in dictionary)
    end_state = max(valid_final_scores, key=valid_final_scores.get)    
    
    # Retrieve the score value from the dictionary using the state ID key
    best_score = last_scores[end_state] 

    # Initialize a list to hold best path (path is computed via traceback; first value is the end state)
    best_path = [end_state]

    # Iterate through list of motif positions in reverse (t = motif length - 1 (since we have defined final state already) to t = 1)
    for t in reversed(range(1, motif_length)):
        
        # Retrieve the previous state from the path list for the end state (end state )
        prev_state = path[t][end_state]   

        # Append the previous state to the best path list
        best_path.append(prev_state)

        # Update the current end state of the path list with the appended state 
        end_state = prev_state

    # Reverse the path list formed from traceback to get forward path
    best_path.reverse() 

    return (best_score, best_path)

def get_max_score(state_ids, initial_vals, transition_vals):
    """
    Compute the maximum achievable log-likelihood score (S-max) for a given TFFM detailed model by simulating the most probable emission path through the Hidden Markov Model (HMM). 
    This function uses a complete Viterbi-style dynamic programming approach to infer the maximum path score through a HMM.

    Args:

        state_ids (List[int]): A sorted list of all state IDs in the detailed TFFM.

        initial_vals (Dict[int, float]): A dictionary mapping from each state ID (key; int) to its initial probability (value; float).
                                          initial_probs[state] = P(start in state s)
        
        transition_vals (DefaultDict[int, Dict[int, float]]): A nested dictionary mapping each source state (parent key; int) to its target states (child key; int) with associated log-transition probabilities (value; float)
                                                          transitions[src][tgt] = log P(tgt | src)
    
    -----------------------------
    Returns
    -----------------------------

        best_score (float): The maximum log-likelihood score. This represents the most probable path (highest probability) to achieve the final state of the motif instance.
        best_path (list): The optimal state path through the HMM.
    
    -----------------------------
    Method Summary:
    -----------------------------

        The Viterbi algorithm computes the most probable path through the HMM by storing at each time step (t) the best log-probability of reaching each state (s), as well as the backpointer to the best previous state:

            1) Initialization
                a) We assign an initial log-probability from the model (log(P_init)) to each background-conditioned entry state. All other states are initialized with log-probability = −∞ (impossible at t = 0).

            2) Dynamic_Recursion(from_t=1_to_T-1)
                a) We consider for each previously reachable state (s_prev) at time step t-1 (initial background states (t[0]) for time step 1), every possible successor state (s_next) reachable via transitions from the previous state (s_prev).
                b) We compute a candidate cumulative score for each possible successor state based on current path score and transition probabilities (for time step 1: log(P(initial)) + log(P(s_next | s_prev)); emission probabilities are ignored since they are deterministic
                c) We update the candidate score for the successor state if it exceeds all other computed scores for the state at that timestep. The previous state that lead into the current successor state at the timestep is then saved for backtracing. 
                    i) This is known as optimal substructure property. We only consider the optimal solution of this subproblem - since all the paths that go through the successor state at a specific time-step continue identically from that state onwards (the next transition only depends on current state in Markov chains)

            3) Termination
                a) After the dynamic recursion, we scan the cumulative scores for each state in the final time step to identify the state with the maximum score

            4) Backtrace
                a) We recover the optimal state path by recursively following the stored backpointers for that final state (starting from best final state to initial state). This reconstructs the optimal path through the HMM that achieved S-max.
       
    """

    # ------------------ Step 1: Dynamically Identify Path with Highest Probability ---------------------

    # Derermine the number of total state IDs (should be 44 for HIF-1A)
    num_states = len(state_ids)
    
    # Define the motif length based on number of <state> elements in the HMM root (each motif position is decomposed into 4 states; includes background-conditioned entry states)
    motif_length = num_states // 4

    # Initialize a list of defaultdicts (one per time step as defined by motif length) where the default value for each key is a negative infinity float (since its represents 0 probability - impossible path) ; Creates matrix where each row corresponds to time step (position in motif; t) and each column corresponds to state in HMM
    dp = [defaultdict(lambda: float("-inf")) for _ in range(motif_length)]

    # Initialize a list of dictionaries (one per time step) to hold best previous state leading to state (s) at time 't'
    path = [{} for _ in range(motif_length)]

    # Loop over all states in list of state IDs (index = state ID)
    for s in state_ids:

        # Determine if initial probability (value) for state ID (key) is greater than zero 
        if initial_vals[s] > 0:

            # Store the log-adjusted initial probaility in dynamic programming table ([t][s] where [t] is row 0 in this case and [s] is key for value); dp[t] retrieves defaultdict for time-step and dp[t][state id] inserts key for the specific state into dictionary
            dp[0][s] = math.log(initial_vals[s]) 

    # Loop over all possible time states following background-conditioned state (outer loop; t = 1 to t = motif length - 1)
    for t in range (1, motif_length):
            
            # Loop over all states that were reached at the previous step (first inner loop; refers to background states for first iteration of outer loop)
            for s_prev in dp[t - 1]:

                # Loop over all possible target states (child key) for the source states (parental key) in the previous time step (second inner loop; transitions[s_prev] accesses the nested dictionary of all valid target states that we can jump to and their associated log-probability values)
                for s_next in transition_vals[s_prev]:

                    # Retrieve the transition value corresponding to the source state (parental key) and target state (child key)
                    transition_score = transition_vals[s_prev][s_next]

                    # Define the total score of the current path by adding transition probability of next state to previous state
                    candidate_score = dp[t - 1][s_prev] + transition_score

                    # Determine if the current path from the previous to current state is the highest-probability path that has been found thus far (only retain highest scoring incoming path for each next state)
                    if candidate_score > dp[t][s_next]:

                        # Update the candidate score with the new best score (if found)
                        dp[t][s_next] = candidate_score

                        # Store the backpointer (previous state) for the current state (key) to the path list at the specific timestep (to later re-construct the best path) 
                        path[t][s_next] = s_prev 

    # ------------------ Step 5: Trace Back Optimal Path ---------------------

    best_score, best_path = trace_back(
        state_ids=state_ids,
        motif_length=motif_length,
        dp=dp,
        path=path
    )                               

    return (best_score, best_path)

# ==============================================================================
#                   Perform Sliding Viterbi Algorithm
# ==============================================================================

# Define a NamedTuple called VHit with field names relevant to Viterbi scoring (to allow access to tuple elements via the descriptive field names)
VHit = namedtuple("VHit", ["end", "score", "path", "sequence", "strand"])

def reverse_complement(sequence):
    """
    Return the reverse complement of a DNA sequence.

    Args:
        sequence (SeqRecord or Seq or str): A Bio.SeqRecord object or Bio.Seq object or a string representing the input DNA sequence to be reverse complemented.
            The input is converted to a Seq object regardless of its original type.

    Returns:
        reverse_sequence (str): A string representing the reverse complemented input DNA sequence
    """

    # Convert the input sequence to a Seq object if it is not already one
    if isinstance(sequence, SeqRecord):
        seq_obj = sequence.seq
    elif isinstance(sequence, Seq):
        seq_obj = sequence
    elif isinstance(sequence, str):
        seq_obj = Seq(sequence)
    else:
        raise TypeError("Input must be a SeqRecord, Seq, or str")

    # Reverse complement the DNA string and return a new SeqRecord object (using the .reverse_complement() method of the Seq object)
    reverse_sequence = str(seq_obj.reverse_complement())

    return reverse_sequence

def Viterbi_algorithm(sequence, state_ids, initial_vals, transition_vals, emission_vectors):
    """
    Slide a detailed-TFFM HMM across a DNA sequence and, at each possible window, compute the single most likely state path (“Viterbi path”) and its log-likelihood.

    Args:
        sequence (SeqRecord or str): A Bio.SeqRecord object or a string representing the input DNA sequence to be scanned.

        state_ids (List[int]): A sorted list of all state IDs in the detailed TFFM.

        initial_vals (Dict[int, float]): A dictionary mapping from each state ID (key; int) to its initial probability (value; float).
                                          initial_probs[state] = P(start in state s)

        emission_vectors (Dict[int, List[int]]): A dictionary mapping each state ID (key; int) to its 1-hot emission vector - indicates which base state s deterministically emits (value; list of ints) 
                                                 emissions_identity[state] = [0.0,1.0,0.0,0.0]
        
        transition_vals (DefaultDict[int, Dict[int, float]]): A nested dictionary mapping each source state (parent key; int) to its target states (child key; int) with associated log-transition probabilities (value; float)
                                                          transitions[src][tgt] = log P(tgt | src)

    Returns:
        hits (list[VHit]): A list of NamedTuples (end, score, path) for every possible L-mer window in the sequence where:
            - end (int):             The 0-based index of the window's terminal base
            - score (float):         The maximal joint log-probability of the Viterbi path through the window (∑logP(transition))
            - path (List[str]):      The state ID sequence for that path
            - sequence (List[str]):  The nucleotide sequence of the L-mer window
            - strand (str):          The strand on which the L-mer window was applied

    -----------------------------
    Overview:
    -----------------------------
    This function implements the classical Viterbi algorithm in a sliding window manner over a DNA sequence using a detailed Transcription Factor Flexible Model (TFFM). The windows are taken at every offset of the sequence (motif_length ≤ i ≤ len(sequence))
    The model is a hidden Markov model (HMM) that uses deterministic emissions and position-conditional transition probabilities to capture the sequence dependencies within a TF binding motif.

    Due to deterministic emissions (each state emits exactly one nucleotide), the only stochasticity arises from the di-nucleotide transitional probabilities. 
    Thus, the scoring reduces to accumulating log-transition probabilities along the most probable path of states that match the observed bases (sum of L-1 log-transition weights)
    """

    # Convert the sequence (regardless of type) to an uppercase string (since the inner-loops are comprised of many slicing operations. strings are are faster [less overhead] than Seq objects)
    if isinstance(sequence, SeqRecord):
        seq = str(sequence.seq).upper()
    elif isinstance(sequence, Seq):
        seq = str(sequence).upper()
    elif isinstance(sequence, str):
        seq = sequence.upper()
    else:
        raise TypeError("The passed 'sequence' must be a SeqRecord, Seq, or str")

    # Initialize a list to hold overall results for sequence (list of NamedTuples)
    hits = []

    # Derermine the motif length and sequence length (each motif position is decomposed into 4 states; includes background-conditioned entry states)
    M = len(state_ids) // 4
    N = len(seq)

    # ----------------------------- Step 1: Viterbi Sliding Window ----------------------------- #

    # Create a dictionary that maps the nucleotide (key) to the one-hot encoded list (value); default to null emission if base is ambiguous (N)
    one_hot_dict = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }

    def viterbi_sliding(window, end, strand):
        """
        Internal helper: runs Viterbi on a single window for a given strand.
        """
        
        # Initialize a list of defaultdicts (one per time step as defined by motif length) where the default value for each key is negative infinity (since its represents 0 probability - impossible path) ; Creates matrix where each row corresponds to time step (position in motif; t) and each column corresponds to state in HMM
        dp = [defaultdict(lambda: float("-inf")) for _ in range(M)]

        # Initialize a list of dictionaries (one per time step) to hold best previous state leading to state (s) at time 't'
        path = [{} for _ in range(M)]

        # Loop over all states in list of state IDs (index = state ID; initializes first row of dp)
        for state_id in state_ids:

            # Extract the letter corresponding to the first nucleotide in the window
            first_base = window[0]

            # Return the corresponding emission vector (one hot list) for the first nucleotide 
            first_emission = one_hot_dict[first_base]
            
            # Determine if the state's emission matches the observed emission
            if emission_vectors[state_id] == first_emission:

                # Retrieve the initial probability of the corresponding state
                init_prob = initial_vals[state_id]

                # Determine if initial probability (value) for state ID (key) is greater than zero 
                if init_prob > 0:

                    # Store the log-adjusted initial probaility in dynamic programming table ([t][s] where [t] is row 0 in this case and [s] is key for value); dp[t] retrieves defaultdict for time-step and dp[t][state id] inserts key for the specific state into dictionary
                    dp[0][state_id] = math.log(init_prob)
        
        # Loop over all possible time states following background-conditioned state (outer loop; t = 1 to t = motif length - 1)
        for t in range (1, M):
                
            # Extract the letter corresponding to the nucleotide (t) in the window
            obs_base = window[t]

            # Return the corresponding emission vector (one hot list) for nucleotide t 
            obs_emission = one_hot_dict[obs_base]
            
            # Loop over all states that were reached at the previous step (first inner loop; refers to background states for first iteration of outer loop)
            for s_prev in dp[t - 1]:

                # Loop over all possible target states (child key) for the source states (parental key) in the previous time step (second inner loop; transitions[s_prev] accesses the nested dictionary of all valid target states that we can jump to and their associated log-probability values)
                for s_next in transition_vals[s_prev]:

                    # Skip any state whose emission vector does not match that of the observed base
                    if emission_vectors[s_next] != obs_emission:
                        continue

                    # Retrieve the transition value corresponding to the source state (parental key) and target state (child key)
                    transition_score = transition_vals[s_prev][s_next]

                    # Define the total score of the current path by adding transition probability of next state to previous state
                    candidate_score = dp[t - 1][s_prev] + transition_score

                    # Determine if the current path from the previous to current state is the highest-probability path that has been found thus far (only retain highest scoring incoming path for each next state)
                    if candidate_score > dp[t][s_next]:

                        # Update the candidate score with the new best score (if found)
                        dp[t][s_next] = candidate_score

                        # Store the backpointer (previous state) for the current state (key) to the path list at the specific timestep (to later re-construct the best path) 
                        path[t][s_next] = s_prev

        # ------------------ Step 5: Trace Back Optimal Path ---------------------

        # Define the final states by creating a shallow copy slice of the last 4 states 
        final_states = state_ids[-4:]
 
        # Retrieve dictionary of all scores at final time-step
        last_scores = dp[-1]              

        # Initialize the best score variable as negative infinity (default)
        best_score = float("-inf")    

        # Determine if the final state is present in the final time-step (last_scores is a dictionary of {state: score}); otherwise default to negative infinity
        valid_final_scores = {last_scores.get(state, float("-inf")) for state in final_states}

        # Determine the maximum cumulative score among paths that end in a final state
        max_valid_final_scores = max(valid_final_scores)

        # Assign null values if path does not end in a valid final state (includes instances where there are no valid paths through the window)
        if max_valid_final_scores == float("-inf"):

            # Return a VHit with negative infinity score, empty path, empty sequence, and strand
            return VHit(end, best_score, [], [], strand)

        # Perform traceback if the path ends in a valid final state
        else:
            best_score, best_path = trace_back(
                state_ids=state_ids,
                motif_length=M,
                dp=dp,
                path=path
            )

            # Define a variable to hold string of nucleotides used in window
            sequence_path = window

            # Convert the list of states that comprise best hit path to a string (with commas)
            best_path = ','.join(str(state) for state in best_path)    

            # Return associated values
            return VHit(end, best_score, best_path, sequence_path, strand)
        
    # Return the reverse complement of the sequence (note: we pass the original SeqRecord object to the reverse_complement here)
    reverse_sequence = reverse_complement(sequence).upper()

    # Loop over each base in sequence string that can be a valid motif end site (window is taken from motif_length ≤ i ≤ len(sequence))
    for end in range(M, N+1):  

        # Define the start site of the forward window 
        start = end - M

        # Create a shallow slice copy of the sequence ranging from the start site to the end site (position of interest) 
        forward_window = seq[start : end]

        # Define the start and end of the reverse complement window 
        rc_start = end - M
        rc_end   = end

        # Create a shallow slice copy of the reverse complement sequence ranging from the start site to the end site (position of interest)
        reverse_window = reverse_sequence[rc_start : rc_end] 

        # Map Reverse complement window to the forward strand exclusive end (to ensure that the reverse strand is processed in the same way as the forward strand)
        end_excl_rev_fw = N - rc_start
        
        # Perform viterbi sliding algorithm for forward strand
        forward_hit = viterbi_sliding(forward_window, end, '+')
        
        # Perform viterbi sliding algorithm for reverse strand
        reverse_hit = viterbi_sliding(reverse_window, end_excl_rev_fw, '-')

        # Append forward and backward scanning results to hit
        hits.append(forward_hit)
        hits.append(reverse_hit)

    return hits

# ==============================================================================
#                       First-order Markov background (enhanced)
# ==============================================================================

class MarkovBG1:
    """
    First-order (dinucleotide) background model with Laplace smoothing and z-score calibration for log-likelihoods
    
    This class is designed to plug directly into the recognition-energy term used by the Fermi-Dirac mapping.

    --------------------------------------------------------------------------
    Overview
    --------------------------------------------------------------------------    
    
    The motif HMM (from the TFFM xml) captures structured, context-dependent preferences within TF binding sites and yields Viterbi log-scores S(i).
      
    MarkovBG1 captures coarse genomic composition (mono/di-nucleotide bias) that is agnostic to the motif. 

    These two models are combined later at the recognition energy level.

    --------------------------------------------------------------------------
    Recognition energy context
    --------------------------------------------------------------------------
    
    For every L-mer window within a sequence, we compute the following:
    
        • Motif score    S(i)   = log P_HMM(best path, window_i)        (≤ 0)
        • Ceiling score  S_max  = max over all possible HMM paths       (≤ 0)
        • Gap            ΔS(i)  = S_max - S(i)                          (≥ 0)

    The recognition energy (before background correction) is then a dimensionaless quantity expressed as:

        ER_raw(i) = (ΔS(i) - S0) / λR          

        where S0 and λR are tunable parameters (see Fermi-Dirac docstring for details).

    This class provides a matched background statistic for the same window for which the recognition energy is computed:

        B(i)   = log P_bg(window_i)     (≤ 0)
        B_z(i) = (B(i) - μ) / σ         (unitless z-score)

    The final corrected recognition energy is then:
  
        E_R(i) = ER_raw(i) + w * B_z(i)

        where w is a single tunable knob (“penalize by this many sigmas”).

    Since B_z(i) is unitless, it can be directly added to ER_raw(i).
    Any constant shift introduced by standardization is absorbed into S0, so there is no unit
    mismatch with the Fermi exponential.

    --------------------------------------------------------------------------
    Model Details
    --------------------------------------------------------------------------
    
    This class represents a 1st-order Markov chain trained on background DNA sequences to estimate:
   
         π(x1)      : Initial/base distribution for the first letter of a window
         T(x→y)     : Transition probabilities between adjacent letters (dinucleotide transition matrix)

    The class estimates log-likelihoods of windows under that model and standardizes them (z-scores)

    In essence, the bg term is a nuisance regressor - it captures genomic idiosyncrasies that
    are common within the given negative corpus. This allows the recognition energy to downweight
    motifs whose composition is similar to the given background, and thereby focus more on motif-specific information.

    --------------------------------------------------------------------------
    Parameters learned and stored
    --------------------------------------------------------------------------
      
      self.pi       : shape (4,)      ; initial base probabilities
      self.T        : shape (4,4)     ; P(next|current) with Laplace smoothing
      self.zparams  : dict or None    ; calibration info for z-scores with keys: 

          {
            "L": <int>,                 # window length used for calibration
            "per_bp": <bool>,           # whether scores were normalized per bp
            "global": {"mu": float, "sigma": float, "n": int},
            "by_gc": [
                {"lo": float, "hi": float, "mu": float, "sigma": float, "n": int}, ...
            ]
          }

    Use flow:
      1) fit(...) on negatives (or any background corpus) to estimate π, T
      2) calibrate_z_from_negatives(...) on GC-uniform negatives for a chosen L
      3) zscore(window, use_gc_bins=True) at inference time
    """

    def __init__(self, pseudocount=1.0, alphabet="ACGT"):

        # Define and store model parameters
        self.pc = float(pseudocount)                        # Pseudocount for Laplace smoothing (default 1.0)
        self.alphabet = alphabet                            # Alphabet (default A/C/G/T); Accepts only these Nucleotides in sequences 
        self.a2i = {a:i for i,a in enumerate(alphabet)}     # Map letter to index (A:0, C:1, G:2, T:3)
        self.pi = None                                      # initial probs P(s1)
        self.T  = None                                      # transitions P(b|a), shape (4,4)
        self.zparams = None                                 # calibration store (see docstring)

    # ------------------------------- Core training --------------------------------

    def fit(self, sequences):
        """
        Estimate π and T from background sequences via Laplace-smoothed counts.

        Arguments
        ---------
        sequences : iterable[str]
            Iterable of DNA strings (A/C/G/T/N). Ns are ignored within strings.

        Method
        ------
        
        1) π is estimated from first valid letter of each sequence (if present).
                c0[b] ~ count of first letters (for π)

        2) T counts dinucleotides across all sequences; where both ends must be A/C/G/T.
                C[a,b] ~ count of bigrams a->b (for T)

        3) A pseudocount of "self.pc" is added to every entry to avoid zeros.
              
        4) Rows are normalized to probabilities.
            π[b]       = (c0[b] + pc) / Σ_b' (c0[b'] + pc)
            T[a,b]     = (C[a,b] + pc) / Σ_b' (C[a,b'] + pc)


        Returns
        -------
        self

        """

        # Determine the number of letters in the alphabet (should be 4 for A/C/G/T)
        K = len(self.alphabet)

        # Initialize array of letters (4) filled with pseudocounts (self.pc) for initial counts
        c0 = np.full(K, self.pc)        

        # Initialize array of letter (4) with pseudocounts (self.pc) for bigram counts
        C  = np.full((K, K), self.pc)   

        # Loop over all sequences in the background corpus
        for s in sequences:

            # Uppercase the sequence
            s = (s or "").upper()

            # keep only A/C/G/T so positions align in bigrams
            s = ''.join(ch for ch in s if ch in self.a2i)

            # Skip empty sequences
            if not s:
                continue

            # Count first letter (for π) 
            c0[self.a2i[s[0]]] += 1

            # Count bigrams (for T)
            for a, b in zip(s[:-1], s[1:]):
                C[self.a2i[a], self.a2i[b]] += 1

        self.pi = c0 / c0.sum()
        self.T  = C / C.sum(axis=1, keepdims=True)
        return self

    # ------------------------------- Scoring --------------------------------------

    def logprob(self, seq):
        """
        Natural-log likelihood log P_bg(seq) under the fitted Markov chain.

        Skips any characters not in A/C/G/T (Ns are ignored). If all chars are invalid/empty, returns 0.0 (neutral).
        """
        if self.pi is None or self.T is None:
            raise RuntimeError("Call fit() before scoring.")
        s = (seq or "").upper()
        idx = [self.a2i[ch] for ch in s if ch in self.a2i]
        if not idx:
            return 0.0
        lp = math.log(self.pi[idx[0]])
        for i in range(1, len(idx)):
            lp += math.log(self.T[idx[i-1], idx[i]])
        return float(lp)

    def logprob_per_bp(self, seq):
        """
        Per-base log-likelihood: log P_bg(seq) / (# valid A/C/G/T in seq).

        This can be useful when you want length-invariant scaling. If you use
        per-bp normalization during calibration, use the same here.
        """
        s = (seq or "").upper()
        L = max(1, sum(ch in self.a2i for ch in s))
        return self.logprob(seq) / L

    # ----------------------- Z-score calibration and use --------------------------

    @staticmethod
    def _windows(seq, L, skip_N=True):
        """
        Yield all length-L windows of `seq` (uppercased).
        If skip_N=True, only yield windows comprised of A/C/G/T entirely.
        """
        s = (seq or "").upper()
        n = len(s)
        for i in range(0, n - L + 1):
            w = s[i:i+L]
            if skip_N and any(ch not in "ACGT" for ch in w):
                continue
            yield w

    @staticmethod
    def _gc_fraction(seq):
        """Return GC fraction in [0,1] for a DNA string (ignores non-ACGT)."""
        s = (seq or "").upper()
        acgt = [ch for ch in s if ch in "ACGT"]
        if not acgt:
            return -1.0
        gc = sum(ch in "GC" for ch in acgt)
        return gc / float(len(acgt))

    def calibrate_z_from_negatives(
        self,
        negatives,
        L,
        per_bp=False,
        gc_bin_edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        skip_N=True,
        winsor=None,
        max_windows_per_seq=None,
        rng=None,
    ):
        """
        Estimate μ and σ for log P_bg over windows of length L using the
        negative corpus, optionally per GC bin. Stores results in self.zparams.

        Arguments
        ---------
        negatives : iterable[str]
            GC-uniform negative sequences.
            These should be representative of the non-motif baseline.
        L : int
            Window length to calibrate (must match the length scored during
            scanning - typically the motif L from the detailed HMM).
        per_bp : bool, default False
            If True, calibrate on per-bp log-likelihoods; if False, on raw log
            likelihoods. Use the same choice later in zscore(...).
        gc_bin_edges : tuple[float], default (0, .2, .4, .6, .8, 1.0)
            Bin edges for GC fraction. If None or length < 2, only a global μ,σ
            is computed.
        skip_N : bool, default True
            Skip windows that contain any non-ACGT (so first-order model applies).
        winsor : float or None
            If set (e.g., 0.005), winsorize tails at that fraction for robustness.
        max_windows_per_seq : int or None
            Randomly subsample at most this many windows from each sequence to keep
            calibration tractable and reduce correlation from overlapping windows.
        rng : np.random.Generator or None
            Optional RNG for reproducible subsampling.

        Returns
        -------
        dict
            A dict with the stored parameters (same object as self.zparams).

        Rationale
        ---------
        The log-likelihood scales ~linearly with L. Therefore, the log-likelihood
        distribution shifts and widens with L. To make the penalty comparable across
        motifs of different lengths, we must re-calibrate μ, σ for each L.
    
        """
        if self.pi is None or self.T is None:
            raise RuntimeError("Call fit() before calibrating z-scores.")

        if rng is None:
            rng = np.random.RandomState()

        # Collect scores globally and per GC bin
        global_vals = []
        by_bin_vals = [[] for _ in range(max(0, len(gc_bin_edges) - 1))]

        def maybe_score(w):
            lp = self.logprob_per_bp(w) if per_bp else self.logprob(w)
            global_vals.append(lp)
            if len(by_bin_vals) > 0:
                g = self._gc_fraction(w)
                # find GC bin index
                for k in range(len(gc_bin_edges) - 1):
                    if gc_bin_edges[k] <= g < gc_bin_edges[k+1] or (k == len(gc_bin_edges)-2 and g == gc_bin_edges[-1]):
                        by_bin_vals[k].append(lp)
                        break

        # Iterate negatives; slide L windows; optionally subsample per sequence
        for seq in negatives:
            wins = list(self._windows(seq, L, skip_N=skip_N))
            if not wins:
                continue
            if max_windows_per_seq is not None and len(wins) > max_windows_per_seq:
                idx = rng.choice(len(wins), size=max_windows_per_seq, replace=False)
                wins = [wins[i] for i in idx]
            for w in wins:
                maybe_score(w)

        def _winsorize(x, p):
            if not x or p is None or p <= 0.0:
                return x
            lo = np.quantile(x, p)
            hi = np.quantile(x, 1.0 - p)
            return np.clip(x, lo, hi)

        # Global μ, σ
        gv = np.asarray(_winsorize(global_vals, winsor), dtype=float)
        mu = float(np.mean(gv)) if gv.size else 0.0
        sigma = float(np.std(gv, ddof=1)) if gv.size > 1 else 1e-9  # guard tiny σ

        zparams = {
            "L": int(L),
            "per_bp": bool(per_bp),
            "global": {"mu": mu, "sigma": sigma, "n": int(gv.size)},
            "by_gc": []
        }

        # Per-GC μ, σ (optional)
        if len(by_bin_vals) > 0:
            for k, vals in enumerate(by_bin_vals):
                arr = np.asarray(_winsorize(vals, winsor), dtype=float)
                mu_k = float(np.mean(arr)) if arr.size else mu
                sd_k = float(np.std(arr, ddof=1)) if arr.size > 1 else sigma
                zparams["by_gc"].append({
                    "lo": float(gc_bin_edges[k]),
                    "hi": float(gc_bin_edges[k+1]),
                    "mu": mu_k,
                    "sigma": max(sd_k, 1e-9),
                    "n": int(arr.size)
                })

        self.zparams = zparams
        return zparams

    def _pick_gc_bin_params(self, gc):
        """
        Helper: choose μ,σ for a given GC fraction based on self.zparams['by_gc'].
        Falls back to global if bins are missing.
        """
        if not self.zparams or not self.zparams.get("by_gc"):
            return self.zparams["global"]["mu"], self.zparams["global"]["sigma"]
        for e in self.zparams["by_gc"]:
            if e["lo"] <= gc < e["hi"] or (gc == 1.0 and e["hi"] == 1.0):
                return e["mu"], e["sigma"]
        return self.zparams["global"]["mu"], self.zparams["global"]["sigma"]

    def zscore(self, window, use_gc_bins=True):
        """
        Compute B_z for a *single* L-mer window using stored calibration.

        Arguments
        ---------
        window : str
            DNA string to score; length should match zparams["L"] if calibrated
            on raw log-likelihoods. If calibrated with per_bp=True, length can
            vary but we still recommend matching L used for μ,σ estimation.
        use_gc_bins : bool
            If True and per-GC bins are available, choose μ,σ by the window's GC.

        Returns
        -------
        float
            z = (log P_bg - μ) / σ   (or per-bp version if calibrated per_bp)

        Notes
        -----
        If σ is extremely small (near-deterministic negatives), we guard with a
        tiny σ to avoid blow-ups; this makes z ≈ 0 in such bins.
        """
        if self.zparams is None:
            raise RuntimeError("Call calibrate_z_from_negatives(...) before zscore().")
        per_bp = bool(self.zparams.get("per_bp", False))
        L_cal = int(self.zparams.get("L", 0))

        if not per_bp and len(window) != L_cal:
            # Optional: raise or warn. Here we just compute, but calibration won’t match perfectly.
            pass

        lp = self.logprob_per_bp(window) if per_bp else self.logprob(window)
        
        mu, sigma = (self._pick_gc_bin_params(self._gc_fraction(window))
                     if use_gc_bins else
                     (self.zparams["global"]["mu"], self.zparams["global"]["sigma"]))
        
        sigma = max(float(sigma), 1e-9)

        return (lp - float(mu)) / sigma

# ==============================================================================
#                   Search–Recognition Effective Model 
# ==============================================================================

def _compute_site_deltas(hits, s_max):
    """
    Convert Viterbi hits into dictionaries mapping each hit's position, score, and s_max value. 

    The function interprets the difference in Viterbi score for each length-L window i that can end in a valid motif state as an energy penalty: E(i) = ΔS = S-max - S(i) where:
        - S(i) is the joint log-probability (log P(path(i), window(i)) of the single most likely Viterbi path through the detailed-TFFM HMM for window i
        - S-max is the absolute maximum joint log-probability over any length-L path in the HMM model (i.e. the “perfect” binding site ceiling)

    ----------------------
    Args:
        hits (list[VHit]): A list (iterable) of NamedTuples (end, score, path) for every possible L-mer window in the sequence where
            - end (int):          The 0-based index of the window's terminal base
            - score (float):      The maximal joint log-probability of the Viterbi path through the window (∑logP(transition))
            - path (List[str]):   The state ID sequence for that path
            - sequence (str):     The nucleotide sequence of the L-mer window
            - strand (str):       The strand orientation of the window ("+" or "-")

        s_max (float): The maximum achievable joint log-probability over the HMM (precomputed).
        
    ----------------------
    Returns:

        sites_forward: List[dict]: entries for valid hits on the forward strand.
        sites_reverse: List[dict]: entries for valid hits on the reverse strand.

    ----------------------
    Note:
    ----------------------

        The conversion to a dictionary is necessary to associate other values to each hit downstream.
    """

    # Initialize lists to hold forward and reverse hits
    sites_forward = []
    sites_reverse = []

    for h in hits:

        # Extract all parameters from the site; Skip the hit if it is None or infinity
        score = getattr(h, "score", float("-inf"))
        if score == float("-inf") or score is None:
            continue
        pos = getattr(h, "end", None)
        sequence = getattr(h, "sequence", None)
        states = getattr(h, "path", None)
        strand = getattr(h, "strand", None)

        # Compute the delta score for the hit
        deltaS = float(s_max) - float(score)

        # Appned each hit's information to the sites list
        if strand == "+":
            sites_forward.append({
                "pos": int(pos),
                "score": float(score),
                "deltaS": float(deltaS),
                "sequence": sequence,
                "states": states,
                "strand": strand
            })

        elif strand == "-":
            sites_reverse.append({
                "pos": int(pos),
                "score": float(score),
                "deltaS": float(deltaS),
                "sequence": sequence,
                "states": states,
                "strand": strand
            })
        
        else:
            raise ValueError("Unexpected strand value: %s" % strand)

    return sites_forward, sites_reverse

def _sigmoid_clamped(x):
    """
    Transforms a given value to a Fermi-Dirac probability using the well known formula: Pbind(i) = 1 / (1 + exp[x])

    Args:
        x (float): The input value to the sigmoid function (Boltzman Weight). For a Fermi-Dirac function, this should take the form:

            x = 1/𝜆 * (ΔS(i) - S₀) 

            where both 𝜆 and S₀ are hyperparameters that need to be tuned to the data.

    Notes:

        The recognition probability uses the Fermi-Dirac form in score units: pR = 1 / (1 + exp(x)),  where x = 1/λ_R * (ΔSi - S0)

        This is a decreasing function of x (not the classical increasing logistic used in ML) where:
           • If ΔSi < S0  ⇒  xi < 0  ⇒  pR ≈ 1  (strong motif, above "chemical potential")
           • If ΔSi > S0  ⇒  xi > 0  ⇒  pR ≈ 0  (weak motif)

           Intuitively, since larger ΔSi should result in lower probabilities, this function is decreasing.

        The exponent is typically written as (S0 - ΔSi)/λR inside a Boltzmann e^{-βE} mapping.
            Here, (S - S0)/λR is just x → -x. We always pass the actual x to this helper function and branch by its sign at runtime.

        Since x can be positive or negative, we handle both cases by branching (switching formulas) to avoid numerical overflow or underflow.
    
            if x < 0:  pR = 1 / (1 + exp(x))                ; classical
            if x ≥ 0:  pR = exp(-x) / (1 + exp(-x))         ; branched

            This keeps all intermediate values between 0 and 1 (exp(x)∈(0,1] and exp(-x)∈(0,1]). 
            This helps with precision downstream as it reduces the relative error that occurs for exp(x) when x >> 0 - which would normally cause a huge value in the denominator for the classical formula (increasing relative error)

        """

    if x >= 0:
        ex = math.exp(-x)
        return ex / (1.0 + ex)
    else:
        ex = math.exp(x)
        return 1.0 / (1.0 + ex)

def _fermi_dirac_probs(
        sites, 
        lamR, 
        S0,
        bg_model=None,
        bg_weight=0.0,
        mu=0.0,
        sigma=1.0
    ):
    """
    Transforms each energy penalty into a probability for the recognition-state using a numerically-stable Fermi function: Pbind(i) = 1 / (1 + exp[1/𝜆 * (ΔS(i) - S₀)]) where:
        - 𝜆 controls steepness of the sigmoid transition
        - S₀ is the chemical potential of the Fermi-Dirac function (value at which there is 50% occupancy)

    This functions adds the recognition-state probability to each site dictionary. 

    --------------------------
    Args:

        sites (List[Dict[str, float]]): List of site dictionaries containing energy penalties.
        lamR (float): Steepness parameter for the Fermi function.
        S0 (float): Chemical potential for the Fermi function.
        bg_model (MarkovBG1 or None): A fitted MarkovBG1 background model. If None, no background correction is applied.
        bg_weight (float): Weight of the background correction term. If 0.0, no background correction is applied.
        mu (float): Mean of the background log-probabilities for z-score calculation. Used only if bg_model has no zparams.
        sigma (float): Standard deviation of the background log-probabilities for z-score calculation. Used only if bg_model has no zparams.

    --------------------------
    Returns:

        List[Dict[str, float]]: The input list of site dictionaries with added recognition-state probabilities.

    -----------------------
    Methods
    -----------------------

    1. Filter valid windows
            Discard any 'Hit' with score = −∞, since they either do not have a valid Viterbi path or do not end in a final state.

    2. Compute adjusted log-probability scores
            For each i ∈ V, compute ΔS(i) = S-max - S(i).  Note that ΔS(i) ≤ 0, and max-i ΔS(i) = 0 by construction.

    3. Exponentiate to Boltzmann weights 
            Compute Boltzman weight for each hit: w(i) = exp[E(i) - E0]. Since 1/𝜆*ΔS(i) ∝ E(i) and 1/𝜆*S0 ∝ E(0); w(i) = exp[1/𝜆 * (ΔS(i) - S0)] (refer to energy penalty above)

    4. Compute Fermi-Dirnac Probability
            Compute probability of binding for the 'Hit' as: 1 / (1 + w(i)])

    """
    # Convert lambda and S0 to float
    lamR = float(lamR)
    S0 = float(S0)

    # Check if background model is provided and if the weight is non-zero
    use_bg = (bg_model is not None) and (bg_weight != 0.0) 

    # Iterate over each site
    for site in sites:

        # Extract the energy penalty for the site
        deltaS = site["deltaS"]

        # Compute the exponent for the Fermi-Dirac function
        ER_raw = (deltaS - S0) / lamR

        if use_bg:

            if bg_model.zparams is not None:

                Bz = bg_model.zscore(site["sequence"])

                site["zscore_params"] = bg_model.zparams.get("global", None)

                if bg_model.zparams.get("by_gc", None):

                    site["zscore_params_by_gc"] = bg_model.zparams.get("by_gc", None)

                ER = ER_raw + (bg_weight * Bz) 

            else:

                Bi = bg_model.logprob(site["sequence"])

                Bz = (Bi - mu) / sigma
                
                ER = ER_raw + (bg_weight * Bz) 

            site["Bz"] = Bz if "Bz" in locals() else None

            site["BgLogProb"] = float(Bi) if "Bi" in locals() else None

        else:
            ER = ER_raw

        # Store the exponent as the recognition energy
        site["ER"] = ER 

        # Transform the exponent into a Fermi-Dirac probability
        pI = _sigmoid_clamped(ER)

        # Add the recognition probability to the dictionary for the site
        site["Pi"] = pI

    return sites

# ==============================================================================
#                   Tridiagonal (Thomas) solver utilities 
# ==============================================================================

def _thomas_solve(a, b, c, s):
    """
    Solve the tridiagonal linear system  A u = s using the Thomas algorithm (a specialized LU factorization for tridiagonals). 

    This function is used in the kinetic two-state TF model when evaluting, within a finite window along DNA, the probability that a transcription factor (TF) 
    binds somewhere along the finite window before dissociation (first-pass probability), given that it starts in the search configuration (state) at position i.

    The model essentially uses recursive propagation to accumulate binding probabilities from all possible paths through the lattice. 
    Each base contributes to the overall success probability either directly (via immediate recognition and binding) or indirectly (via sliding to neighboring bases and recursively accumulating their success probabilities).

    ------------------------------------------------------------------------------
    Problem Definition
    ------------------------------------------------------------------------------

    Consider a continuous-time two-state markov model on a 1D lattice of L bases (the window). 
    At each base i, the TF is in the Search state Si. From Si it can do one of the following:

        • Hop (slide) left to S{i-1} with rate ki-,
        • Hop (slide) right to S{i+1} with rate ki+,
        • Enter Recognition state at i with rate kSR(i),
        • Dissociate from the DNA (off) at i with rate koff(i).

    Let Λi = ki^- + ki+ + kSR(i) + koff(i) be the total exit rate from Si. The corresponding embedded (one-step) branching probabilities are as follows:

        ℓi-     = ki- / Λi      (left hop)
        ℓi+     = ki+ / Λi      (right hop)
        c_cap_i = kSR(i) / Λi   (“capture” into Recognition at i)
        off_i   = koff(i) / Λi  (dissociate to solution)

        Note ℓi- + ℓi+ + c_cap_i + off_i = 1 

    In Recognition state at i, binding (absorbing success) occurs with probability β, and return to Search (R → Si) occurs with probability γ = 1 − β.
    
    ------------------------------------------------------------------------------
    Tridiagonal Matrix Formulation
    ------------------------------------------------------------------------------

    Let ui be the conditional probability of binding somewhere within the window given the initial start of the search from base i
        
        P(“bind somewhere in this window before off” | start at Si). The phrase "start at Si" is the conditioning of the random walk.

    Perform a first-step decomposition at Si. After one embedded step, there are six possible scenarios that contribute to the success probability of binding in the window:

        • With probability ℓi^- the TF goes to S{i-1}, and the success probability is u{i-1}. 

        • With probability ℓi^+ the TF goes to S{i+1}, and the success probability is u{i+1}.

        • With probability c_cap_i the TF enters Recognition:

            - with probability β the TF binds immediately (success contributes 1),
            - with probability γ the TF returns to Search at i, and the success probability is again ui (the TF “restarts” at i).

        • With probability off_i the TF dissociates (contributes 0 to success).

    This embedded step can be used to derive a tridiagonal system of equations for the unknowns u_i.
     
        Success probability at i: ui = ℓi- u{i-1} + ℓi+ u{i+1} + γ c_cap_i u_i + β c_cap_i.

            This equation may be difficult to grasp. Consider that:

                1) Each u{i-1} and u{i+1} represents the probability of transitioning to the left or right then binding.
                    The contribution of this comes from the sliding probabilities ℓi- and ℓi+ and the corresponding success at the neighbouring bases.

                2) Each u{i} represents the probability of remaining at the current position and binding.
                    The contribution of this comes from c_cap_i and the success probability β.

        Move the unknown terms to the left to solve for binding: ui - γ c_cap_i u_i - ℓi- u{i-1} - ℓi+ u{i+1} = β c_cap_i. 

        Group the two ui terms: (1 - γ c_cap_i) ui - ℓi- u{i-1} - ℓi+ u{i+1} = β c_cap_i. 

    This is exactly a tridiagonal relation for each interior row i (matrix A: length L x L). Identifying:

        sub diagonal:               a[i] := - ℓi-,              where ℓ[i] is the probability of the TF sliding left (left hop); ki- / Λi 
        main diagonal:              b[i] := 1 - γ c_cap_i,      where c_cap_i is the probability of the TF transitioning to a recognition state; kSR(i) / Λi and γ is the probability of returning to search state; 1 - β
        super diagonal:             c[i] := - ℓi+,              where ℓ[i] is the probability of the TF sliding right (right hop); ki+ / Λi
        RHS vector:                 s[i] := β c_cap_i,          where c_cap_i is the probability of the TF transitioning to a recognition state; kSR(i) / Λi and β is the probability of binding upon recognition.

        Then: si = β c_cap_i = (Au)i = (ai u{i-1} + bi u{i} + ci u{i+1}),   i = 0, …, L-1.

        The LHS (matrix A) essentially captures all the recursive "keep searching" pathways (all slides to the left and right and returns from R).
        The RHS captures the "binding" pathways (the capture into recognition and binding upon recognition).

        The off-constant does not appear in the matrix as it contributes 0 to the success probability. However, it does reduce the fractions of ℓi± and c_cap_i via the denominator Λi.
          So it shrinks the off-diagonal entries a[i] and c[i] and pushes b[i] closer to 1. This makes the matrix more diagonally dominant, which stabilizes the thomas algorithm (see below)

    ------------------------------------------------------------------------------
    Boundary Conditions
    ------------------------------------------------------------------------------
        
    The matrix contains a reflecting boundary at both ends:

        • Reflecting left boundary: no hop beyond index 0 of the given window
            ℓ0- = 0 ⇒ a0 = 0.

        • Reflecting right boundary: no hop beyond index L-1 (last base) of the given window
            ℓ{L-1}+ = 0 ⇒ c{L-1} = 0.

    Also, consider the special case where β = 1 (immediate binding upon Recognition). In this case γ = 0 ⇒ b[i] = 1.
        The off-diagonals a[i], c[i] still couple neighbors via sliding, so the system remains coupled and must be solved using the Thomas algorithm,

    ------------------------------------------------------------------------------
    Algorithm Stability
    ------------------------------------------------------------------------------

    The diagonal matrix has 0 ≤ ℓi^±, c_cap_i, offi ≤ 1 and sum to 1, hence

        |ai| + |ci| = ℓi- + ℓi+ ≤ 1 - c_cap_i - offi. 

        The two-off diagonals are both strictly less than 1.

    Meanwhile bi = 1 - γ c_cap_i ≥ 1 - c_cap_i (since 0 ≤ γ ≤ 1).

        Therefore bi > |ai| + |ci| provided offi > 0 or (γ < 1 and c_cap_i > 0).

    Therefore, in all physical regimes used in our model, A is strictly diagonally dominant.
    Hence, we essentially have an diagonally dominant Z-matrix or a (non-singular) M-matrix.

    A Z-matrix is a matrix where the off-diagonal entries are non-positive and the diagonal entries are positive (remember ai and ci contain −ℓi±).
    An M-matrix is a matrix where the off-diagonal entries are strictly less than the diagonal entries, that is nonsingular (as we proved above).

    As such, the forward sweep denominators (pivots) are strictly positive and a unique solution exists and satisfies 0 ≤ u_i ≤ 1 for all i (see below).
    This means, the Thomas algorithm is appropriate for this system, providing a numerically stable solver.

    ------------------------------------------------------------------------------
    Algorithm Methodology
    ------------------------------------------------------------------------------

    The Thomas Algorithm computes modified coefficients c'i (effective super-diagonal) and d'i (effective RHS) by a single forward sweep.
    The forward sweep eliminates the sub-diagonal, producing an upper-triangular system. Then we recover u (success probability) by back substitution.

    At each step, we update the modified coefficients c'[i] and d'[i] based on the current row:

        Forward Sweep: i = 0:

            c'[0] = c0 / b0
            d'[0] = s0 / b0

        Forward Sweep: i = 1..L-1:

            den_i   = bi - ai * c'i-1                       The denominator is strictly > 0 in our M-matrix since bi is positive and ai/ci are negative
            c'i     = (ci / den_i) if i < L-1 else 0
            d'i     = (si - ai * d'i-1) / den_i

        Back substitution:

            u{L-1}  = d'{L-1}
            ui      = d'i - c'i * u{i+1},   for i = L-2, …, 0.

    Intuition:  During forward sweep, d'[i] accumulates contributions from all sources s[0..i] to the left through chains of the neighbor couplings (ie. the left-to-right elimination)
                During back substitution, each u[i] then absorbs the influence of all sources s[i..L] to the right through the u[i+1], u[i+2], …
                
                Hence each u[i] is a is a global linear combination of all sj across the window.
                Ie. the full sum over paths that eventually bind before off, starting from Si.

    ------------------------------------------------------------------------------
    Algorithm Example
    ------------------------------------------------------------------------------

    Assume an original system represented by a 4x4 tridiagonal matrix (with reflecting boundaries; a[1] and c[L] = 0 in 1-based indexing):

        [b0  c0   0   0] [u0] = [s0]
        [a1  b1  c1   0] [u1] = [s1]
        [ 0  a2  b2  c2] [u2] = [s2]
        [ 0   0  a3  b3] [u3] = [s3]

    During the forward sweep we calculate the modified coefficients as such:

        For i = 0:
            c'[0] = c0 / b0
            d'[0] = s0 / b0

        For i = 1:
            den1  = b1 - a1 * c'[0]
            c'[1] = c1 / den1
            d'[1] = (s1 - a1 * d'[0]) / den1

        For i = 2:
            den2  = b2 - a2 * c'[1]
            c'[2] = c2 / den2
            d'[2] = (s2 - a2 * d'[1]) / den2

        For i = 3:
            den3  = b3 - a3 * c'[2]
            c'[3] = 0
            d'[3] = (s3 - a3 * d'[2]) / den3

    During the backward substitution we solve for ui using the new upper-tridiagonal system consisting of c' and d' for each i:

            u3 = d'[3]
            u2 = d'[2] - c'[2] * u3
            u1 = d'[1] - c'[1] * u2
            u0 = d'[0] - c'[0] * u1
    
    ----------------------------------------------------------------------------
    Exponential Decay of Contributions
    ----------------------------------------------------------------------------

    There are two composite probabilities per visit at site i:

        • Absorption:   qi = β ci + oi                      (bind OR dissociate)
        • Survival:     σi = 1 - qi = ℓi^- + ℓi^+ + γ*ci    (hop OR return to search)

    Exponential decay of site contributions occurs when qi > 0 (i.e., σi < 1): Each step has a chance to absorb (either bind or off). 
    Over many steps, survival weights shrink geometrically (as the product of σ ; ∏σ). This geometric penalty beats the combinatorial growth in the number of hop sequences and enforces exponential damping with distance.

    To understand combinatorial path growth, consider a model with homogenous coefficients that do not depend on position. Here, we have symmetric hops (ℓi^- = ℓi^+ = ℓi).

        Although a single length n-path through this model has a weight ∝ (ℓ)^{n} (ie. it decays exponentially due to multiplicity of ℓ for each site; a specific length-n path has weight ∏ℓ over n-steps),
        there are combinatorially many such paths. The overall contributions from all paths offsets the multiplicative decay.

        For example consider a site at i=0 and a target site at i=2 - The site at position 2 can be reached from the start site (i=0) via multiple 4-length paths:

            RRRL = (ℓ)^4 
            LRRR = (ℓ)^4 
            RLRR = (ℓ)^4 
            RRLR = (ℓ)^4

            where L is a hop left; R is a hop right. Note that there are 16 total possible 4-length paths (4 of which can reach i=2).
            The sum of the probabilities for these 16 paths offset the multiplicative decay (see below for an example).

    Now consider our model. Remember, in the tridiagonal case, each branching probability is weighted by a sum including koff (although it does not appear on the RHS). 
    This enlarges Λ (shrinking ℓ and c) and results in the sum of each row i < 1. 

    Let L be the “continue” operator (no RHS) whose row i has [ℓi^-, γ ci, ℓi^+] in positions (i-1, i, i+1). Then the sum of each row is σi.
    This forces geometric decay. Consider the following.

        Imagine a tree of "keep going" possibilities (along the continue operator):

            After 1 step, at most a fraction σmax of the original probability is still continuing. 
            After 2 steps, at most a fraction σmax^2 of the original probability is still continuing.
            After L steps, at most a fraction σmax^L of the original probability is still continuing.

        So even though there are many different step sequence (combinatorial growth), the sum of their probabilities cannot exceed σmax^k. 

    If there is any absorption anywhere (qi > 0) then the sum of the row is less than 1 (σi < 1) forcing geometric decay.

        For example, consider a 3-length path (with only hops; ci = 0). Assume symmetric hops between R and L and an σmax = 0.9 - so R=0.45 and L=0.45.

            There are 8 paths in total. The probability of each path is as follows:

                RRR = (0.45)^3 * _______   = (0.45)^3  = 0.091125
                RLR = (0.45)^2 * (0.45)    = (0.45)^3  = 0.091125
                RRL = (0.45)^2 * (0.45)    = (0.45)^3  = 0.091125
                RLL = (0.45)^2 * (0.45)    = (0.45)^3  = 0.091125
                LLL = _______  * (0.45)^3  = (0.45)^3  = 0.091125
                LRL = (0.45)   * (0.45)^2  = (0.45)^3  = 0.091125
                LRR = (0.45)^2 * (0.45)    = (0.45)^3  = 0.091125
                LLR = (0.45)   * (0.45)^2  = (0.45)^3  = 0.091125

            The sum of the probability mass for the paths is 0.729. 

        To reach sites that are further away from i=0, a longer path length is required. 
        However, the maximum probability of this path length decreases exponentially, so the contribution of distant sites decreases as well.
            i.e, to contribute from a site j that is d = | j - i | bases away, a trajectory must take at least d steps - so its weight carries at least d survival factors (mass reach j at k ≤ σmax^k and k ≥ d)

    If there is no absorption anywhere (q_i = 0), then the sum of the row equal 1 (σ_i = 1) and the probability mass is conserved in each step.

        For example, again consider a 3-length path (with only hops; ci = 0). Assume symmetric hops between R and L and an σmax = 1 - so R=0.5 and L=0.5.

            There are 8 paths in total. The probability of each path is as follows:

                RRR = (0.50)^3 * _______   = (0.50)^3  = 0.125
                RLR = (0.50)^2 * (0.50)    = (0.50)^3  = 0.125
                RRL = (0.50)^2 * (0.50)    = (0.50)^3  = 0.125
                RLL = (0.50)   * (0.50)^2  = (0.50)^3  = 0.125
                LLL = _______  * (0.50)^3  = (0.50)^3  = 0.125
                LRL = (0.50)   * (0.50)^2  = (0.50)^3  = 0.125
                LRR = (0.50)^2 * (0.50)    = (0.50)^3  = 0.125
                LLR = (0.50)   * (0.50)^2  = (0.50)^3  = 0.125

            The sum of the probability mass for all the paths is 1.00
            Although each single path is less likely, the huge number of paths adds up to 1 again (there's no net loss with distance). So there is no decay and distant sites do not damp out.

    Args:
    --------------------

        a (list[float]): A list of sub-diagonal entries; with reflecting boundary - a[0] must be 0.

        b (list[float]): A list of main-diagonal entries; b[i] = 1 − γ c_cap_i (always > 0 in our model).

        c (list[float]): A list of super-diagonal entries; with reflecting boundary - c[L−1] must be 0.

        s (list[float]): A list of right-hand side entries; s[i] = β c_cap_i.

    Returns
    --------------

        u (list[float]): A list with the solution vector; u[i] = P(bind anywhere in the window before off | start at S_i).

    Raises
    --------------

    ValueError
        If the input arrays have mismatched lengths

    ZeroDivisionError
       If the forward-sweep denominator is zero (should not happen with diagonally dominant M-matrix (indicates invalid coefficients)

    """

    # --- 0) Dimension checks ---------------------------------------------------

    # Determine length of main diagonal
    L = len(b)

    # Ensure all tridiagonal coefficients and RHS have the same length
    if not (len(a) == len(c) == len(s) == L):
        raise ValueError("Tridiagonal arrays must have the same length.")

    # --- 1) Allocate modified coefficients (upper-triangular form) ------------
    
    # Initialize modified coefficients for the upper-triangular system     
    cprime = [0.0] * L # cprime[i] is the effective super-diagonal after eliminating the sub-diagonal up to row i
    dprime = [0.0] * L # dprime[i] is the effective RHS after the eliminating the sub-diagonal up to row i

    # --- 2) Initialize at row 0 ------------------------------------------------

    # Determine forward-sweep denominator for i=0
    denom = b[0]

    # Check for forward-sweep denominator being 0
    if denom == 0.0:
        raise ZeroDivisionError("Singular system at row 0 (b[0] == 0)")

    # Determine the modified coefficients for i=0
    cprime[0] = c[0] / denom    # c'0; Effective forward coupling to u[0]
    dprime[0] = s[0] / denom    # d'0; Effective RHS after dividing the first row by b[0]

    # --- 3) Forward sweep: Eliminate sub-diagonal one row at a time -----------
   
    # Iterate over each stacked row of the matrix
    for i in range(1, L):

        # Compute the modified denominator
        denom = b[i] - a[i] * cprime[i - 1]  

        # Check for singularity (denominator cannot be 0)
        if denom == 0.0:
            raise ZeroDivisionError("Singular system at row %s (denom == 0)." % i)

        # Compute the modified coefficients for the upper-triangular matrix 
        cprime[i] = (c[i] / denom) if i < L - 1 else 0.0    # last row has no super-diagonal
        dprime[i] = (s[i] - a[i] * dprime[i - 1]) / denom

    # --- 4) Back substitution: solve upper-triangular matrix ------------------

    # Initialize the solution vector
    u = [0.0] * L

    # Fill the solution vector with the effective RHS (used for chain-coupling below)
    u[L - 1] = dprime[L - 1]

    # Iterate backwards from second last row to fill in the solution vector
    for i in range(L - 2, -1, -1):

        # At each row i, we can compute u[i] as u[i] = dprime[i] - cprime[i] * u[i+1]  (for i=L-2..0), given u[L-1] = dprime[L-1]
        u[i] = dprime[i] - cprime[i] * u[i + 1] # The u[i] is updated at each step and used in the next iteration (as u[i+1])

    return u

# ============================================================
#             Global-rate Calibration Helpers
# ============================================================

def set_koff_from_rms(D1_bp2_per_s, s_target_bp):
    """
    Set dissociation rate k_off from a target 1D sliding RMS distance.
        RMS ≈ sqrt(2 D1 / k_off)  =>  k_off = 2 D1 / RMS^2

    Args:
        D1_bp2_per_s (float): 1D diffusion coefficient (bp^2 / s)
        s_target_bp  (float): target RMS sliding span for a single encounter (bp)

    Returns:
        float: k_off (1/s)
    """

    D1 = float(D1_bp2_per_s)

    s2 = float(s_target_bp) * float(s_target_bp)

    if s2 <= 0.0 or D1 <= 0.0:
        raise ValueError("D1 and s_target_bp must be positive.")
    
    return 2.0 * D1 / s2

def set_alpha0_from_attempts(koff, A_target):
    """
    Set the baseline S to R transition scale alpha0 so the mean number of
    recognition entries per encounter ≈ A_target:
        E[A] ≈ <k_SR>/k_off ≈ alpha0 / k_off  =>  alpha0 = A_target * k_off

    Args:
        koff (float): k_off (1/s)
        A_target (float): desired attempts per encounter (e.g., 0.3-1.0)

    Returns:
        float: alpha0 (1/s)
    """
    return float(A_target) * float(koff)

# ============================================================
#           Compute Threshold Values for Fermi Ramp
# ============================================================

def ramp_anchor_weights(Pi_window, theta_low=0.20, theta_high=0.35):
    """
    Linear ramp w_i in [0,1] from theta_low -> theta_high, else 0/1 outside.
    This wi multiplies the RHS (“productive binding upon recognition”) but
    leaves the movement (sliding/recognition entry) untouched.

    Args
    ----
    Pi_window : list[float]
        Recognition probabilities Π_i for the sites in a centered window.
    theta_low, theta_high : float
        Ramp start and end. Choose via calibration (see function below).

    Returns
    -------
    w : list[float]
        Same length as Pi_window. 0 if Π<=θ_low; 1 if Π>=θ_high; linear in-between.
    """

    # Convert thresholds to float
    tL = float(theta_low)
    tH = float(theta_high)

    # Check if thresholds are valid
    if tH <= tL:
        raise ValueError("theta_high must be > theta_low")
    
    # Initialize weights list
    w = []

    # Precompute inverse of the ramp width 
    inv = 1.0 / (tH - tL)

    # Iterate over each recognition probability in the window
    for pi in Pi_window:

        # If pi is below the lower threshold, weight is 0
        if pi <= tL:
            w.append(0.0)

        # If pi is above the upper threshold, weight is 1
        elif pi >= tH:
            w.append(1.0)

        # If pi is between the thresholds, weight is linearly interpolated
        else:
            w.append((pi - tL) * inv)

    return w

# ============================================================
#           Compute Directional coefficients 
# ============================================================

def _build_energy_directional_coeffs(
    Pi_window,
    ER_window,
    beta,                   # bind-upon-recognition probability (used in system="bind")
    Koff,                   # off-rate from Search (const by default, see use_const_koff)
    p,                      # coupling that maps recognition energy ER -> search energy ES = p*ER
    alpha0,                 # base S->R scale (will be reweighted by ER)
    G=0.0,                  # optional barrier/offset in k_SR exponent
    regional_rhs=True,      # if True, RHS is “anywhere in window”; if False, caller can inject a custom RHS later
    center_ES=True,         # median-center ES within the window for numerical robustness
    use_const_koff=False,   # if False, k_off scales with ES (less robust on genome negatives)
    system="bind",          # "bind"  : first-pass bind probability (RHS = β c_i, diag = 1 - γ c_i)
                            # "attempt": expected # of recognition entries (RHS = c_i,   diag = 1 - c_i)
    edge_censor=3e-4,       # small extra off-rate at the two edges (None to disable)
    c_edge_max=0.99,        # max capture prob at edges (to avoid trapping)
    use_ramp=False,         # if True, apply a Fermi ramp to the RHS (see below)
    theta_low=0.01,         # ramp start (if using ramp)
    theta_high=0.80,        # ramp end (if using ramp)
):
    """
    Build the tridiagonal (a,b,c) and RHS system for a single window, under an energy-aware sliding model that obeys detailed balance on a 
    Search-energy landscape ES that is built from ER. This routine is used right before the Thomas solve.

    Inputs (per position i in the window)
    -------------------------------------
    Pi_window : List[float]
        Recognition probabilities Π_i for reporting (not used to build rates).
    
    ER_window : List[float]
        Recognition energies for each index; ES is derived from ER via ES = p·ER.
    
    beta : float
        Bind-upon-recognition probability (0..1). Enters the "bind" system.
    
    Koff : float
        Baseline off-rate from Search. With use_const_koff=True, k_off ≡ Koff.
    
    p : float
        Coupling between ER and ES: ES = p * ER. Smaller p leads to flatter ES landscape.
    
    alpha0 : float
        Base scale for k_SR; k_SR,i = alpha0 * exp(0.5*(p-1)*ER_i - G).
    
    G : float
        Barrier/shift for recognition and search configuration transitions (applied only in k_SR exponent).
    
    regional_rhs : bool
        If True, RHS rewards success anywhere. Otherwise RHS=0 here; caller can set a custom target RHS.
    
    center_ES : bool
        If True, subtract a (pseudo-)median from ES before using it in hops and ES-dependent off.
   
    use_const_koff : bool
        If True, k_off is constant (=Koff). If False, k_off = Koff * exp(ES_i).
    
    system : {"bind","attempt"}
        "bind":     first-pass binding* probability u_i.
                        Equation: (1 - γ c_i) u_i - ℓ^-_i u_{i-1} - ℓ^+_i u_{i+1} = β c_i with γ = 1 - β
        "attempt:   expected number of recognition entries A_i before off.
                        Equation: (1 - c_i) A_i - ℓ^-_i A_{i-1} - ℓ^+_i A_{i+1} = c_i
                        (Recognition always returns to Search; β is effectively 0 here.)

    edge_censor : float or None
        Add a small extra k_off at both edges to reduce edge trapping.

    system : {"bind", "attempt", "dwell"}
        
    Returns
    -------
    a, b, c, s : lists of floats (length L)
        Tridiagonal coefficients - Sub-diagonal (a), main diagonal (b), super-diagonal (c), and RHS (s).
        Reflecting boundaries are enforced: a[0] = 0, c[L-1] = 0.

    Notes on construction
    ---------------------
    1) Search-energy ES:
           ES_i = p * ER_i      

       We enforce detailed balance for hops using ES:
           k_{i→i+1} ∝ exp(-0.5 (ES_{i+1} - ES_i)),
           k_{i→i-1} ∝ exp(-0.5 (ES_{i-1} - ES_i}).

    2) Embedded probabilities are built from rates:
           Λ_i   = k^-_i + k^+_i + k_SR,i + k_off,i
           ℓ^-_i = k^-_i / Λ_i
           ℓ^+_i = k^+_i / Λ_i
           c_i   = k_SR,i / Λ_i
           off_i = k_off,i / Λ_i
           
       (We don't place off_i explicitly in the linear system; it dilutes ℓ^± and c through Λ_i and thereby increases diagonal dominance. See above.)

    3) System switch ("bind" vs "attempt"):
       
       "bind":

            diag b_i = 1 - γ c_i, 
            RHS s_i = β c_i

       "attempt": 
            
            diag b_i = 1 - c_i,  
            RHS s_i = c_i
            
       Off-diagonals are identical in the two systems: a_i = -ℓ^-_i, c_i = -ℓ^+_i

    """

    # ------------------------------ 0) shape checks ------------------------------
    
    # Validate window length
    L = len(ER_window)
    # Recognition probabilities (Π) must align (same length) with ER
    if len(Pi_window) != L:
        raise ValueError("Pi_window and ER_window must have same length")

    # ------------------------------ 1) Build Search energy ES from recognition energy ER ------------------------------
    
    # Map ER -> ES with a single coupling p (dimensionless)
    ES = [p * float(ER_window[i]) for i in range(L)]

    # Optional (pseudo-)median centering; with odd L, ES_sorted[L//2] is the true median
    if center_ES and L > 0:
        med = sorted(ES)[L // 2]
        ES = [e - med for e in ES]

    # ------------------------------ 2) Arrhenius-consistent hop rates obeying detailed balance ------------------------------
    
    # Allocate left/right per-site hop rates on the Search landscape
    k_plus  = [0.0] * L  # rate i -> i+1
    k_minus = [0.0] * L  # rate i -> i-1

    # Enforce detailed balance via local ES differences:
    for i in range(L):

        # Check if i is not at the right edge
        if i < L - 1:

            # Compute energy difference to the right neighbor
            dE = ES[i + 1] - ES[i]

            # Compute the right hop rate using Arrhenius relation: k{i→i+1} ∝ exp(-0.5 (ES_{i+1} - ES_i))
            k_plus[i]  = math.exp(-0.5 * dE)

        # Check if i is not at the left edge
        if i > 0:

            # Compute energy difference to the left neighbor
            dE = ES[i - 1] - ES[i]

            # Compute the left hop rate using Arrhenius relation: k{i→i-1} ∝ exp(-0.5 (ES_{i-1} - ES_i))
            k_minus[i] = math.exp(-0.5 * dE)

    # ------------------------------ 3) S->R rate and off-rate --------------------------------------------
    
    # Recognition entry rate modulated by ER (ties S<->R landscape)
    k_SR = [float(alpha0) * math.exp(0.5 * (p - 1.0) * ER_window[i] - float(G)) for i in range(L)]
    
    if use_const_koff:
        # Create a constant off-rate array
        k_off = [float(Koff)] * L

    else:
        # Create an ES-dependent off-rate array (by multiplying the constant Koff by the site-specific search-energy factor exp(ES_i))
        k_off = [float(Koff) * math.exp(ES[i]) for i in range(L)]

    if edge_censor:

        # Add a small extra off-rate at both edges (prevents endless ping-pong near boundaries and improves diagonal dominance)
        k_off[0]     += float(edge_censor)
        k_off[L - 1] += float(edge_censor)

    # Cap the capture probability at the edges to avoid trapping (if requested)
    if c_edge_max is not None and (k_SR[0] / (k_minus[0] + k_plus[0] + k_SR[0] + k_off[0])) > float(c_edge_max):

        # Adjust the total exit rate at the left edge to enforce cprob[0] = c_edge_max
        Lambda_target = (k_SR[0] / c_edge_max)
        
        # Determine the maximum between 0 and the required increase in total exit rate (in case k_SR[0] is already below the target)
        k_add = max(0.0, Lambda_target - (k_minus[0] + k_plus[0] + k_SR[0] + k_off[0]))

        # Increase k_off[0] to achieve the target Lambda (since k_SR[0], k_minus[0], k_plus[0] are fixed)
        k_off[0] += k_add

    # ------------------------------ 4) Embedded-chain branching probabilities ------------------------------
    
    # Allocate per-site embedded probabilities: left hop, right hop, capture
    ell_m = [0.0] * L  # left  hop probabilities
    ell_p = [0.0] * L  # right hop probabilities
    cprob = [0.0] * L  # capture (S->R) probabilities

    # Normalize rates by total exit Λ_i to form probabilities
    for i in range(L):

        # Compute the total exit rate from site i (sum of all outgoing rates)
        Lambda = k_minus[i] + k_plus[i] + k_SR[i] + k_off[i]

        # Guard against non-positive total exit rate (should not happen with valid rates)
        if Lambda <= 0.0:
            raise ValueError("Non-positive total exit rate at index %s" % i)

        # Compute embedded branching probabilities by normalizing rates by total exit rate
        ell_m[i] = k_minus[i] / Lambda
        ell_p[i] = k_plus[i]  / Lambda
        cprob[i] = k_SR[i]    / Lambda
       
    # ------------------------------ 5) Assemble tridiagonal system (a,b,c) and RHS s ------------------------------
    
    # Allocate output tri-diagonal vectors and RHS
    a = [0.0] * L  # sub-diagonal (i couples to i-1)
    b = [0.0] * L  # main diagonal
    c = [0.0] * L  # super-diagonal (i couples to i+1)
    s = [0.0] * L  # RHS

    # Iterate over each position to fill in sub- and super-diagonal entries (off-diagonals are identical for both systems)
    for i in range(L):

        # Fill in sub- and super-diagonal entries with reflecting boundaries
        a[i] = -ell_m[i] if i > 0   else 0.0   # reflecting at left edge
        c[i] = -ell_p[i] if i < L-1 else 0.0   # reflecting at right edge

    # Determine system key (default to "bind" if None or invalid)
    sys_key = (system or "bind").lower()

    # ------------------ 6a) First-Pass Bind Probability: RHS and Diagonal ------------------------------

    if sys_key == "bind":

        # Compute ramp weights if requested (else all 1.0); these weights wi multiply β in the RHS only
        w = ramp_anchor_weights(Pi_window, theta_low, theta_high) if use_ramp else [1.0]*L

        # Iterate over each position to fill in diagonal and RHS
        for i in range(L):

            # Compute effective bind-upon-recognition probability at this position
            beta_eff = float(beta) * w[i]

            # Compute effective return-to-search probability at this position (complement of beta_eff)
            gamma_eff = 1.0 - float(beta_eff)

            # Fill in diagonal entry
            b[i] = 1.0 - (gamma_eff * cprob[i])

            # Fill in RHS entry (only if binding is allowed at this position)
            if regional_rhs:
                s[i] = float(beta_eff) * cprob[i] 
            else:
                s[i] = 0.0

    # ------------------ 6b) Expected Attempts Before Off: RHS and Diagonal -----------------------------

    elif sys_key in ("attempt", "dwell"):
      
        # Iterate over each position to fill in diagonal and RHS
        for i in range(L):

            # Fill in diagonal entry
            b[i] = 1.0 - cprob[i]

            # Fill in RHS entry 
            s[i] = cprob[i] if regional_rhs else 0.0

    else:
        raise ValueError("Unknown system='%s'. Use 'bind' or 'attempt'." % system)

    return a, b, c, s
    
# -------------------------------------------------------
# 4) Window helpers + start prior
# -------------------------------------------------------

def _window_indices_centered(n, center_idx, Lwin):

    # Half-width in indices; we want Lwin total indices: [center - half, ..., center + half]
    half = Lwin // 2

    # Propose raw [lo, hi) bounds around center
    lo = center_idx - half
    hi = center_idx + half + 1

    # If lo underflows, shift the interval right to keep length Lwin
    if lo < 0:
        hi += -lo
        lo = 0

    # If hi overflows, shift left to keep length Lwin (clip lo at 0)
    if hi > n:
        shift = hi - n
        lo = max(0, lo - shift)
        hi = n

    # Return inclusive-exclusive bounds [lo, hi)
    return lo, hi

def _gaussian_start_prior(L, sigma_bp):

    # Center index for the discrete Gaussian (integer center in [0..L-1])
    c = L // 2

    # Container for unnormalized weights
    w = []

    # Guard against non-positive sigma to avoid division by zero
    sigma = float(sigma_bp)
    if sigma <= 0.0:

        # Fall back to a delta at center if sigma is invalid
        w = [0.0] * L
        w[c] = 1.0
        return w
    
    # Precompute inverse of 2σ^2 for speed/stability
    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    # Build unnormalized weights exp(-(i-c)^2 / (2σ^2))
    for i in range(L):
        w.append(math.exp(- (i - c) * (i - c) * inv2s2))

    # Normalize to sum to 1 with high-precision summation
    Z = math.fsum(w)
    return [wi / Z for wi in w]


# -------------------------------------------------------
#           5) Main pipeline For Dwell
# -------------------------------------------------------

def _log_mean_exp(values):
    """
    Compute the log-mean-exp of a list of values for numerical stability.
    log-mean-exp(x) = log( (1/N) ∑ exp(x_i) ) = logsumexp(x) - log(N)

    Args:
        values (list[float]): List of values to compute log-mean-exp

    Returns:
        float: log-mean-exp of the input values
    """
    if not values:
        raise ValueError("Input list is empty.")

    # Compute maximum value
    max_val = max(values)

    log_mean_exp = max_val + math.log(math.fsum(math.exp(v - max_val) for v in values) / len(values))

    return log_mean_exp


def _calibrate_alpha0(ER_window, p_reference, Koff):
    """
    Compute baseline S->R transition scale alpha0 so that the typical S->R hazard is commensurate with Koff:
        E[k_SR] ≈ alpha0 * E[exp(-0.5*(1 - p_ref)·ER)]  =>  alpha0 = Koff / E[exp(-0.5*(1 - p_ref)·ER)]

    Args:
        ER_window (list[float]): List of recognition energies in the window
        p_reference (float): Reference coupling of search and recognition states for calibration (e.g., 0.3)
        Koff (float): Off-rate from Search (1/s)

    Returns:
        float: alpha0 (1/s)
    """

    ER_max = max(ER_window)

    v = [ -0.5 * (1.0 - float(p_reference)) * e for e in ER_window]
    
    logA = _log_mean_exp(v)

    return float(Koff) * math.exp(-(1.0 - float(p_reference)) * ER_max - logA)

def dwell_pipeline(
    hits,
    s_max,
    lam_R = 1.4,
    S0 = 4.0,
    beta = 0.35,                
    Koff  = None,
    p = 0.20,                 
    sliding_window_bp = 200,
    start_prior = "uniform",
    start_prior_sigma_bp = 45,
    regional_rhs = True,
    bg_model = None,
    bg_weight = 1.5,
    center_ES = False,           
    use_const_koff  = False,
    alpha0 = None,
    G = None,
    system_key = "bind",
    D1_bp2_s   = 3e5,           # 1D diffusion in bp^2/s 
    s_target   = 75.0,          # target RMS sliding span (bp)
    A_target   = 4,              # expected recognition entries per encounter
    theta_low = 0.05,
    theta_high = 0.80,
    use_ramp = True,
):
    """
    Compute regional first-passage binding probabilities per site by centering a basepair window and solving an M-matrix tridiagonal system.

    ------------------------------------------
    Pipeline (per strand)
    ------------------------------------------

    1) Convert Viterbi hits to a site dicts with ΔS = S_max - S(i).

    2) Map each ΔS to a recognition energy (ER) and probability Π via Fermi-Dirac - Πi = 1 / (1 + exp(ER))  

        Here, ER is an uncorrected energy that can be adjusted by a background model. 
        If a background model is provided, we compute a z-score for each site's background log-probability and penalize ER accordingly.

            Bi = logprob_bg(sequence_i)
            Zi = (Bi - μ) / σ
            ER   = ER_raw + w·Zi
           
    3) Sort sites by genomic position for windowing.

    4) For each center site, extract a basepair window of specified length Lwin (odd integer) using a half-size on each side.
     
    5) Build tridiagonal matrix Au = s where u is the unknown vector of binding probabilities starting from each site in the window.

    6) Solve the tridiagonal matrix with Thomas algorithm 

    7) Apply a start prior over sites (uniform or discrete Gaussian) and report the regional dwell probability:
           P_bind_any = ∑i prior[i] · u[i]

    --------------------------------------------
    Auto-Calibration of α0
    --------------------------------------------
    
    If alpha0 is not provided (None), we auto-calibrate it so that the typical S→R hazard (averaged over the window's ER distribution) is commensurate with Koff (the dissociation off rate from DNA).
    
        α0 ≈ Koff / E[exp(-0.5*(1 - p_ref)·ER)]

    The idea is that the TF should be able to dissociate from the DNA at roughly the same rate as it transitions into a recognition state (averaged over the sites in the window). 
     
    Returns
    --------------------------------------------
    List[dict], List[dict]: A tuple of two lists of site dictionaries (one for each strand) with the corresponding keys for each site in the given sequence:
    
          "position", 
          "strand",
          "effective probability",                  # regional dwell probability
          "cluster score",                          # dwell x local ∑Π
          "background model",                       # log P_bg(center seq) or None
          "recognition probability (Pi_center)",
          "start prior", 
          "start prior param",
          "window_length",
          "states", "sequence"
        
    """

    # --- 0) Compute Delta Energy ΔS and Fermi-Dirac Probabilities Π  -----------------------------------

    # Convert Viterbi hits to per-site dicts with ΔS etc.
    sites_fwd, sites_rev = _compute_site_deltas(hits, s_max)

    # Map the ΔS to a recognition energy ER and Fermi-Dirac probability Π 
    sites_forward = _fermi_dirac_probs(sites_fwd,  lamR=lam_R, S0=S0,
                                    bg_model=bg_model, bg_weight=bg_weight,
                                    )
    
    sites_reverse = _fermi_dirac_probs(sites_rev, lamR=lam_R, S0=S0,
                                    bg_model=bg_model, bg_weight=bg_weight,
                                    )

    # --- 0.5) Calibrate Global Constants ----------------------------
 
    def _calibrate_delta_G(ER, p_reference):
        """
        Compute transition barrier G for kSR

        Args:
            ER (list[float]): List of recognition energies in the sequence
            p_reference (float): Reference coupling of search and recognition states for calibration (e.g., 0.3)

        Returns:
            float: G (dimensionless): Barrier/offset for k_SR exponent
        """

        # Retrieve the worst (highest) recognition energy in the sequence
        E_worst = max(ER)                       

        # Compute G so that the worst site has the same search and recognition energies
        G = -(1.0 - float(p_reference)) * float(E_worst)     

        return float(G)
    
    # Compute Koff based on desired RMS sliding distance
    Koff = set_koff_from_rms(D1_bp2_per_s=D1_bp2_s, s_target_bp=s_target)

    # Compute alpha0 based on desired attempts per encounter
    if alpha0 is None:
        a0 = set_alpha0_from_attempts(koff=Koff, A_target=A_target)
    else:
        a0 = alpha0

    # --- 1) Strand processor ---------------------------------------------------

    def _process_one_strand(sites):
        if not sites:
            return []

        # Sort by genomic position (ensures window slices are local)
        sites_sorted = sorted(sites, key=lambda d: d["pos"])
        n = len(sites_sorted)

        # Enforce odd window length for symmetric centering
        Lwin = int(sliding_window_bp)
        if Lwin % 2 == 0: 
            Lwin += 1

        # Pull Π and ER vectors aligned to sorted sites
        Pis       = [s["Pi"]  for s in sites_sorted]
        ERs       = [s["ER"]  for s in sites_sorted]

        # Pick G (if None) and α0 (if None)
        DeltaG = G if G is not None else _calibrate_delta_G(ERs, p_reference=0.3)

        a0 = alpha0 if alpha0 is not None else _calibrate_alpha0(ERs, p_reference=0.3, Koff=Koff)

        out = []

        # Slide the window center across all positions
        for idx, site in enumerate(sites_sorted):

            # Retrieve the indices for the window around the site
            lo, hi = _window_indices_centered(n, idx, Lwin)
          
            # Slice the probability and energy lists based on the window indices
            Pi_w = Pis[lo:hi]
            ER_w = ERs[lo:hi]

            # Define the length of the window
            L    = len(Pi_w)

            # Build start prior over indices in this window
            if start_prior == "gaussian":
                pi = _gaussian_start_prior(L, sigma_bp=float(start_prior_sigma_bp))
            else:
                pi = [1.0/L]*L

            # --- Solve for bind-probability (dwell) ---------------------------

            if system_key == "bind":
                a, b, c, s_vec = _build_energy_directional_coeffs(
                    Pi_w, ER_w,
                    alpha0=a0, beta=beta, Koff=Koff, p=p, G=DeltaG,
                    regional_rhs=regional_rhs,
                    center_ES=center_ES, use_const_koff=use_const_koff,
                    system="bind", 
                    theta_low=theta_low, theta_high=theta_high, use_ramp=use_ramp
                )

                u = _thomas_solve(a, b, c, s_vec)

                # Regional dwell probability (bind-anywhere in window)
                P_bind_any = sum(pi[i] * u[i] for i in range(L))

            # --- Solve for expected attempts (recognitions) -------------------

            elif system_key == "attempt":

                a, b, c, s_vec = _build_energy_directional_coeffs(
                    Pi_w, ER_w,
                    alpha0=alpha0, beta=0.0, Koff=Koff, p=p, G=DeltaG,
                    regional_rhs=regional_rhs,
                    center_ES=center_ES, use_const_koff=use_const_koff,
                    system="attempt", theta_low=theta_low, theta_high=theta_high, use_ramp=use_ramp
                )

                A = _thomas_solve(a, b, c, s_vec) 

                # Regional expected attempts (recognitions anywhere in window)
                A_any = sum(pi[i]*A[i] for i in range(L))
            
            # Package the per-site report with both dwell & attempts
            out.append({
                "position": int(site["pos"]),
                "strand":   str(site["strand"]),
                "z-score param": str(site.get("zscore_params", None)),
                "dwell": float(P_bind_any) if system_key == "bind" else float(A_any) if system_key == "attempt" else None,
                "cluster score": None,
                "background model": str(site["BgLogProb"]) if site["BgLogProb"] is not None else str(site["Bz"]),
                "recognition probability (Pi_center)": float(site["Pi"]),
                "start prior": "gaussian" if start_prior == "gaussian" else "uniform",
                "start prior param": float(start_prior_sigma_bp) if start_prior == "gaussian" else 1.0/max(1,L),
                "window_length": L,
                "states": str(site.get("states", None)),
                "sequence": str(site.get("sequence", None)).upper() if site.get("sequence", None) else None
            })

        return out

    return _process_one_strand(sites_forward), _process_one_strand(sites_reverse)

def _extract_pi_from_site(site):
    
    key = "recognition probability (Pi_center)"

    pI = float(site.get(key, 0.0))

    return pI

def _p_anchor_from_band(Pi_list, mode="any", top_k=3, min_pi=0.10, weights=None):
    """
    Combine anchor recognition probabilities Pi_j in a band.
    mode:
      - "any": noisy-OR over all anchors >= min_pi
      - "max": use max Pi
      - "topk": noisy-OR over top_k anchors by Pi
    """
    anchors = [p for p in Pi_list if p >= float(min_pi)]
    if not anchors:
        return 0.0
    if mode == "max":
        return max(anchors)
    if mode == "topk":
        anchors = sorted(anchors, reverse=True)[:int(top_k)]
        prod = 1.0
        for p in anchors:
            prod *= (1.0 - p)
        return 1.0 - prod
    # default "any"
    prod = 1.0
    for p in anchors:
        prod *= (1.0 - p)
    return 1.0 - prod

def _compound_with_dwell(D_dwell, p_single_any, kappa=1.0, alpha=1.0):
    """
    Turn “more local attempts because of dwell” into a total success probability.
    - Poisson attempts (clean, 1 parameter):
         P_total = 1 - exp(-kappa * D_dwell * p_single_any)
    """
    p = max(0.0, min(1.0, float(p_single_any)))
    D = max(0.0, float(D_dwell))
   
    return 1.0 - math.exp(-(kappa * D) * p)

def add_anchored_scores_for_band(
    sites,
    band_bp=650,                # look ±band_bp for anchors
    min_anchor_pi=0.10,         # ignore extremely weak anchors
    exclude_center=True,        # exclude the very center as its own anchor
    kappa=2.50,                 # scale linking dwell to expected attempts
    dwell_bp=15,               # donut size to exclude around center
):
    """
    Post-process the per-site results produced by effective_site_probabilities(...).
    For each center site i (with a computed 'effective probability' = regional dwell),
    look ±band_bp and collect anchors' Pi values. Compute:
        p_anchor_any = 1 - ∏(1 - Pi_j)
        P_total      = compound(D_dwell_i, p_anchor_any)

    Mutates each site dict to add:
        - "p_anchor_any"
        - "anchored_total"   (final anchored probability for that center)
        - "anchors_in_band"  (count used)
    """
    if not sites:
        return sites

    # Sort by position index so that index ranges align with local genome order
    sites_sorted = sorted(sites, key=lambda d: d["position"])
    
    # Collect Π in parallel for speed
    pis       = [_extract_pi_from_site(s) for s in sites_sorted]
    n = len(sites_sorted)

    # Convert band/donut from bp to site indices (assume 1bp per site)
    band = int(band_bp)
    donut = int(dwell_bp)

    # Walk each center index
    for idx in range(n):

        lo, hi = _window_indices_centered(n, idx, 500)

        anchors = []

        for j in range(lo, hi):
            
            if exclude_center and abs(j - idx) <= 20:
                continue

            pj = pis[j]

            if pj >= float(min_anchor_pi):
                anchors.append(pj)

        p_anchor_any = _p_anchor_from_band(anchors, mode="max", top_k=3, min_pi=min_anchor_pi)
        D_dwell      = float(sites_sorted[idx]["dwell"])
        anchored_final = _compound_with_dwell(D_dwell, p_anchor_any, kappa=kappa)
        number_of_anchors = len(anchors)

        # # window bounds in site index space
        # lo = max(0, idx - band)
        # hi = min(n, idx + band + 1)

        # # # Compute inner exclusion zone around center 
        # if exclude_center:
            
        #     # Left anchors end before center; right anchors start after center
        #     left_lo,  left_hi  = max(lo, idx - donut), idx
        #     right_lo, right_hi = idx + 1, min(hi, idx + donut + 1)

        # else:

        #     # No exclusion: consider full [lo, idx) and (idx, hi)
        #     left_lo,  left_hi  = lo, idx
        #     right_lo, right_hi = idx + 1, hi

        # def _side_anchor(lo, hi, pis):
        #     """
        #     Collect anchors on one side (left or right) of the center site.
            
        #     Args:
        #         idx_range: range object with indices to consider as anchors
        #         pis: list of recognition probabilities aligned to sites_sorted
            
        #     Returns:
                
        #         anchored: final anchored probability for this side
        #         p_anchor_any: single-attempt success probability from anchors
        #         number_of_anchors: count of anchors used (for diagnostics)
        #     """

        #     # Initialize anchor list
        #     anchors = []
        
        #     # Iterate over the index range and collect anchors above threshold
        #     for j in range(lo, hi):

        #         # Index the anchor probability from the pre-extracted list using the site's index
        #         pj = pis[j]

        #         # Append the anchor to the list if it meets the minimum Pi threshold
        #         if pj >= float(min_anchor_pi):
        #             anchors.append(pj)

        #     # Combine anchors to form single-attempt success probability
        #     p_anchor_any = _p_anchor_from_band(anchors, mode="max", top_k=3, min_pi=min_anchor_pi)
            
        #     # Retrieve the center site's dwell (effective probability)
        #     D_dwell      = float(sites_sorted[idx]["dwell"])

        #     # Compound attempts and per-attempt success into final anchored probability
        #     anchored     = _compound_with_dwell(D_dwell, p_anchor_any,
        #                                         kappa=kappa)
            
        #     # Determine number of anchors used (for diagnostics)
        #     number_of_anchors = len(anchors)

        #     return anchored, p_anchor_any, number_of_anchors

        # # Process left and right sides separately
        # anchored_left, p_anchor_left, n_anchors_left   = _side_anchor(left_lo, left_hi, pis)
        # anchored_right, p_anchor_right, n_anchor_right  = _side_anchor(right_lo, right_hi, pis)

        # # Choose the side with the higher anchored probability
        # if anchored_left >= anchored_right:
        #     anchored_final = anchored_left
        #     p_anchor_any   = p_anchor_left
        #     anchors        = n_anchors_left
        # else:
        #     anchored_final = anchored_right
        #     p_anchor_any   = p_anchor_right
        #     anchors        = n_anchor_right

        # Mutate the site dict in place to add the new keys
        sites_sorted[idx]["p_anchor_any"]   = float(p_anchor_any)
        sites_sorted[idx]["anchored_total"] = float(anchored_final)
        sites_sorted[idx]["anchors_in_band"]= int(number_of_anchors)

    return sites_sorted


# ==============================================================================
#                   Compute Local Cluster Scores
# ==============================================================================

def thermodynamic_cluster_binding_score(site_probs, window_size=200, sequence_length=None, min_dist=5):
    """
    Computes a promoter-level transcription factor (TF) binding score by scanning for localized clusters of high-probability TF binding sites (TFBS), while enforcing a minimum spacing constraint between sites.

    This function implements a deterministic 1D sliding-window approach over the promoter sequence, evaluating each window for the presence of homotypic clusters of high-probability sites. 
    This simulates energy wells formed by adjacent high-affinity sites that can increase TF residency time - while avoiding overcounting of nearby motif hits that are sterically or competitively incompatible due to spacing constraints.

    Within each window, a dynamic programming algorithm identifies the optimal subset of non-overlapping binding sites (separated by at least "min_dist" base pairs), maximizing the total cumulative binding probability. 

    ------------------------
    Args:

        site_probs (list of tuples): A list of the site, score, and ferni-dirac probability for each 'hit' (end, score, P_bind)

        window_size (int): Size of local window to assess clustering (default = 100 bp).

        sequence_length (int): Total length of the promoter sequence. If None, inferred from max position.

        min_dist (int): Minimum distance (in base pairs) required between adjacent binding sites within a cluster (default = 5 bp).

    ------------------------
    Returns:

        local_scores (list of floats): Binding score per window.

        global_binding_score (float): Sum of weighted cumulative probability over all windows (proxy for total TF recruitment potential).

    ------------------------
    Method Summary:
    -------------------------

        Step 1: Hit Extraction:
            
            For each sliding window of fixed size (e.g., 400 bp), we first extract all motif hits whose genomic coordinates fall within the window. 
            Each hit is represented by a tuple containing: i) the genomic position (end index of the motif match), ii) the raw motif score (used optionally), and iii) the binding probability Pbind derived via a Fermi-Dirac function.

        Step 2: Positional Sorting:

            Hits are sorted in ascending order of their genomic position, allowing the algorithm to process motif candidates from 5' to 3' (left to right along the sequence). 
            This sorting is needed to allow the forward construction of optimal scoring configurations via dynamic programming.

        Step 3: Legal Predecessor Mapping (prev[i]):

            To enforce the minimum spacing constraint between TFBS, we define for each motif hit i its most recent legal predecessor - the last site to the left of i whose position is at least "min_dist" base pairs upstream.
            
            This mapping is stored in an index array prev[i], where the default value is -1 (indicating no legal predecessor) and:
                If prev[i] = j, it means that hit j is the last site before i that satisfies the spacing requirement.
                If no such site exists (e.g., i is the first hit or all previous hits are too close). In this case prev[i] = -1.

            This mapping ensures that if site i is included in the final cluster, no other site between prev[i]+1 and i-1 can be included in the dynamic programming, as all would violate the spacing constraint. 
            Thus, prev[i] defines the most recent compatible state from which we can legally extend our cluster to include i.

        Step 5: Dynamic Programming

            The primary focus of the algorithm is the dynamic programming array dp[i], where each entry stores the maximum total binding score achievable using motif hits up to and including index i.

        Step 6: Inclusion vs. Exclusion Decision

            For each motif hit i (processed left-to-right), we evaluate two mutually exclusive options:

                a. Exclusion: We skip the current site i and inherit the best score computed so far (up to i-1) (i.e. we do not include site i in our cluster)   
                    i) This option is chosen when including site i does not lead to a better score, or when doing so would violate spacing constraints and prevent the inclusion of more valuable hits later (see below).

                b. Inclusion: We include the current site i in the cluster, and add its binding probability to the optimal score up to its legal predecessor (prev[i]), thereby building a non-overlapping, high-probability subset.
                    i) This represents the cumulative binding potential of a legal configuration ending with site i.

        Step 7: Recurrence and Optimal Choice

            Recurrence and Optimal Choice: At each step i, we update the DP table depending on which option yields a higher score: inclusion or exclusion.
            This recurrence ensures that at each index, the algorithm stores the globally optimal solution for all motifs from position 0 up to i, without violating spacing constraints.

        Step 8: Final Score Extraction
            After processing all motif hits in the current window, the final element of the DP array (dp[-1]) holds the maximum cumulative score of a non-overlapping subset of TFBS for that window. 
            This value represents the optimal binding potential within the defined cluster.

    -------------------------
    Notes on Scoring Logic:
    -------------------------

     As it may be difficult to intuitively grasp why the inclusion of a site may not always yield a higher cumulative score in dynamic programming, we provide additional context below.

        The final position of the dynamic programming array is the maximum cumulative score of a non-overlapping subset of TFBS up to position i. 
            Therefore, it is possible that this final position is between i and prev[i] such that it violates the spacing constraint if site i were to be included. So, if we want to include site i, we cannot build on dp[i-1] and must instead go back to the last legal predecessor (prev[i]) in the dp array to build on the score.
        
        In this case, for site i to be included, the score computed from Pbind(i) + the dynamic score of the closest legal predecessor (dp[prev[i]]) would have to exceed the score computed from dp[i-1] (the best score up to i-1).
            If dp[i-1] is greater than Pbind(i) + dp[prev[i]], it indicates that the best score up to i - 1 already incorporates high-probability sites (in a legal, spaced configuration) and the inclusion of site i would displace these and lead to a lower overall score. 
            
   """

    # Return no value if the site probability list is empty
    if not site_probs:
        return [], 0.0

    # Determine the sequence length based on maximum end index of 'hits' (0-based end position)
    if sequence_length is None:
        sequence_length = max(s['position'] for s in site_probs) + 1

    # Sort site_probs by end position
    site_probs_sorted = sorted(site_probs, key=lambda x: x["position"])

    # Create an list of positions and probabilities for fast access
    positions = [hit["position"] for hit in site_probs_sorted]
    probabilities = [hit["effective probability"] for hit in site_probs_sorted]

    # Initialize a list for local cluster scores per sliding window
    local_scores = []

    # Iterate over each possible cluster of site probabilities (performs sliding window to find max cluster probability)
    for window_start in range(0, sequence_length - window_size + 1):

        # Define the end of the current window
        window_end = window_start + window_size

        # Create a tuple of position and probability for all hits within current window
        hits_in_window = [(pos, p) for pos, p in zip(positions, probabilities) if window_start <= pos < window_end]

        # If no hits in the current window, append 0.0 to local scores and continue to next window
        if not hits_in_window:
            local_scores.append(0.0)
            continue

        # Compute the number of hits in the current window (should equal window size)
        n = len(hits_in_window)

        # Initialize a dynamic programming list to select optimal subset with minimum distance constraint
        dp = [0.0] * n

        # Initialize a previous index list to track the closest compatible site for each hit (default to -1)
        prev = [-1] * n

        # Iterate over each hit in the current window to fill the previous index array (outer loop)
        for i in range(n):

            # Iterate backwards from the current hit to find a previous hit that satisfies the distance constraint
            for j in range(i - 1, -1, -1):

                # Check if the distance between hits is greater than or equal to min_dist
                if hits_in_window[i][0] - hits_in_window[j][0] >= min_dist:

                    # If so, store the previous hit at the index of i in the prev array
                    prev[i] = j

                    break

        # Iterate over each hit in the current window to fill the dynamic programming array (inner loop)
        for i in range(n):

            # Compute the score of the current hit (hits_in_window[i][1] is the probability of the hit)
            include = hits_in_window[i][1]

            # If there is a previous hit that is at least min_dist apart add its score to the current score
            if prev[i] != -1:

                # Retrieve the dynamic score of the closest legal previous hit for i (ignoring hits between prev[i] and i in cluster) and add it to the current score
                include += dp[prev[i]]

            # Compute the score for excluding the current hit position (if this is the first hit, exclude score is 0)
            exclude = dp[i - 1] if i > 0 else 0.0
            
            # Store the maximum of either including or excluding the current hit
            dp[i] = max(include, exclude)

        # Retrieve the maximum score from the last entry in the dynamic programming array
        max_sum = dp[-1]

        # Append the optimized local score for the current window 
        local_scores.append(max_sum)

    return local_scores

def thermodynamic_cluster_binding_score_no_min(site_probs, window_size=200, sequence_length=None):
    """
    Computes a promoter-level binding score by scanning for local homotypic clusters of high-probability TFBS with no spacing constraints.

    Implements a deterministic 1D walk where windows of fixed size are assessed for cumulative binding potential.
    This simulates energy wells formed by adjacent high-affinity sites that can increase TF residency time.

    This function is equivalent to the thermodynamic_cluster_binding_score function when min_dist=0. However, it is computationally more efficient as it does not require solving an optimal subset selection problem.

    Args:
        site_probs (list of tuples): A list of the site, score, and ferni-dirac probability for each 'hit' (end, score, P_bind)
        window_size (int): Size of local window to assess clustering (default = 100 bp).
        sequence_length (int): Total length of the promoter sequence. If None, inferred from max position.

    Returns:
        local_scores (list of floats): Binding score per window.
        global_binding_score (float): Sum of weighted cumulative probability over all windows (proxy for total TF recruitment potential).
    """
    # Return no value if the site probability list is empty
    if not site_probs:
        return [], 0.0

    # Determine the sequence length based on maximum end index of 'hits' (0-based end position)
    if sequence_length is None:
        sequence_length = max(s['pos'] for s in site_probs) + 1

    # Sort site_probs by end position
    site_probs_sorted = sorted(site_probs, key=lambda x: x["pos"])

    # Create an list of positions and probabilities for fast access
    positions = [hit["pos"] for hit in site_probs_sorted]
    probabilities = [hit["Pi"] for hit in site_probs_sorted]
    
    # Initialize array of 0s for cumulative probabilities
    prob_array = np.array([0.0] * sequence_length)

    # Iterate through the tuple of site probabilities for the hits
    for end, p in zip(positions, probabilities):
        if 0 <= end < sequence_length:

            # Assign the site probability value to its corresponding index in the array
            prob_array[end] += p 

    # Initialize a list to hold local scores for homotypic clusters
    local_scores = []

    # Iterate over each possible cluster of site probabilities (performs sliding window to find max cluster probability
    for i in range(0, sequence_length - window_size + 1):

        # Create a shallow slice of the array corresponding to the specific window and sum over all probability values in array
        local_sum = np.sum(prob_array[i:i + window_size])

        # Append the local sum to the local score
        local_scores.append(local_sum)

    return local_scores 

# ==============================================================================
#                       Execute Complete Script
# ==============================================================================

def thermodynamic_scanning_two_state(
        promoter_seqs,
        xml_path,
        lamR = 1.4,
        S0 = 4.0,
        cluster_window_size = 150,
        min_dist = 5,
        cluster_constraint = False,
        parse_header_strand = False,
        both_models = False,
        negative_sequences = None
    ):
    """
    Performs the complete thermodynamic scanning of promoter sequences.

    Args:
        promoter_seqs: List of promoter sequences to scan.

        xml_path: Path to the XML file containing HMM parameters.

        lamR: Regularization parameter for occupancy probability (default = 1.7).

        S0: Initial chemical potential value (default = 5.0).

        tau: Search slope constant (default = 0.25).

        search_window_bp: Size of the search window in base pairs (default = 200).

        cluster_window_size: Size of the clustering window in base pairs (default = 200).
        
        min_dist: Minimum distance between clusters (default = 5).

        cluster_constraint: Whether to apply clustering constraints (default = True).

        parse_fasta_header: Whether to parse FASTA header information (default = False).

        both_models: Whether to compute both thermodynamic occupancy and two-state model (default = False). Otherwise, only two-state model is computed.

    Returns:

        Tuple[List[Dict[str, float | str]], List[Dict[str, float | str]], List[Dict[str, float | str]]]: The scanning results for each promoter sequence:

            - Two_State_Model_Results: A list of results from the two-state model for each sequence:

                gene: The gene associated with the promoter sequence (found from FASTA header)
                maximum local score: The maximum cluster score calculated from the effective probabilities
                best hit effective probabilities: The maximum effective probability found in the sequence (the corresponding site is called the "best hit")
                best hit visitation prior: The visitation prior probability of the best hit
                best hit recognition probability: The recognition probability of the best hit
                best hit start: The genomic start coordinates of the best hit
                best hit end: The genomic end coordinates of the best hit
                best hit sequence: The nucleotide sequence of the best hit
                best hit strand: The strand (+ or -) of the best hit
                best hit state: The hidden states through the HMM for the best hit 

            - Binding_Values: A list of binding values for each promoter sequence.

                gene: The gene associated with the promoter sequence (found from FASTA header)
                number of hits: The total number of valid Viterbi hits in the sequence
                best hit: The maximum Viterbi hit in the sequence
                best state path: The path of the best hit through the hidden markov model
                best sequence: The sequence (observed nucleotides) of the best hit
                best strand: The strand on which the best hit is located
                best hit start: The start position of the best hit in the promoter sequence
                best hit end: The end position of the best hit in the promoter sequence
                best upstream distance: The distance from the best hit to the transcription start site
                maximum site probability: The maximum site probability across all hits in the sequence
                maximum local score: The maximum cluster-derived score across the sequence

            - Probability_Summaries: A list of probability summaries for each promoter sequence.

                site probabilities: A list of all site probabilities for sequence
                cluster score: A list of all cluster scores for sequence


    """
    # Set up logging
    setup_logging()

    if both_models is False:
        logging.info("[Scanning] Two-state model only; skipping thermodynamic occupancy.")  
    else:
        logging.info("[Scanning] Both thermodynamic occupancy and Two-State model will be computed.")

    # Determine all HMM Parameters
    state_ids, initial_vals, emission_vectors, transition_vals = extract_probabilities(xml_path=xml_path)

    # Determine absolute maximum joint log-probability over any length-L path in the HMM model 
    s_max, s_max_path = get_max_score(
        state_ids=state_ids,
        initial_vals=initial_vals,
        transition_vals=transition_vals
    )

    logging.info("[Scanning] The maximum joint log-probability is %.3f ; whose path is %s" % (s_max, s_max_path))

    # ---------------- Compute Background Markov Model ---------------------------

    this_dir = os.path.dirname(os.path.abspath(__file__)) 

    hg_38_genome = os.path.join(this_dir, "../Extraction of Fasta Sequences from DEGs/input_data/hg38.fa")

    def _open_text(path):
        return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")

    def sniff_seq_file(path, max_lines=50):
        """
        Return 'fasta' or 'bed' by peeking at content.
        - FASTA: first nonempty line starts with '>'
        - BED:   first nonempty, non-comment line has >=3 tab columns with ints in cols 2–3
        """
        with _open_text(path) as fh:
            for _ in range(max_lines):
                line = fh.readline()
                if not line:
                    break
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("track") or s.startswith("browser"):
                    continue
                if s.startswith(">"):
                    return "fasta"
                toks = s.split("\t")
                if len(toks) >= 3 and toks[1].isdigit() and toks[2].isdigit():
                    return "bed"
        raise ValueError("Could not determine format for: %s" % {path})

    # ---------- FASTA -> sequences ----------
    

    def fasta_to_sequences(fasta_path, min_len=1):
        """Yield uppercase A/C/G/T/N strings from FASTA."""
        for rec in SeqIO.parse(fasta_path, "fasta"):
            seq = str(rec.seq).upper()
            if len(seq) >= min_len:
                yield seq

    # ---------- BED -> sequences (requires reference FASTA) ----------
    def bed_to_sequences(bed_path, ref_fasta_path, pad=0):
        """
        Fetch sequences for BED intervals (0-based, half-open).
        Tries pyfaidx first, falls back to pysam. Returns uppercase strings.
        """
        fetch = None
        try:
            from pyfaidx import Fasta
            fa = Fasta(ref_fasta_path, as_raw=True, sequence_always_upper=True)
            def fetch(chrom, s, e):
                s = max(0, int(s) - pad)
                e = int(e) + pad
                return str(fa[chrom][s:e])
        except Exception:
            try:
                import pysam
                fa = pysam.FastaFile(ref_fasta_path)
                def fetch(chrom, s, e):
                    s = max(0, int(s) - pad)
                    e = int(e) + pad
                    return fa.fetch(chrom, s, e).upper()
            except Exception as e:
                raise ImportError("Install either 'pyfaidx' or 'pysam' to fetch sequences from a reference FASTA.") 

        seqs = []
        with _open_text(bed_path) as fh:
            for line in fh:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 3:
                    continue
                chrom, start, end = f[0], f[1], f[2]
                seqs.append(fetch(chrom, start, end))
        return seqs

    # ---------- One entry point ----------

    def load_sequences(path, ref_fasta=None, pad=0):
        """
        Returns an iterable of uppercase sequences suitable for MarkovBG1.fit(...) and Kinetic RHS Ramp.
        - FASTA: reads sequences directly
        - BED: requires ref_fasta to fetch the DNA for intervals
        """
        kind = sniff_seq_file(path)
        logging.info("[Background Model] Detected %s format for: %s" % (kind.upper(), path))
        if kind == "fasta":
            return fasta_to_sequences(path)
        elif kind == "bed":
            if not ref_fasta:
                raise ValueError("BED input requires a reference FASTA (ref_fasta=...).")
            return bed_to_sequences(path, ref_fasta, pad=pad)
        else:
            raise ValueError("Unsupported format: %s" % {kind})
        
    if negative_sequences:
        bg = MarkovBG1(pseudocount=1.0)

        logging.info("[Background Model] Fitting background model to negative sequences: %s" % negative_sequences)

        # 1) Fit on negatives
        seq_iter_fit = load_sequences(negative_sequences, ref_fasta=hg_38_genome)
        bg.fit(seq_iter_fit)

        # 2) Calibrate μ,σ from empirical windows (needs a fresh iterator)
        logging.info("[Background Model] Calibrating background model Z-score from negative sequences: %s" % negative_sequences)
        seq_iter_cal = load_sequences(negative_sequences, ref_fasta=hg_38_genome)
        bg.calibrate_z_from_negatives(
           L = len(state_ids)//4,
           negatives = seq_iter_cal,     # pass the iterator explicitly
        )
        logging.info("[Background Model] Calibrated background model: μ=%.3f, σ=%.3f" % (bg.zparams.get("global", None).get("mu", None), bg.zparams.get("global", None).get("sigma", None)))
    else:
        bg = None
        logging.info("[Background Model] No negative sequences provided; skipping background model fitting.")


    # ---------------- Compute Thresholds for Kinetic Model RHS Ramp ---------------------------

    # theta_low, theta_high, rep = calibrate_ramp_thresholds_on_negatives(
    #     negatives_source = seq_iter_fit,
    #     state_ids=state_ids,
    #     initial_vals=initial_vals,
    #     transition_vals=transition_vals,
    #     emission_vectors=emission_vectors,
    #     s_max=s_max,
    #     lamR=lamR,                     
    #     S0=S0,                          
    #     bg_model=bg,                   
    #     bg_weight=0.6,          
    #     window_bp=400,
    #     target_anchors_per_window=5,   # target number of anchors per 200bp window
    #     max_windows_per_seq=200
    # )

    # print("[Ramp-Calib] θ_low=%.3f θ_high=%.3f ; negatives expect ~%.2f anchors/200bp" %
    #     (theta_low, theta_high, rep["est_anchors_per_window"]))


    # ---------------- Compute Probability based on Thermodynamic Equilibrium Model ---------------------------

    def _thermodynamic_model(record, viterbi_hits, state_ids): 
        """
        Internal Helper: Performs thermodynamic modeling for a given record and its Viterbi hits.

        Args:
            record: The SeqRecord object containing the sequence information.

            viterbi_hits: A list of NamedTuples (end, score, path) for every possible L-mer window in the sequence where:
                - end (int):             The 0-based index of the window's terminal base
                - score (float):         The maximal joint log-probability of the Viterbi path through the window (∑logP(transition))
                - path (List[str]):      The state ID sequence for that path
                - sequence (List[str]):  The nucleotide sequence of the L-mer window
                - strand (str):          The strand on which the L-mer window was applied

            state_ids: A list of sorted state IDs corresponding to the HMM states.
        """

        # Retrieve viterbi hit and its corresponding state sequence and nucleotide sequence; key= parameter for max() applies lambda function to each element of list which is the NamedTuple (see above annotations on key parameter)
        best_viterbi_hit        = max(viterbi_hits, key=lambda h: h.score)
        best_viterbi_end        = int(best_viterbi_hit.end)
        best_viterbi_start      = int(best_viterbi_end) - (len(state_ids)//4)
        best_viterbi_start      = float(best_viterbi_hit.end) - (len(state_ids) // 4) + 1 # Adjusted start position based on length of motif
        best_viterbi_score      = best_viterbi_hit.score
        best_viterbi_sequence   = best_viterbi_hit.sequence
        best_viterbi_path       = best_viterbi_hit.path
        best_viterbi_strand     = best_viterbi_hit.strand

        logging.info("[Thermodynamic Scanning] Best Viterbi hit for %s: %s" % (record.id, best_viterbi_hit))

        # Convert Viterbi hits to per-site dicts with ΔS etc.
        sites_fwd, sites_rev = _compute_site_deltas(viterbi_hits, s_max)

        # Map the ΔS to a recognition energy ER and Fermi-Dirac probability Π 
        sites_forward = _fermi_dirac_probs(
            sites=sites_fwd,
            lamR = lamR,
            S0 = S0,
            bg_model = bg,
            bg_weight = 0.6
        )
        
        sites_reverse = _fermi_dirac_probs(
            sites=sites_rev,
            lamR = lamR,
            S0 = S0,
            bg_model = bg,
            bg_weight = 0.6
        )

        # Extract all site probabilities (after filtering out invalid probabilities)
        site_probabilities_forward = [site["Pi"] for site in sites_forward if site["Pi"] > -1.0] 
        site_probabilities_reverse = [site["Pi"] for site in sites_reverse if site["Pi"] > -1.0]

        # Determine total number of valid hits
        total_hits = len(site_probabilities_forward) + len(site_probabilities_reverse)

        # Compute homotypic cluster scores depending on whether spacing constraint is applied
        if cluster_constraint:
            local_scores_forward = thermodynamic_cluster_binding_score(
                site_probs=sites_forward,
                window_size=cluster_window_size,
                min_dist=min_dist
        )
            local_scores_reverse = thermodynamic_cluster_binding_score(
                site_probs=sites_reverse,
                window_size=cluster_window_size,
                min_dist=min_dist
        )
        else:
            # Compute homotypic cluster scores and global binding scores with no spacing constraint
            local_scores_forward = thermodynamic_cluster_binding_score_no_min(
                site_probs=sites_forward,
                window_size=cluster_window_size
            )
            local_scores_reverse = thermodynamic_cluster_binding_score_no_min(
                site_probs=sites_reverse,
                window_size=cluster_window_size
            )

        # Determine maximum site probability and local cluster score for forward and reverse strands
        max_site_prob_forward = max(site_probabilities_forward) if site_probabilities_forward else 0.0
        max_site_prob_reverse = max(site_probabilities_reverse) if site_probabilities_reverse else 0.0
        max_local_score_forward = max(local_scores_forward) if local_scores_forward else 0.0
        max_local_score_reverse = max(local_scores_reverse) if local_scores_reverse else 0.0

        # Determine overall maximum site probability and local cluster score across both strands
        max_site_prob = max(max_site_prob_forward, max_site_prob_reverse)
        max_local_score = max(max_local_score_forward, max_local_score_reverse)

        # Combine site probabilities and local scores from both strands for output
        site_probabilities = site_probabilities_forward + site_probabilities_reverse
        local_scores = local_scores_forward + local_scores_reverse

        # Scan the FASTA header embedded in the SeqRecord object's identifier (ID) and unpack the returned tuple; extracts relevant metadata (fasta header can differ on whether it contains strand information or not)
        if parse_header_strand == True:
            gene, chrom, promoter_s, promoter_e, strand = parse_fasta_header(record.id)
        else:
            gene, chrom, promoter_s, promoter_e, strand = parse_fasta_header(record.id, has_strand=False)

        # Convert strand-relative hit coordinates into absolute genomic coordinates in + strand coordinate system for best hits 
        best_genomic_hit_start, best_genomic_hit_end = transform_absolute_coordinates(
            promoter_s=promoter_s, 
            promoter_e=promoter_e, 
            hit_strand=best_viterbi_strand, 
            hit_start=best_viterbi_start, 
            hit_end=best_viterbi_end) 

        # Determine distance from mid-point of each best hit motif to transcriptional start site 
        best_upstream_distance = compute_upstream_distance(
            promoter_s=promoter_s, 
            hit_start=best_genomic_hit_start, 
            hit_end=best_genomic_hit_end, 
            strand=strand
            ) if best_genomic_hit_start >= 0 else None # Note: >=0 is used for conditional since coordinate transformation returns -1 for hit_start if header cannot be parsed

        logging.info("[Thermodynamic Scanning] The maximum score of the cluster for %s is: %s" % (gene, max_local_score))

        return gene, total_hits, best_viterbi_score, best_viterbi_sequence, best_viterbi_path, max_site_prob, max_local_score, best_genomic_hit_start, best_genomic_hit_end, best_upstream_distance, site_probabilities, local_scores
   
    # ---------------- Compute Probability based on Two-State Model ---------------------------

    def _two_state_model(record, viterbi_hits, state_ids, method="bind"):
        """
        Internal Helper: Performs two-state thermodynamic modeling for a given record and its Viterbi hits.

        Args:
            record: The SeqRecord object containing the sequence information.
            viterbi_hits: A list of NamedTuples (end, score, path) for every possible L-mer window in the sequence where:
                - end (int):             The 0-based index of the window's terminal base
                - score (float):         The maximal joint log-probability of the Viterbi path through the window (∑logP(transition))
                - path (List[str]):      The state ID sequence for that path
                - sequence (List[str]):  The nucleotide sequence of the L-mer window
                - strand (str):          The strand on which the L-mer window was applied
        """
    
        if method == "bind":

            logging.info("[Two-State Scanning] Computing poisson anchor probabilities using the 'bind' method.")

            site_probs_eff_forward, site_probs_eff_reverse = dwell_pipeline(
                hits=viterbi_hits,
                s_max=s_max,
                lam_R=lamR,
                S0=S0,
                bg_model = bg,
                system_key="bind",
            )

            site_probs_eff_forward = add_anchored_scores_for_band(
                sites=site_probs_eff_forward,
                exclude_center=True
            )

            site_probs_eff_reverse = add_anchored_scores_for_band(
                sites=site_probs_eff_reverse,
                exclude_center=True
            )

        elif method == "attempt":
            
            logging.info("[Two-State Scanning] Computing poisson anchor probabilities using the 'attempt' method.")

            site_probs_eff_forward, site_probs_eff_reverse = dwell_pipeline(
                hits=viterbi_hits,
                s_max=s_max,
                lam_R=lamR,
                S0=S0,
                bg_model = bg,
                system_key="attempt"
            )

            site_probs_eff_forward = add_anchored_scores_for_band(
                sites=site_probs_eff_forward,
                exclude_center=False
            )

            site_probs_eff_reverse = add_anchored_scores_for_band(
                sites=site_probs_eff_reverse,
                exclude_center=False
            )

        # Determine best hit
        best_p_eff_hit_forward =        max(site_probs_eff_forward, key=lambda h: h["anchored_total"]) if site_probs_eff_forward else None
        best_p_eff_hit_reverse =        max(site_probs_eff_reverse, key=lambda h: h["anchored_total"]) if site_probs_eff_reverse else None
        best_p_eff_hit         =        max(best_p_eff_hit_forward, best_p_eff_hit_reverse, key=lambda h: h["anchored_total"]) if best_p_eff_hit_forward and best_p_eff_hit_reverse else None

        # Retrieve corresponding parameters of best site
        best_p_anc =            best_p_eff_hit.get("anchored_total", None) 
        best_p_dwell =          best_p_eff_hit.get("dwell", None)
        best_pR =               best_p_eff_hit.get("recognition probability", None)
        best_end =              best_p_eff_hit.get("position", None)
        best_start =            best_end - (len(state_ids) // 4) + 1                # Adjusted start position based on length of motif
        best_state =            best_p_eff_hit.get("states", None)
        best_sequence =         best_p_eff_hit.get("sequence", None)
        best_strand =           best_p_eff_hit.get("strand", None)
        
        # Determine maximum cluster window
        max_local_cluster_forward = max(site_probs_eff_forward, key=lambda h: h["cluster score"]) if site_probs_eff_forward else None
        max_local_cluster_reverse = max(site_probs_eff_reverse, key=lambda h: h["cluster score"]) if site_probs_eff_reverse else None
        max_local_cluster =         max(max_local_cluster_forward, max_local_cluster_reverse, key=lambda h: h["cluster score"]) if max_local_cluster_forward and max_local_cluster_reverse else None

        # Retrieve corresponding cluster score 
        max_local_score = max_local_cluster.get("cluster score", 0.0) if max_local_cluster else 0.0

        logging.info("[Two-State Scanning] Best effective probability hit for %s: %s" % (record.id, best_p_eff_hit))
        
        """
        # Compute HC scores with spacing constraint applied
        if cluster_constraint:

            logging.info("[Two-State Scanning] Computing local cluster scores with minimum distance constraint of %s bp." % min_dist)

            logging.info("[Two-State Scanning] Computing local cluster scores for forward strand.")
            
            local_scores_forward = thermodynamic_cluster_binding_score(
                site_probs=site_probs_eff_forward,
                window_size=cluster_window_size,
                min_dist=min_dist
            )

            logging.info("[Two-State Scanning] Computing local cluster scores for reverse strand.")

            local_scores_reverse = thermodynamic_cluster_binding_score(
                site_probs=site_probs_eff_reverse,
                window_size=cluster_window_size,
                min_dist=min_dist
        )

        # Compute HC scores with no spacing constraint
        else:

            logging.info("[Two-State Scanning] Computing local cluster scores with no minimum distance constraint.")

            logging.info("[Two-State Scanning] Computing local cluster scores for forward strand.")

            local_scores_forward = thermodynamic_cluster_binding_score_no_min(
                site_probs=site_probs_eff_forward,
                window_size=cluster_window_size
            )

            logging.info("[Two-State Scanning] Computing local cluster scores for reverse strand.")

            local_scores_reverse = thermodynamic_cluster_binding_score_no_min(
                site_probs=site_probs_eff_reverse,
                window_size=cluster_window_size
            )

        # Determine maximum cluster score
        max_local_score_forward = max(local_scores_forward) if local_scores_forward else 0.0
        logging.info("[Two-State Scanning] The maximum score of the cluster for %s is: %s" % (record.id, max_local_score_forward))

        max_local_score_reverse = max(local_scores_reverse) if local_scores_reverse else 0.0
        logging.info("[Two-State Scanning] The maximum score of the cluster for %s is: %s" % (record.id, max_local_score_reverse))

        max_local_score = max(max_local_score_forward, max_local_score_reverse)
        logging.info("[Two-State Scanning] The maximum score of the cluster for %s is: %s" % (record.id, max_local_score))
        """

        # Scan the FASTA header embedded in the SeqRecord object's identifier (ID) and unpack the returned tuple; extracts relevant metadata (fasta header can differ on whether it contains strand information or not)
        if parse_header_strand == True:
            gene, chrom, promoter_s, promoter_e, strand = parse_fasta_header(record.id)
        else:
            gene, chrom, promoter_s, promoter_e, strand = parse_fasta_header(record.id, has_strand=False)

        # Convert strand-relative hit coordinates into absolute genomic coordinates in + strand coordinate system for best hits 
        best_genomic_hit_start, best_genomic_hit_end = transform_absolute_coordinates(
            promoter_s=promoter_s, 
            promoter_e=promoter_e, 
            hit_strand=best_strand, 
            hit_start=best_start, 
            hit_end=best_end) 

        # Determine distance from mid-point of each best hit motif to transcriptional start site 
        best_upstream_distance = compute_upstream_distance(
            promoter_s=promoter_s, 
            hit_start=best_genomic_hit_start, 
            hit_end=best_genomic_hit_end, 
            strand=strand
            ) if best_genomic_hit_start >= 0 else None # Note: >=0 is used for conditional since coordinate transformation returns -1 for hit_start if header cannot be parsed

        return gene, max_local_score, best_p_anc, best_p_dwell, best_pR, best_genomic_hit_start, best_genomic_hit_end, best_upstream_distance, best_strand, best_sequence, best_state, 
    
    # ---------- Loop Through All Sequences and Compute Final Probabilities ------------

    binding_values = []
    probability_summaries = []
    two_state_model = []
    cluster_scores_two = []

    for record in promoter_seqs:

        # Compute the joint log probability for each window state in sequence
        viterbi_hits = Viterbi_algorithm(
            sequence=record,
            state_ids=state_ids,
            initial_vals=initial_vals,
            transition_vals=transition_vals,
            emission_vectors=emission_vectors
        )

        # ---------------- Compute Probability based on Thermodynamic Two-State Model ---------------------------

        gene, max_local_score, best_p_anc, best_p_dwell, best_pR, best_genomic_hit_start, best_genomic_hit_end, best_upstream_distance, best_strand, best_sequence, best_state = _two_state_model(
            record=record,
            viterbi_hits=viterbi_hits,
            state_ids=state_ids
        )

        two_state_model.append({
            "gene": gene, 
            "maximum local score": max_local_score,
            "best hit effective probability": best_p_anc,
            "dwell": best_p_dwell,
            "best hit recognition probability": best_pR,
            "best hit start": best_genomic_hit_start,
            "best hit end": best_genomic_hit_end,
            "best upstream distance": best_upstream_distance,
            "best hit strand": best_strand,
            "best hit sequence": best_sequence,
            "best hit state": best_state
        })

        cluster_scores_two.append({
            "two state": max_local_score
        })

        # ---------------- Compute Probability based on Thermodynamic Equilibrium Model ---------------------------

        if both_models: 

            gene, total_hits, best_viterbi_score, best_viterbi_sequence, best_viterbi_path, max_site_prob, max_local_score, best_genomic_hit_start, best_genomic_hit_end, best_upstream_distance, site_probabilities, local_scores = _thermodynamic_model(
                record=record,
                viterbi_hits=viterbi_hits,
                state_ids=state_ids
            )

            binding_values.append({
                "gene": gene, 
                "number of hits": total_hits,
                "best hit": best_viterbi_score,
                "best sequence": best_viterbi_sequence,
                "best state path": best_viterbi_path,
                "maximum site probability": max_site_prob,
                "maximum local score": max_local_score,
                "best hit start": best_genomic_hit_start,
                "best hit end": best_genomic_hit_end,
                "best upstream distance": best_upstream_distance
            })
            
            probability_summaries.append({
                "gene": gene,
                "site probabilities": site_probabilities,
                "cluster scores": local_scores
            })

    logging.info("[Scanning] Completed scanning of promoter sequences.")

    if both_models:
        return (binding_values, two_state_model, probability_summaries)
    else:
        return two_state_model, cluster_scores_two

