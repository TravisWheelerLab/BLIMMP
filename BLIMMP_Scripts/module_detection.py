#BLIMMP Source Code
#Date: December 17 2025
#Author: Neha S

import os
import glob
import re
import sys
import math
import ast
import json
import argparse
import pandas as pd
import numpy as np
from math import exp
from pathlib import Path
import operator as op
from dataclasses import dataclass
from numba import njit
from scipy.stats import beta
import warnings
import logging
from typing import Dict, Tuple, Set, Union, List, Any, Optional
import zipfile    


## CONSTANTS
KINGDOM = {"bacillati", "fusobacteriati", "mycoplasmatota", "pseudomonadati", "thermotogati"}
PHYLUM  = {"bacillota", "acidobacteriota", "actinomycetota", "campylobacterota", "cyanobacteriota",
           "deinococcota", "fcb_group", "mycoplasmatota", "myxococcota", "pseudomonadota",
           "pvc_group", "spirochaetota", "thermodesulfobacteriota", "thermotogota"}
KO_RE = re.compile(r'^K\d{5}$')

## Configurations
@dataclass(frozen=True)
class Paths:
    counts_dir: Path
    onehop_dir: Path
    twohop_dir: Path
    module_neighbor_dir: Path
    module_eq_json: Path
    module_json_dir: Path
    kofam_ko_list_path: Path
    module_frequencies: Path
    module_reaction_dir: Path


@dataclass(frozen=True)
class RunConfig:
    input_file: str
    fmt: str                   # 'tbl' or 'domtblout'
    sigma: float               # 0..1
    taxonomy: str              # bacteria / phylum / kingdom tag
    output_prefix: str
    verbose: bool = False
    logfile_path: Optional[str] = None


@njit
def _hmm_union_len_per_group_py(gids, starts, ends, n_groups):
    covered = np.zeros(n_groups, dtype=np.int64)

    cur_gid = gids[0]
    cur_s   = starts[0]
    cur_e   = ends[0]

    for i in range(1, len(starts)):
        g = gids[i]
        s = starts[i]
        e = ends[i]

        if g != cur_gid:
            covered[cur_gid] += (cur_e - cur_s + 1)
            cur_gid = g
            cur_s   = s
            cur_e   = e
        else:
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                covered[cur_gid] += (cur_e - cur_s + 1)
                cur_s = s
                cur_e = e

    covered[cur_gid] += (cur_e - cur_s + 1)

    return covered


class HMMParsers:

    #Process hmmer domtblout file
    @staticmethod
    def process_domtblout(path):
        print("Processing the domtblout file ...\n")
        #Cols to process
        cols = ['target name', 'target_accession', 'tlen', 'query_name',
            'query_accession', 'qlen', 'full_Evalue', 'full_score',
            'full_bias', 'n_domains', 'of_domains', 'c_Evalue',
            'i_Evalue', 'i_score', 'i_bias', 'hmm from', 'hmm to',
            'ali from', 'ali to', 'env from', 'env to', 'acc'
        ]

        usecols=[
            'target name','query_name',
            'hmm from','hmm to','tlen',
            'ali from','ali to','qlen',
            'full_score','full_Evalue',
            'i_score','i_Evalue'
        ]

        df = pd.read_csv(
            path,
            comment='#',
            header=None,
            names=cols,
            usecols=list(range(22)),
            sep= r"\s+",  
            engine='c',
            low_memory=False,
            memory_map=True
        )
        
        df = df[usecols].copy()

        #Don't need this with comment lines #skiprows=3, #skipfooter=10
        #Auto detect the header columns
        pat = KO_RE
        #Does the pattern match query or target
        q_matches = df['query_name'].astype(str).str.fullmatch(pat.pattern, na=False)
        t_matches = df['target name'].astype(str).str.fullmatch(pat.pattern, na=False)

        #Mean of number of matches
        fq = q_matches.mean()
        ft = t_matches.mean() 

        if (fq > 0 or ft > 0) and (fq >= ft):
            # query_name looks more like KO IDs
            df = df.rename(columns={'query_name': 'KO id', 'target name': 'target name'})
            df['hmm_len'] = df['qlen']
        elif ft > 0:
            # target name looks more like KO IDs
            df = df.rename(columns={'target name': 'KO id', 'query_name': 'target name'})
            df['hmm_len'] = df['tlen']
        else:
            print("Error: Can't read the KO ids.")

        #df = df.rename(columns={'i_score': 'score', 'i_Evalue': 'E-value'})
        df = df.rename(columns={'full_score': 'score', 'full_Evalue': 'E-value'})

        # Compute alignment/hmm segment lengths and remove zero-length hits
        df['ali_span'] = (df['ali to'] - df['ali from']).abs()
        df['hmm_span'] = (df['hmm to'] - df['hmm from']).abs()
        df["strand"] = np.where(df["ali to"] >= df["ali from"], "+", "-")

        df = df[(df['ali_span'] > 0) & (df['hmm_span'] > 0)].copy()

        df['per_hit_hmm_coverage'] = (df['hmm_span'] + 1) / df['hmm_len']

        # Drop helper spans if you don't want them hanging around
        df.drop(columns=['ali_span', 'hmm_span'], inplace=True)

        print("\nSTAGE 3: before union coverage")
        print("  shape:", df.shape)
        print("  unique (KO, target, strand):",df[['KO id','target name','strand']].drop_duplicates().shape[0])


        # after building df, hmm_len, etc.
        df['group_id'] = df.groupby(
            ['strand', 'target name', 'KO id'], 
            sort=False
        ).ngroup()

        n_groups = df['group_id'].max() + 1

        starts = df[['hmm from', 'hmm to']].min(axis=1).to_numpy(np.int64)
        ends   = df[['hmm from', 'hmm to']].max(axis=1).to_numpy(np.int64)
        gids   = df['group_id'].to_numpy(np.int64)

        order = np.lexsort((starts, gids))  # sort by gid, then start

        starts = starts[order]
        ends   = ends[order]
        gids   = gids[order]

        # union length per group (Numba)
        #covered_per_group = _hmm_union_len_per_group(gids, starts, ends, n_groups)

        # DEBUG: verify Numba vs Python
        covered_py = _hmm_union_len_per_group_py(gids, starts, ends, n_groups)


        # hmm_len per group
        hmm_len_per_group = (
            df.groupby('group_id', sort=False)['hmm_len']
            .first()
            .to_numpy()
        )

        coverage_per_group = covered_py / hmm_len_per_group

        # broadcast back
        gid_full = df['group_id'].to_numpy()
        df['hmm_covered_len']       = covered_py[gid_full]
        df['hmm_coverage_fraction'] = coverage_per_group[gid_full]

        print("\nSTAGE 4: after union coverage")
        print("  shape:", df.shape)
        print("  coverage stats:",
            "min =", df['hmm_coverage_fraction'].min(),
            "max =", df['hmm_coverage_fraction'].max())
        print("  unique (KO, target):",
            df[['KO id','target name']].drop_duplicates().shape[0])

        df.to_csv("domtblout_with_coverage.csv")
            
        return df


@njit
def _assign_groups_numba(starts, ends, frac_thresh):
    n = len(starts)
    g_st = np.empty(n, dtype=np.float64)
    g_en = np.empty(n, dtype=np.float64)
    gcount = 0
    grp_ids = np.empty(n, dtype=np.int32)

    for i in range(n):
        s = starts[i]; e = ends[i]
        assigned = False
        for g in range(gcount - 1, -1, -1):
            gs = g_st[g]; ge = g_en[g]
            if s > ge:
                break
            overlap = min(e, ge) - max(s, gs)
            if overlap <= 0.0:
                continue
            short_len = (e - s) if (e - s) < (ge - gs) else (ge - gs)
            if (overlap / short_len) >= frac_thresh:
                grp_ids[i] = g + 1
                if s < gs: g_st[g] = s
                if e > ge: g_en[g] = e
                assigned = True
                break
        if not assigned:
            g_st[gcount] = s
            g_en[gcount] = e
            gcount += 1
            grp_ids[i] = gcount
    return grp_ids


## Overlap grouping and per-position adjudication logic
class Overlap:
    @staticmethod
    def cluster_strand(df, from_col="ali from", to_col="ali to", frac_thresh=0.6):
        # compute strand-agnostic interval endpoints
        starts = df[[from_col, to_col]].min(axis=1).to_numpy(dtype=np.float64)
        ends   = df[[from_col, to_col]].max(axis=1).to_numpy(dtype=np.float64)
        orig_idx = df.index.to_numpy()

        # sort by start (stable)
        order = np.argsort(starts, kind="mergesort")
        starts = starts[order]; ends = ends[order]; orig_idx = orig_idx[order]

        # fast assignment
        grp_ids = _assign_groups_numba(starts, ends, float(frac_thresh))

        # return same shape/column as before
        out = pd.Series(grp_ids, index=orig_idx).sort_index()
        return out.to_frame("grp_id")

    @staticmethod
    def assign_overlap_groups(df_hits):
        print("Assigning overlapping groups...\n")
        df = df_hits.copy()

        out = []
        for (tgt, strand), sub in df.groupby(["target name","strand"], sort=False):
            clustered = Overlap.cluster_strand(sub)      # ← now uses the Numba path
            sub = sub.join(clustered, how="left")
            out.append(sub)

        result = pd.concat(out).sort_index()
        result["overlap_group"] = (
            result["target name"].astype(str)
            + "_" + result["grp_id"].astype(str)
            + "_" + result["strand"].astype(str)
        )
        return result


class File_Helpers:
    @staticmethod
    def load_kofamdb_file(kofampath):
        if not os.path.exists(kofampath):
            raise ImportError(f"KOfam DB file not found: {kofampath}")

        df = pd.read_csv(
        kofampath,
        sep=r"\s+",
        header=None,
        skiprows=1,
        usecols=[0, 1, 2],
        names=['KO id', 'kofam_score_threshold', 'score_type'])

        if df.shape[1] < 2:
            raise ValueError(f"KOfam file {kofampath} has <2 columns; can't parse thresholds.")

        df['KO id'] = df['KO id'].astype(str).str.strip().str.upper()
        df['kofam_score_threshold'] = pd.to_numeric(df['kofam_score_threshold'], errors='coerce')
        df = df.dropna(subset=['kofam_score_threshold'])

        if 'score_type' not in df.columns:
            df['score_type'] = 'full'
        else:
            df['score_type'] = (
                df['score_type']
                .fillna('full')
                .astype(str).str.strip().str.lower()
            )
            df.loc[~df['score_type'].isin(['full', 'domain']), 'score_type'] = 'full'

        
        return dict(zip(df['KO id'], zip(df['kofam_score_threshold'], df['score_type'])))
    
    @staticmethod
    def load_module_eq(module_eq_path):
        with open(module_eq_path, "r") as fh:
            module_equations = json.load(fh) 
        return module_equations
        
    @staticmethod
    def load_module_freq(module_fre_paths):
        module_freq = {}

        with open(module_fre_paths) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                module_id, freq = line.split("\t")
                module_freq[module_id] = float(freq)
        return module_freq
    
    @staticmethod
    def read_ko_occurrence(kooccpath):
        df = pd.read_csv(kooccpath, sep="\t", skiprows=1, header=None, names=['KO id','count','occurences'])
        print("The prior taxonomical class chosen is from file: ", kooccpath,"\n")
        df['KO_freq'] = df['occurences'].astype(float)
        return df
    
    @staticmethod
    def lineage_paths(taxonomy: str, paths: Paths):
        val = (taxonomy or "").strip().lower()
        if val in PHYLUM: level, name = "phylum", val
        elif val in KINGDOM: level, name = "kingdom", val
        else: level, name = "domain", "bacteria"
        tag = "domain_level_priors" if level == "domain" else f"{name}_{level}_level_priors"
        want_counts = paths.counts_dir / f"ko_freq_ko_matrix_sampleids_{tag}.tsv"
        want_one    = paths.onehop_dir  / f"One_Hop_Refilled_{tag}.json"
        want_two    = paths.twohop_dir  / f"Two_Hop_Refilled_{tag}.json"
        want_all = paths.module_neighbor_dir / f"Module_AllHop_Refilled_{tag}.json"
        if not (want_counts.exists() and want_one.exists() and want_two.exists()):
            if tag != "domain_level_priors":
                print(f"[taxonomy] Using domain-level fallbacks for '{tag}'.", file=sys.stderr)
            tag = "domain_level_priors"
            want_counts = paths.counts_dir / f"ko_freq_ko_matrix_sampleids_{tag}.tsv"
            want_one    = paths.onehop_dir  / f"One_Hop_Refilled_{tag}.json"
            want_two    = paths.twohop_dir  / f"Two_Hop_Refilled_{tag}.json"
            want_all = paths.module_neighbor_dir / f"Module_AllHop_Refilled_{tag}.json"
        return want_counts, want_one, want_two, want_all, tag
    
    @staticmethod
    def modules_to_kos(module_json_dir):

        ko_to_modules: dict[str, list[str]] = {}
        pattern = os.path.join(module_json_dir, "module_*_nodes.json")

        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath)                          
            module_name = filename.replace("_nodes.json", "").replace("module_", "")  # like 'M00001'

            with open(filepath, "r") as f:
                module_nodes = json.load(f)                                 # list of node ids

            # Node names that start with K; keep KO part before first underscore
            module_kos = {n.split("_", 1)[0] for n in module_nodes if isinstance(n, str) and n.startswith("K")}

            for ko in module_kos:
                ko_to_modules.setdefault(ko, []).append(module_name)

        return {ko: ",".join(sorted(mods)) for ko, mods in ko_to_modules.items()}
    
    @staticmethod
    def load_module_reactions(module_reaction_dir):
        #print("here")
        with open(module_reaction_dir, "r") as fh:
            module_reaction = json.load(fh) 
        return module_reaction
    
    @staticmethod
    def reactions_for_module_bestpath(module_map, module_id: str, best_path: str):
        #print("here")
        if not module_id or not best_path:
            return ""

        mod = module_map.get(module_id)
        if not mod:
            return ""

        kos = re.findall(r"K\d{5}", str(best_path))

        rxns = set()
        for ko in kos:
            for r in mod.get(ko, []):
                rxns.add(r)

        return ",".join(sorted(rxns))


  

class PositionScores:
    @staticmethod
    def compute_perposition_overlapgroup_softmax(df):
        #Calculate the softmax for each annotation in a group
        #Computation:
        # log_w_i = score_i * ln(2)
        # group_log_sum = logsumexp(log_w_i for each overlap_group)
        # X_i = exp(log_w_i - group_log_sum)
        df = df.copy().astype({'score': float})
        # Compute log(2**score) = score * ln(2)
        df['log_per_hit_weight'] = df['score'] * np.log(2.0)
        # log-sum-exp within each overlap_group
        df['group_log_sum'] = df.groupby('overlap_group')['log_per_hit_weight'].transform(lambda x: np.logaddexp.reduce(x.values))
        with np.errstate(divide='ignore', invalid='ignore'):
            df['overlap_relative_position_confidence'] = np.exp(df['log_per_hit_weight'] - df['group_log_sum']).fillna(0.0)
        return df
    

    @staticmethod
    def calculate_best_hit_with_noise(df,e_threshold=1e-4):
        df = df.copy()
        # noise term: noise_weight = 2**(-log2(e_threshold)) noise_logw = -ln(e_threshold)
        noise_logw = -np.log(e_threshold)
        # per-hit log-weight: ln(2**score) = score * ln(2)
        df['log_per_hit_weight'] = df['score'] * np.log(2)
        # group log-sum of per-hit weights
        df['group_log_sum'] = df.groupby('overlap_group')['log_per_hit_weight'].transform(lambda x: np.logaddexp.reduce(x.values))
        # total log-weight = log(group_sum + noise_weight)
        df['total_log_weight'] = np.logaddexp(df['group_log_sum'], noise_logw)
        # hit confidence = per_hit_weight / total_weight
        # log-space: exp(log_w − total_log_w)
        df['hit_conf'] = np.exp(df['log_per_hit_weight'] - df['total_log_weight'])
        # debug: print any nans
        nan_rows = df[df['hit_conf'].isna()][['score','log_per_hit_weight','group_log_sum','total_log_weight','hit_conf']]
        if not nan_rows.empty:
            print("Rows with NaN hit_conf:\n", nan_rows)
        # Pick the max-confidence row per overlap_group
        idx = (df.groupby('overlap_group')['hit_conf'].idxmax().dropna().astype(int))
        winners = df.loc[idx, ['overlap_group', 'KO id', 'score', 'hit_conf']].rename(
        columns={
            'KO id': 'overlapgroup_winner',
            'score': 'overlapgroup_winner_score',
            'hit_conf': 'overlapgroup_winner_hit_conf'})
        return winners.loc[idx].reset_index(drop=True)
    
    @staticmethod
    def winner_info_and_flags(df, kofampath):
        print("Calculating hit confidence for each KO ids...\n")
        df_soft = PositionScores.compute_perposition_overlapgroup_softmax(df)
        winners = PositionScores.calculate_best_hit_with_noise(df)
        keep_cols = ['overlap_group', 'overlapgroup_winner', 'overlapgroup_winner_score', 'overlapgroup_winner_hit_conf']
        df_new = df_soft.merge(winners[keep_cols], on='overlap_group', how='left', validate='many_to_one')

        #Merge the kofam score threshold here
        kofam_map = None

        #To add: 

        try:
            # load_kofamdb_file
            kofam_map = File_Helpers.load_kofamdb_file(kofampath)
            if len(kofam_map) > 0:
                # map thresholds to rows
                df_new['kofam_score_threshold'] = df_new['KO id'].map(lambda ko: kofam_map.get(ko, (np.nan, None))[0])
                df_new['kofam_score_type'] = df_new['KO id'].map(lambda ko: kofam_map.get(ko, (np.nan, None))[1])

                # conditions to pass
                compare_score = pd.Series(np.where(df_new['kofam_score_type'].eq('domain'),df_new['i_score'],df_new['score']),index=df_new.index)
                conditions = (df_new['kofam_score_threshold'].notna() & compare_score.notna() & (compare_score >= df_new['kofam_score_threshold']))

                # Outcompeted flag
                df_new['is_outcompeted'] = (df_new['KO id'] != df_new['overlapgroup_winner'])

                #Below threshold flag 
                # ? = Outcompeted and above kofam threshold
                # ! = Not outcompeted (winner) and below kofam threshold
                # below threshold, has score, not a winner
                # below threshold no score

                # hit_conf = 1.0 when above threshold, else relative confidence
                df_new['hit_conf'] = np.where(conditions,1.0,df_new['overlap_relative_position_confidence'])
                # flag is_dubious = True if passes threshold but NOT the overlapgroup winner
                df_new['flag_is_dubious'] = conditions & (df_new['KO id'] != df_new['overlapgroup_winner'])

                # flag is_below_kofam_threshold = True when threshold missing OR score < threshold
                df_new['flag_is_below_kofam_threshold'] = ~conditions
        except ImportError:
            kofam_map = None

        return df_new


class NeighborCalculations:
    @staticmethod
    def make_neighbor_dictionary(NEIGHBOR_TXT, df=None):
        with open(NEIGHBOR_TXT) as f:
            neighbor_data = json.load(f)
            
        #Nj_count = 0
        #Nij_count=0

        adj_raw = {}
        ko_counts = {}
        
        for ko, nbrs in neighbor_data.items():
            ko_counts[ko] = float(nbrs.get("_count", 0.0))
            
            
        for ko, nbrs in neighbor_data.items():
            out = {}
            for nb, val in nbrs.items():
                if nb == "_count":
                    continue
                
                Nij = float(val)
                Nj = ko_counts.get(nb)
                
                if Nj is None:
                    raise ValueError(f"Missing _count for neighbor KO '{nb}' " f"(referenced from '{ko}')")
                    #Nj_count += 1

                #if Nij > Nj:
                    raise ValueError(f"Inconsistent counts: Nij > Nj "f"(i='{ko}', j='{nb}', Nij={Nij}, Nj={Nj})")
                #    Nij_count += 1
                
                out[nb] = Nij
            if out:
                adj_raw[ko] = out

        #print(len(neighbor_data))
        #print(Nj_count,Nij_count)
        #print(len(ko_counts))
        return adj_raw, ko_counts




class CalculateKOProbabilities:
    @staticmethod
    def sigma_completeness_alteration(df, sigma_val):
        sigma_val_update = 1 - ((np.exp(3 * sigma_val) - 1) / np.exp(3))
        df['sigma'] = sigma_val_update
        return df


    @staticmethod
    def calculate_dk_per_ko(df, ko_occ, verbose: bool = False, M=0.7):
        logging.info("Calculating per-KO probabilities...")

        df = pd.merge(df, ko_occ, on="KO id", how='left')
        df['KO_freq'] = df['KO_freq'].fillna(0.0)
        
        # Check how many hit_conf values are missing
        #missing_conf = df['hit_conf'].isna().sum()
        #print(f"Missing hit_conf values: {missing_conf} out of {total_rows} ({100*missing_conf/total_rows:.2f}%)")
        # Calculate Dk
        seed = df['hit_conf'].fillna(0.0)
        df['Dk'] = seed + (1 - seed) * 0.7 * df['KO_freq'] * df['KO_freq']
        
        if verbose:
            for _, row in df.iterrows():
                term = (1 - row['hit_conf']) * M * (row['KO_freq'] ** 2)
                logging.debug(
                    f"KO id: {row['KO id']}, "
                    f"O_i (hit_conf): {row['hit_conf']:.4f}, "
                    f"F_i (KO_freq): {row['KO_freq']:.4f}, "
                    f"(1-O_i)*M*F_i²: {term:.4f} = P_i: {row['Dk']:.4f}"
                )

        
        return df
    
    @staticmethod
    def calculate_reliable_conditional_prob(i, j, neighbor_map, ko_counts, lambda_param=50, verbose=False):
        """
        Calculate R(i,j) = N_{i and j} / (N_j + λ)
        """ 
        #min permissible influencer
        min_frac = 0.25

        # Get co-occurrence count: N_{i and j}
        n_i_and_j = neighbor_map.get(i, {}).get(j, 0)
        
        # Get count of j: N_j
        n_j = ko_counts.get(j, 0)
        
        # Calculate R(i,j) with lambda regularization
        if n_j <= 0:
            if verbose:
                logging.debug(
                    "KO %s <- buddy %s excluded: Nj=0 (no support)",
                    i, j
                )
            return 0.0

        frac = n_i_and_j /n_j

        if frac < min_frac:
            if verbose:
                logging.debug(
                    "KO %s <- buddy %s excluded: Nij/Nj = %.3f < %.2f "
                    "(Nij=%d Nj=%d)",
                    i, j, frac, min_frac, n_i_and_j, n_j
                )
            return 0.0
        
        r_ij = n_i_and_j / (n_j + lambda_param)
        
        return r_ij



    @staticmethod
    def dk_neighbor_update(df, neighbor_map, ko_counts, alpha=0.6, lambda_param=50, return_used=False, verbose: bool = False):
        logging.info("Updating per-KO probabilities based on the influence neighborhood...\n")
        s = df[['KO id','Dk','hit_conf','count']].copy()
         
        s['KO id']    = s['KO id'].astype(str).str.strip().str.upper()
        s['Dk']       = pd.to_numeric(s['Dk'], errors='coerce').fillna(0.0)
        s['hit_conf'] = pd.to_numeric(s['hit_conf'], errors='coerce').fillna(0.0)
        s['count']    = pd.to_numeric(s['count'], errors='coerce').fillna(0.0)

        dk_dict                = dict(zip(s['KO id'], s['Dk']))
        hit_conf_map_current   = dict(zip(s['KO id'], s['hit_conf']))

        new_dk = {}
        used_neighbors = {}

        buddy_stats_map = {}   #for viz
        

        for i, p_i in dk_dict.items():
            module_families = neighbor_map.get(i, {})
            #Default
            buddy_stats_map[i] = {
                "alpha": float(alpha),
                "lambda": float(lambda_param),
                "S": 0.0,
                "X": None,
                "shift": 0.0,
                "buddy_count_used": 0,
                "buddies": []
            }

            if not module_families:
                new_dk[i] = p_i
                used_neighbors[i] = []
                if verbose:
                    logging.debug(
                        "KO %s: no neighbors, C_i = P_i = %.4f",
                        i, p_i
                    )
                continue


            # Calculate R(i,j) for each family j
            rij_map = {}
            for j in module_families.keys():
                if j == i:
                    continue
                r_ij = CalculateKOProbabilities.calculate_reliable_conditional_prob(
                    i, j, neighbor_map, ko_counts, lambda_param, verbose
                )
                if r_ij > 0.0 and math.isfinite(r_ij):
                    #squared for dampening
                    rij_map[j] = r_ij #*r_ij

            used_neighbors[i] = sorted(rij_map.keys())
            
            if not rij_map:
                new_dk[i] = p_i
                if verbose:
                    logging.debug("KO %s: neighbors found but no reliable R(i,j); C_i = P_i = %.4f",i, p_i)
                continue

            # Sum of reliable conditional probabilities (weights)
            S = sum(rij_map.values())

            if S <= 0.0:
                new_dk[i] = p_i
                if verbose:
                    logging.debug("KO %s: S <= 0 (S=%.4f); C_i = P_i = %.4f",i, S, p_i)
                continue
                
            # Spring calculation with R(i,j) as weights
            X = alpha ** (1.0 / S)
            a_i = 1.0 - p_i

            # Update using buddy influence
            shift = 0.0

            r_pj_values = []
            buddies_list = []
            for j, r_ij in rij_map.items():
                pj = dk_dict.get(j, 0.0)
                weight = r_ij / S
                contrib = a_i * weight * X * pj
                shift += contrib
                r_pj_values.append(r_ij * pj)
                
                
                Nij = neighbor_map[i].get(j, 0.0)
                Nj  = ko_counts.get(j, 0.0)
                
                if Nij < 0 or Nj < 0:
                    raise ValueError(f"Negative counts: i={i} j={j} Nij={Nij} Nj={Nj}")

                if Nij > Nj:
                    raise ValueError(f"Inconsistent counts (Nij > Nj): i={i} j={j} Nij={Nij} Nj={Nj}. " f"Check neighbor_map counts vs ko_counts definition.")
                    
                if r_ij > 1.0 + 1e-12:
                    raise ValueError(f"R(i,j) > 1: i={i} j={j} R={r_ij} Nij={Nij} Nj={Nj} lambda={lambda_param}")

                if verbose:
                    Nj_l = Nj + lambda_param
                    logging.debug("KO %s <- buddy %s | Nij=%.4f Nj=%.4f Nj+λ=%.4f R(i,j)=%.4f "
                        "weight=%.4f pj=%.4f contrib=%.6f",i, j, Nij, Nj, Nj_l, r_ij, weight, pj, contrib)
                
                buddies_list.append({
                    "ko": j,
                    "Nij": float(Nij),
                    "Nj": float(Nj),
                    "Rij": float(r_ij),
                    "weight": float(weight),
                    "pj": float(pj),          
                    "contrib": float(a_i * weight * X * pj)  
                })    

            new_val = min(p_i + shift, 1.0)
            new_dk[i] = new_val

            # write buddy stats for THIS i (inside loop)
            buddy_stats_map[i] = {
                "alpha": float(alpha),
                "lambda": float(lambda_param),
                "S": float(S),
                "X": float(X),
                "shift": float(shift),
                "buddy_count_used": int(len(buddies_list)),
                "buddies": sorted(buddies_list, key=lambda x: x["weight"], reverse=True),
            }

            if verbose:
                strong_buddies = sum(
                    1 for j, r_ij in rij_map.items()
                    if r_ij > 0.7 and dk_dict.get(j, 0.0) > 0.8
                )
                max_signal = max(r_pj_values) if r_pj_values else 0.0

                logging.debug(
                    "KO %s summary | P_i=%.4f C_i=%.4f shift=%.4f buddies=%d "
                    "| strong_buddies=%d | max(R*P_j)=%.4f",
                    i, p_i, new_val, shift, len(rij_map),
                    strong_buddies, max_signal
                )


        # Create output DataFrame with C_i
        df = df.copy()
        df['Dk_Neighbor'] = df['KO id'].map(new_dk).fillna(df['Dk'])
        for ko in dk_dict.keys():
            buddy_stats_map.setdefault(ko, None)
        
        df['buddy_stats'] = df['KO id'].map(buddy_stats_map)
        if return_used:
            return df, used_neighbors
        return df
        

def logm(level, mod_id, msg, *args):
    logging.log(level, "module=%s " + msg, mod_id, *args)

def logmk(level, mod_id, ko, msg, *args):
    logging.log(level, "module=%s ko=%s " + msg, mod_id, ko, *args)



class CalculateModuleProbabilities:
    _ALLOWED_FUNCS  = {"max": max, "min": min}
    _ALLOWED_BINOPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
    _ALLOWED_UNARY  = {ast.UAdd: op.pos, ast.USub: op.neg}

    KO_TOKEN_EXTRACT = re.compile(r'K(\d{5})(?:\w+)?')
    KO_TOKEN_STRICT  = re.compile(r'^K\d{5}$')

    # Matches "-K12345" only when "-" is UNARY:
    # start of string OR preceded by "(" or "," or "+" or "*" or "/" (possibly with spaces)
    _OPT_KO = re.compile(r'(?:(?<=^)|(?<=[(,+*/]))\s*-\s*(K\d{5})\b')

    @staticmethod
    def _ignore_optional_kos(eq: str) -> str:
        """
        I don't think this function is required anymore, might remove in the next version.
        Replace unary optional KO tokens like '-K12345' with a neutral value so the
        expression remains valid. Does NOT touch '1 - K12345' (binary subtraction).
        """
        CMP = CalculateModuleProbabilities  
        s = eq

        def repl(m: re.Match) -> str:
            # Decide neutral element by nearby operator context.
            # Look backwards from match start to find the last non-space char.
            start = m.start()
            j = start - 1
            while j >= 0 and s[j].isspace():
                j -= 1
            prev = s[j] if j >= 0 else ''

            # If it appears right after a '*', neutral factor is 1 (ignore in product).
            # Otherwise, use 0 (ignore in sums / function args).
            return "1" if prev == "*" else "0"

        return CMP._OPT_KO.sub(repl, s)

    @staticmethod
    def _eval_ast(node, env):
            CMP = CalculateModuleProbabilities  

            if isinstance(node, ast.Expression):
                return CMP._eval_ast(node.body, env)

            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)

            if isinstance(node, ast.Name):
                name = node.id
                if not CMP.KO_TOKEN_STRICT.fullmatch(name):
                    raise ValueError(f"Unknown variable '{name}'")
                return float(env.get(name, 0.0))

            if isinstance(node, ast.UnaryOp) and type(node.op) in CMP._ALLOWED_UNARY:
                return CMP._ALLOWED_UNARY[type(node.op)](CMP._eval_ast(node.operand, env))

            if isinstance(node, ast.BinOp) and type(node.op) in CMP._ALLOWED_BINOPS:
                left = CMP._eval_ast(node.left, env)
                right = CMP._eval_ast(node.right, env)
                return CMP._ALLOWED_BINOPS[type(node.op)](left, right)

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in CMP._ALLOWED_FUNCS:
                    func = CMP._ALLOWED_FUNCS[node.func.id]
                    args = [CMP._eval_ast(a, env) for a in node.args]
                    return float(func(*args))
                raise ValueError("Only max(...) and min(...) calls are allowed")

            if isinstance(node, ast.Tuple):
                return tuple(CMP._eval_ast(elt, env) for elt in node.elts)

            raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


    @staticmethod
    def _normalize_equation(eq):
        # Replace any Kxxxxx_suffix with plain Kxxxxx (e.g., K00844_xyz -> K00844)
        return CalculateModuleProbabilities.KO_TOKEN_EXTRACT.sub(
            lambda m: f"K{m.group(1)}", eq
        )

    @staticmethod
    def eval_equation(eq, dk_map, debug=False):
        CMP = CalculateModuleProbabilities
        eq_norm = CMP._normalize_equation(eq)
        # NEW: ignore unary optional -Kxxxxx tokens
        eq_clean = CMP._ignore_optional_kos(eq_norm)
        # Build env only for tokens present
        tokens = {f"K{m}" for m in CMP.KO_TOKEN_EXTRACT.findall(eq_clean)}
        env = {k: float(dk_map.get(k, 0.0)) for k in tokens}
        if debug:
            print("\n[eval_equation] Evaluating:", eq)
            print("Tokens:", tokens)
            print("Env:", env)


        val = float(CMP._eval_ast(ast.parse(eq_clean, mode="eval"), env))
        val_clamped = max(0.0, min(1.0, val))
        if debug: print(f"→ Result {val} (clamped {val_clamped})\n")
        return val_clamped

    @staticmethod
    def build_dk_maps_from_df(df: pd.DataFrame,before_col: str = "Dk",after_col: str = "Dk_Neighbor") -> tuple[dict, dict]:
        # Normalize KO ids
        s = df[["KO id", before_col]].copy()
        s["KO id"] = s["KO id"].astype(str).str.strip().str.upper()
        Dk = dict(zip(s["KO id"], pd.to_numeric(s[before_col], errors="coerce").fillna(0.0)))

        if after_col in df.columns:
            t = df[["KO id", after_col]].copy()
            t["KO id"] = t["KO id"].astype(str).str.strip().str.upper()
            Dk_Neighbor = dict(zip(t["KO id"], pd.to_numeric(t[after_col], errors="coerce").fillna(0.0)))
        else:
            Dk_Neighbor = Dk  # fallback

        return Dk, Dk_Neighbor

    @staticmethod
    def evaluate_step_probabilities(module_dict: dict,
                                   df: pd.DataFrame,
                                   before_col: str = "Dk",
                                   after_col: str = "Dk_Neighbor",  verbose: bool = False) -> pd.DataFrame:
        CMP = CalculateModuleProbabilities
        Dk, Dk_Neighbor = CMP.build_dk_maps_from_df(df, before_col, after_col)

        log_lines = []
        if verbose:
            log_lines.append("=" * 80)
            log_lines.append("DEBUG: STEP-LEVEL PROBABILITIES")
            log_lines.append("=" * 80)
            log_lines.append(f"Total modules in module_dict: {len(module_dict)}")

        rows_steps = []
        for mod_id, entry in module_dict.items():

            mod_eq = entry.get("module_equation", "")
            steps  = entry.get("steps", [])

            if verbose:
                logm(logging.DEBUG, mod_id, "module_equation=%s", mod_eq)
                logm(logging.DEBUG, mod_id, "n_steps=%d", len(steps))



            # Step-level
            for s in steps:
                idx = int(s["step"])
                eqn = s["equation"]

                p_b = CMP.eval_equation(eqn, Dk)
                p_a = CMP.eval_equation(eqn, Dk_Neighbor)

                # --- Extract KO IDs from equation (simple regex) ---
                kos_in_eq = sorted(set(re.findall(r"K\d{5}", eqn)))


                for ko in kos_in_eq:
                    v_b = Dk.get(ko, 0.0)
                    v_a = Dk_Neighbor.get(ko, 0.0)
                    log_lines.append(f"  {ko:<10} {v_b:<14.4f} {v_a:<20.4f}")    

                # Step-level probabilities
                rows_steps.append({
                    "module": mod_id,
                    "multiline": False,
                    "step": idx,
                    "equation": eqn,
                    "p_before": p_b,
                    "p_after":  p_a,
                })

                if verbose:
                    logm(logging.DEBUG, mod_id, "step=%d equation=%s", idx, eqn)
                    logm(logging.DEBUG, mod_id, "step=%d kos=%s", idx, ",".join(kos_in_eq))
                    for ko in kos_in_eq:
                        logmk(logging.DEBUG, mod_id, ko, "step=%d Dk=%.4f Dk_Neighbor=%.4f",
                            idx, Dk.get(ko, 0.0), Dk_Neighbor.get(ko, 0.0))
                    logm(logging.DEBUG, mod_id, "step=%d p_before=%.6f p_after=%.6f", idx, p_b, p_a)


        
        steps_df   = pd.DataFrame(rows_steps).sort_values(["module","step"]).reset_index(drop=True)
        if verbose:
            all_modules  = set(module_dict.keys())
            step_modules = set(steps_df["module"].unique())
            missing      = sorted(all_modules - step_modules)

            logging.debug("=" * 80)
            logging.debug("MODULE COVERAGE CHECK")
            logging.debug("Total modules in module_dict: %d", len(all_modules))
            logging.debug("Total modules in steps_df:     %d", len(step_modules))
            logging.debug("Missing modules:              %d", len(missing))
            if missing:
                logging.debug("First few missing: %s", ", ".join(missing[:20]))
            logging.debug("=" * 80)
        return steps_df
    
    @staticmethod
    def calculate_confidence(E: float, n: int, freq: float, 
                            thresh: float, prior_str: float) -> float:
        
        alpha_prior = freq * prior_str
        beta_prior = (1 - freq) * prior_str
        alpha_post = alpha_prior + E
        beta_post = beta_prior + (n - E)
        confidence = 1 - beta.cdf(thresh, alpha_post, beta_post)

        return confidence
    
    @staticmethod
    def calculate_module_confidence(steps_df: pd.DataFrame,
                                    module_dict: dict,
                                    genome_completeness: float,
                                    module_frequencies: dict = None,
                                    default_frequency: float = 0.5,
                                    prior_strength: float = 1.0,
                                    default_beta_thresh: float = 0.65,
                                    verbose: bool = False) -> pd.DataFrame:
        

        CMP = CalculateModuleProbabilities

        if module_frequencies is None:
            module_frequencies = {}

        # Validate genome completeness (warn but proceed)
        if genome_completeness < 0.4:
            warnings.warn(
                f"Genome completeness ({genome_completeness:.2f}) is very low (< 0.4). "
                "This may lead to spurious results. Proceed with caution.",
                UserWarning
            )
        
        if genome_completeness < 0 or genome_completeness > 1.0:
            raise ValueError(
                f"Genome completeness must be between 0 and 1.0, got {genome_completeness}"
            )
    
        beta_threshold = default_beta_thresh * genome_completeness

        rows_modules = []

        for mod_id in steps_df['module'].unique():
            # Get steps for this module
            module_steps = steps_df[steps_df['module'] == mod_id]
            
            # Calculate E (sum of step probabilities) and n_steps
            E_before = module_steps['p_before'].sum()
            E_after = module_steps['p_after'].sum()
            n_steps = len(module_steps)

            # Get module frequency (enforce minimum of 0.001 for zero values)
            freq = module_frequencies.get(mod_id, default_frequency)
            if freq == 0:
                freq = 0.001
            if freq == 1.0:
                freq = 0.999
        
            # Compute priors and posteriors explicitly (for debug visibility)
            prior_strength = n_steps*0.1
            alpha_prior = freq * prior_strength
            beta_prior = (1.0 - freq) * prior_strength

            alpha_post_before = alpha_prior + E_before
            beta_post_before  = beta_prior + (n_steps - E_before)

            alpha_post_after  = alpha_prior + E_after
            beta_post_after   = beta_prior + (n_steps - E_after)

            # Calculate confidence using Bayesian approach
            conf_before = CMP.calculate_confidence(E_before, n_steps, freq, beta_threshold, prior_strength)
            conf_after = CMP.calculate_confidence(E_after, n_steps, freq, beta_threshold, prior_strength)
                    
                
            mod_eq = module_dict.get(mod_id, {}).get("module_equation", "")
            
            rows_modules.append({
                "module": mod_id,
                "module_equation": mod_eq,
                "n_steps": n_steps,
                "E_before": E_before,
                "E_after": E_after,
                "module_frequency": freq,
                "module_probability_before": conf_before,
                "module_probability_after": conf_after,
            })

            if verbose:
                logm(logging.DEBUG, mod_id, "n_steps=%d", n_steps)
                logm(logging.DEBUG, mod_id, "E_before=%.4f E_after=%.4f", E_before, E_after)
                logm(logging.DEBUG, mod_id, "freq=%.4f prior_strength=%.4f", freq, prior_strength)
                logm(logging.DEBUG, mod_id, "alpha_prior=%.4f beta_prior=%.4f", alpha_prior, beta_prior)
                logm(logging.DEBUG, mod_id, "posterior_before alpha=%.4f beta=%.4f", alpha_post_before, beta_post_before)
                logm(logging.DEBUG, mod_id, "posterior_after  alpha=%.4f beta=%.4f", alpha_post_after, beta_post_after)
                logm(logging.DEBUG, mod_id, "beta_threshold=%.4f", beta_threshold)
                logm(logging.DEBUG, mod_id, "confidence_before=%.3f confidence_after=%.3f", conf_before, conf_after)

        
        modules_df = pd.DataFrame(rows_modules).sort_values(["module"]).reset_index(drop=True)
        
        
        return modules_df
        
    @staticmethod
    def evaluate_multiline_step_probabilities(
        module_dict_multiline: dict,
        df: pd.DataFrame,
        before_col: str = "Dk",
        after_col: str = "Dk_Neighbor",
        step_format: str = "path.step",   # "path.step" or "path_step"
        verbose: bool = False
    ) -> pd.DataFrame:
        CMP = CalculateModuleProbabilities
        Dk, Dk_Neighbor = CMP.build_dk_maps_from_df(df, before_col, after_col)

        rows_steps = []

        for mod_id, entry in module_dict_multiline.items():
            lines = entry.get("lines", [])
            if not lines:
                continue

            for line_obj in lines:
                path = int(line_obj.get("line", 0))  # your JSON uses "line"
                steps = line_obj.get("steps", [])

                for s in steps:
                    step_idx = int(s["step"])
                    eqn = s["equation"]

                    p_b = CMP.eval_equation(eqn, Dk)
                    p_a = CMP.eval_equation(eqn, Dk_Neighbor)

                    if step_format == "path_step":
                        step_label = f"{path}_{step_idx}"
                    else:
                        step_label = f"{path}.{step_idx}"

                    rows_steps.append({
                        "module": mod_id,
                        "multiline": True,
                        "step": step_label,
                        "equation": eqn,
                        "p_before": p_b,
                        "p_after":  p_a,
                    })

                    if verbose:
                        #logging.getLogger().info(f"--- Module {mod_id} ---")
                        kos = sorted(set(re.findall(r"K\d{5}", eqn)))
                        logging.debug("module=%s multiline=%s step=%s equation=%s", mod_id, True, step_label, eqn)

                        for ko in kos:
                            logging.debug(
                                "module=%s ko=%s Dk=%.4f Dk_Neighbor=%.4f", mod_id,
                                ko, Dk.get(ko, 0.0), Dk_Neighbor.get(ko, 0.0)
                            )
                        logging.debug(
                            "       p_before=%.6f  p_after=%.6f",
                            p_b, p_a
                        )


        steps_df = pd.DataFrame(rows_steps)

        if steps_df.empty:
            # return empty with expected columns for safety
            return pd.DataFrame(columns=["module","multiline","step","equation","p_before","p_after"])
        def _step_key(x):
            try:
                if isinstance(x, str) and "." in x:
                    a,b = x.split(".", 1)
                    return (int(a), int(b))
                if isinstance(x, str) and "_" in x:
                    a,b = x.split("_", 1)
                    return (int(a), int(b))
            except Exception:
                pass
            return (10**9, 10**9)

        steps_df["_sort"] = steps_df["step"].map(_step_key)
        steps_df = steps_df.sort_values(["module","_sort"]).drop(columns=["_sort"]).reset_index(drop=True)
        return steps_df




    @staticmethod
    def calculate_multiline_module_confidence_from_steps(
        multiline_steps_df: pd.DataFrame,
        module_dict_multiline: dict,
        genome_completeness: float,
        module_frequencies: dict = None,
        default_frequency: float = 0.5,
        prior_strength: float = 1.0,
        default_beta_thresh: float = 0.65,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Input:
        multiline_steps_df from evaluate_multiline_step_probabilities()

        Output:
        modules_df rows for multiline modules with SAME columns as calculate_module_confidence:
            module, module_equation, n_steps, E_before, E_after, module_frequency,
            module_probability_before, module_probability_after

        BUT E_before/E_after/n_steps correspond to the WINNING PATH (max conf_after).
        """
        CMP = CalculateModuleProbabilities
        if module_frequencies is None:
            module_frequencies = {}

        if multiline_steps_df.empty:
            return pd.DataFrame(columns=[
                "module","module_equation","n_steps","E_before","E_after",
                "module_frequency","module_probability_before","module_probability_after"
            ])

        # ---- parse "step" label into path + step_idx ----
        def _parse_path(step_label):
            if isinstance(step_label, str) and "." in step_label:
                a, _ = step_label.split(".", 1)
                return int(a)
            if isinstance(step_label, str) and "_" in step_label:
                a, _ = step_label.split("_", 1)
                return int(a)
            # fallback: treat as path 1
            return 1

        df = multiline_steps_df.copy()
        df["path"] = df["step"].map(_parse_path)

        beta_threshold = default_beta_thresh * genome_completeness

        rows_modules = []

        for mod_id, gmod in df.groupby("module", sort=False):
            # module frequency guardrails (avoid alpha/beta degeneracy)
            freq = float(module_frequencies.get(mod_id, default_frequency))
            if freq <= 0.0: freq = 0.001
            if freq >= 1.0: freq = 0.999

            # compute confidence for each path independently
            path_rows = []
            for path_id, gpath in gmod.groupby("path", sort=True):
                E_before = float(gpath["p_before"].sum())
                E_after  = float(gpath["p_after"].sum())
                n_steps  = int(len(gpath))

                # you were overriding prior_strength to n_steps*0.1 in your single-line fn
                # keep that same behavior for comparability:
                ps = n_steps * 0.1

                conf_before = CMP.calculate_confidence(E_before, n_steps, freq, beta_threshold, ps)
                conf_after  = CMP.calculate_confidence(E_after,  n_steps, freq, beta_threshold, ps)

                path_rows.append({
                    "path": path_id,
                    "n_steps": n_steps,
                    "E_before": E_before,
                    "E_after": E_after,
                    "conf_before": conf_before,
                    "conf_after": conf_after,
                })
                if verbose:
                    #logging.getLogger().info(f"--- Module {mod_id} ---")
                    logging.debug(
                        "[PATH] module=%s path=%s n_steps=%d",
                        mod_id, path_id, n_steps
                    )
                    logging.debug(
                        "       E_before=%.4f E_after=%.4f",
                        E_before, E_after
                    )
                    logging.debug(
                        "       conf_before=%.6f conf_after=%.6f",
                        conf_before, conf_after
                    )


            # choose winning path by max conf_after
            best = max(path_rows, key=lambda r: r["conf_after"])
            mod_eq = module_dict_multiline.get(mod_id, {}).get("module_equation", "")

            rows_modules.append({
                "module": mod_id,
                "module_equation": mod_eq,
                "n_steps": best["n_steps"],
                "E_before": best["E_before"],
                "E_after": best["E_after"],
                "module_frequency": freq,
                "module_probability_before": best["conf_before"],
                "module_probability_after":  best["conf_after"],
            })
            if verbose:
                logging.debug(
                    "[PATH-SELECT] module=%s best_conf=%.6f",
                    mod_id, best["conf_after"]
                )


        return pd.DataFrame(rows_modules).sort_values("module").reset_index(drop=True)



class ModuleBestPath:
    """
    Semantic inverse + best-path evaluator for BLIMMP step equations.

    - Inverts the exact serialization produced by to_symbolic_plain
    - Builds a semantic AST (KO / AND / OR)
    - Evaluates best-path score + KO set
    """

    # -------------------------
    # Internal semantic AST
    # -------------------------

    class _Node: pass

    class _KO(_Node):
        def __init__(self, kid: str):
            self.kid = kid

    class _AND(_Node):
        def __init__(self, kids: List["_Node"]):
            self.kids = kids

    class _OR(_Node):
        def __init__(self, kids: List["_Node"]):
            self.kids = kids

    # -------------------------
    # Construction
    # -------------------------

    def __init__(
        self,
        module_eq: Dict[str, Any],
        ko_df: pd.DataFrame,
        *,
        ko_id_col: str = "KO id",
        ko_prob_col: str = "Dk_Neighbor",
        score_col: str = "score",
        keep_duplicate_kos: str = "max",
    ):
        self.module_eq = module_eq
        self.pKO, self.ko_score = self._build_pko_and_score(
            ko_df,
            ko_id_col=ko_id_col,
            ko_prob_col=ko_prob_col,
            score_col=score_col,
            keep=keep_duplicate_kos,
        )


    # -------------------------
    # Public API
    # -------------------------

    def run_all(self):
        rows = []
        failures = []

        for mid, payload in self.module_eq.items():

            # single-line modules
            for st in payload.get("steps", []):
                try:
                    node = self._parse_step_equation(st["equation"])
                    score, kos = self._eval_best_path(node)
                    rows.append({
                        "module": mid,
                        "multiline": False,
                        "step": st["step"],
                        "best_path_score": score,
                        "best_path_kos": ",".join(sorted(kos)),
                    })
                except Exception as e:
                    failures.append({
                        "module": mid,
                        "kind": "step_equation",
                        "step": st["step"],
                        "error": str(e),
                        "equation_head": st["equation"][:200],
                    })

            # multiline modules
            for line in payload.get("lines", []):
                line_no = line.get("line")
                for st in line.get("steps", []):
                    try:
                        node = self._parse_step_equation(st["equation"])
                        score, kos = self._eval_best_path(node)
                        rows.append({
                            "module": mid,
                            "multiline": True,
                            "step": f"{line_no}.{st['step']}",
                            "best_path_score": score,
                            "best_path_kos": ",".join(sorted(kos)),
                        })
                    except Exception as e:
                        failures.append({
                            "module": mid,
                            "kind": "line_step_equation",
                            "step": f"{line_no}.{st['step']}",
                            "error": str(e),
                            "equation_head": st["equation"][:200],
                        })


        return pd.DataFrame(rows), pd.DataFrame(failures)

    # ============================================================
    # ==========  Semantic inverse of to_symbolic_plain ==========
    # ============================================================

    _KO_RE = re.compile(r"K\d{5}")

    def _parse_step_equation(self, expr: str) -> "_Node":
        s = expr.strip()
        s = self._unwrap_parens(s)

        # atomic KO
        if self._KO_RE.fullmatch(s):
            return self._KO(s)

        # max(A,B,...)  → OR
        if s.startswith("max(") and s.endswith(")"):
            inner = s[4:-1]
            parts = self._split_top_level(inner, ",")
            return self._OR([self._parse_step_equation(p) for p in parts])

        # noisy-OR: (1 - ((1 - A)*(1 - B)*...))
        if s.startswith("1 - ("):
            inner = self._unwrap_parens(s[4:])
            terms = self._split_top_level(inner, "*")
            kids = []
            for t in terms:
                t = self._unwrap_parens(t)
                if not t.startswith("1 - "):
                    raise ValueError(f"Invalid noisy-OR term: {t}")
                kids.append(self._parse_step_equation(t[4:].strip()))
            return self._OR(kids)

        # AND / sequence
        if "*" in s:
            parts = self._split_top_level(s, "*")
            return self._AND([self._parse_step_equation(p) for p in parts])

        raise ValueError(f"Unrecognized step equation format: {s}")

    # -------------------------
    # String helpers
    # -------------------------

    def _unwrap_parens(self, s: str) -> str:
        s = s.strip()
        while s.startswith("(") and s.endswith(")"):
            depth = 0
            ok = True
            for i, ch in enumerate(s):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if depth == 0 and i < len(s) - 1:
                    ok = False
                    break
            if not ok:
                break
            s = s[1:-1].strip()
        return s

    def _split_top_level(self, s: str, sep: str) -> List[str]:
        parts, buf, depth = [], [], 0
        for ch in s:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == sep and depth == 0:
                parts.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        return parts

    # ============================================================
    # ==================== Best-path evaluation ==================
    # ============================================================
    def _tie_score(self, kos: Set[str]) -> float:
        # Use max score among KOs used in that branch (usually 1 KO)
        best = float("-inf")
        for k in kos:
            best = max(best, self.ko_score.get(k, float("-inf")))
        return best


    def _eval_best_path(self, node: "_Node") -> Tuple[float, Set[str]]:
        if isinstance(node, self._KO):
            return float(self.pKO.get(node.kid, 0.0)), {node.kid}

        if isinstance(node, self._AND):
            score = 1.0
            path: Set[str] = set()
            for ch in node.kids:
                s, ks = self._eval_best_path(ch)
                score *= s
                path |= ks
            return score, path

        if isinstance(node, self._OR):
            best_s = -1.0
            best_path: Set[str] = set()
            best_tie = float("-inf")

            for ch in node.kids:
                s, ks = self._eval_best_path(ch)

                if s > best_s:
                    best_s, best_path = s, ks
                    best_tie = self._tie_score(ks) if abs(s - 1.0) < 1e-12 else float("-inf")
                    continue

                # tie on probability
                if abs(s - best_s) < 1e-12 and abs(s - 1.0) < 1e-12:
                    tie = self._tie_score(ks)
                    if tie > best_tie:
                        best_s, best_path, best_tie = s, ks, tie
                    elif abs(tie - best_tie) < 1e-12:
                        # deterministic fallback: pick lexicographically smallest KO-set
                        if sorted(ks) < sorted(best_path):
                            best_s, best_path, best_tie = s, ks, tie

            return (0.0 if best_s < 0 else best_s), best_path


        raise TypeError(type(node))

    # ============================================================
    # ===================== KO probability map ===================
    # ============================================================

    def _build_pko_and_score(
        self,
        df: pd.DataFrame,
        *,
        ko_id_col: str,
        ko_prob_col: str,
        score_col: str,
        keep: str,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:

        if ko_id_col not in df.columns:
            raise ValueError(f"KO id col not found: {ko_id_col!r}")
        if ko_prob_col not in df.columns:
            raise ValueError(f"KO prob col not found: {ko_prob_col!r}")
        if score_col not in df.columns:
            raise ValueError(f"Score col not found: {score_col!r}")

        tmp = df[[ko_id_col, ko_prob_col, score_col]].copy()
        tmp["KO_base"] = tmp[ko_id_col].astype(str).str.extract(r"(K\d{5})", expand=False)
        tmp["p"] = pd.to_numeric(tmp[ko_prob_col], errors="coerce")
        tmp["score"] = pd.to_numeric(tmp[score_col], errors="coerce")
        tmp = tmp.dropna(subset=["KO_base", "p"])

        # If score missing for some rows, treat as -inf so it never wins a tie
        tmp["score"] = tmp["score"].fillna(float("-inf"))

        if keep == "max":
            # probability: max over duplicates
            p_df = tmp.groupby("KO_base", as_index=False)["p"].max()
            # score: max over duplicates (best hit) — consistent with your HMM dedup logic
            s_df = tmp.groupby("KO_base", as_index=False)["score"].max()
        elif keep == "last":
            tmp = tmp.drop_duplicates("KO_base", keep="last")
            p_df = tmp[["KO_base", "p"]]
            s_df = tmp[["KO_base", "score"]]
        else:
            raise ValueError("keep_duplicate_kos must be 'max' or 'last'")

        pKO = dict(zip(p_df["KO_base"], p_df["p"].astype(float)))
        ko_score = dict(zip(s_df["KO_base"], s_df["score"].astype(float)))
        return pKO, ko_score
    

    @staticmethod
    def compute_module_best_paths(steps_df):
        module_best_rows = []

        for module, g in steps_df.groupby("module", sort=False):
            if not g["multiline"].any():
                all_kos = []

                for _, r in g.sort_values("step").iterrows():
                    if pd.notna(r["best_path_kos"]) and r["best_path_kos"] != "":
                        all_kos.extend(r["best_path_kos"].split(","))

                module_best_rows.append({"module": module,"module_best_path_kos": ",".join(sorted(set(all_kos))),})
            else:
                best_score = -1.0
                best_kos = ""
                best_ko_count = 10**9

                g = g.copy()
                g["line"] = g["step"].astype(str).str.split(".").str[0]

                for line, gl in g.groupby("line", sort=False):

                    score = 1.0
                    kos = []

                    for _, r in gl.iterrows():
                        score *= float(r["best_path_score"]) if pd.notna(r["best_path_score"]) else 0.0
                        if pd.notna(r["best_path_kos"]) and r["best_path_kos"] != "":
                            kos.extend(r["best_path_kos"].split(","))

                    kos = sorted(set(kos))
                    ko_count = len(kos)

                    if score > best_score:
                        best_score = score
                        best_kos = ",".join(kos)
                        best_ko_count = ko_count

                    elif score == best_score:
                        if ko_count < best_ko_count:
                            best_kos = ",".join(kos)
                            best_ko_count = ko_count

                module_best_rows.append({
                    "module": module,
                    "module_best_path_kos": best_kos,
                })

        return pd.DataFrame(module_best_rows)






class FileWriters:
    @staticmethod
    def ensure_dir(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


    @staticmethod
    def write_csv_outputs(df_ko: pd.DataFrame,steps_df: pd.DataFrame, modules_df: pd.DataFrame,output_prefix: str,*,basename = "blimmp"):
        """
        Write 3 CSVs:
        1) KO CSV
        2) Steps CSV
        3) Modules CSV
        Returns: {
        'ko_csv': <path>,
        'steps_csv': <path>,
        'modules_csv': <path>,
        'df_ko_out': <DataFrame used>,
        'steps_out': <DataFrame used>,
        'modules_out': <DataFrame used>,
        }
        """
        ko_path = f"{output_prefix}_{basename}_dk.csv"
        FileWriters.ensure_dir(ko_path)
        RENAME_KO_TABLE = {
            "KO id": "KO id",
            "Dk_Neighbor": "KO_conf_final",   #
            "Dk": "KO_conf_raw",    #
            "hmm_len": "hmm_len",
            "target name": "ORF_name",
            "qlen": "ORF_len",
            "E-value": "E-value",
            "score": "score",
            "i_Evalue": "domain_evalue",
            "i_score": "domain_score",
            "hmm from": "hmm from",
            "hmm to": "hmm to",
            "ali from": "ali from",
            "ali to": "ali to",
            "overlapgroup_winner": "ORF_best_match",
            "overlapgroup_winner_score": "ORF_best_match_score",
            "kofam_score_threshold": "kofam_score_threshold",
            "is_outcompeted": "is_outcompeted",
            "hit_conf": "KO_annotation_conf",
            "flag_is_dubious": "flag_is_dubious",
            "flag_is_below_kofam_threshold": "flag_is_below_kofam_threshold",
            "KO_Neighbors": "KO_Neighbors", 
            "KO_Neighbor_Count": "KO_Neighbor_Count", # int
            "Modules": "Modules",
            "count": "KO_count", # int
            "KO_freq": "KO_Background_Freq", #
        }

        df_ko = df_ko.rename(columns=RENAME_KO_TABLE)
        keep_step_cols = list(RENAME_KO_TABLE.values())
        df_ko = df_ko[keep_step_cols]
        df_ko["KO_count"] = pd.to_numeric(df_ko["KO_count"], errors="coerce")
        df_ko["KO_count"] = (df_ko["KO_count"].replace([np.inf, -np.inf], np.nan).fillna(0).astype(int))
        df_ko["KO_Neighbor_Count"] = df_ko["KO_Neighbor_Count"].astype(int)
        df_ko["KO_conf_final"] = df_ko["KO_conf_final"].round(3)
        df_ko["KO_conf_raw"] = df_ko["KO_conf_raw"].round(3)
        df_ko["KO_Background_Freq"] = df_ko["KO_Background_Freq"].round(3)
        df_ko.to_csv(ko_path, index=False)

        steps_csv_path = f"{output_prefix}_{basename}_module_steps.csv"
        FileWriters.ensure_dir(steps_csv_path)
        rename_step_prob = {
            "module": "module",
            "step": "step_id",
            "p_after": "step_probability", #
            "equation": "equation",
            "best_path_kos" : "best_path_kos",
            "best_path_score": "best_path_ko_confidence",
            "best_path_reactions": "best_path_reactions"
        }

        steps_df = steps_df.rename(columns=rename_step_prob)
        keep_step_cols = list(rename_step_prob.values())
        steps_df = steps_df[keep_step_cols]
        steps_df["step_probability"] = steps_df["step_probability"].round(3)
        steps_df["best_path_ko_confidence"] = steps_df["best_path_ko_confidence"].round(3)
        steps_df.to_csv(steps_csv_path, index=False)

        modules_csv_path = f"{output_prefix}_{basename}_module_probabilities.csv"
        FileWriters.ensure_dir(modules_csv_path)

        #Rename column names

        rename_module_prob = {
            "module": "module",
            "module_frequency": "module_frequency",
            "n_steps": "num_steps",
            "E_before": "num_steps_present_raw",
            "E_after": "num_steps_present",
            "module_probability_before": "module_confidence_raw",
            "module_probability_after": "module_confidence",
            "module_best_path_kos": "best_path",
            "module_best_path_reactions": "best_path_reactions",
        }
        modules_df = modules_df.rename(columns=rename_module_prob)
        keep_cols = list(rename_module_prob.values())
        modules_df = modules_df[keep_cols]
        modules_df["num_steps"] = modules_df["num_steps"].astype(int)
        modules_df["num_steps_present_raw"] = modules_df["num_steps_present_raw"].round(2)
        modules_df["num_steps_present"] = modules_df["num_steps_present"].round(2)

        modules_df["module_frequency"] = modules_df["module_frequency"].round(3)
        modules_df["module_confidence"] = modules_df["module_confidence"].round(3)
        modules_df["num_steps_present"] = modules_df["num_steps_present"].round(1)
        modules_df.to_csv(modules_csv_path, index=False)

        print(f"[Done.] KO-level file written to {ko_path}")
        print(f"[Done.] Step-level file written to {steps_csv_path}")
        print(f"[Done.] Module-level file written to {modules_csv_path}")

        return {
        "ko_csv": ko_path,
        "steps_csv": steps_csv_path,
        "modules_csv": modules_csv_path}
    
    
    @staticmethod
    def write_module_json(
        df: pd.DataFrame,
        modules_df: pd.DataFrame,
        steps_df: pd.DataFrame,   
        output_prefix: str,
        *,
        basename: str = "blimmp",
        module_json_dir: str| None = None,
        module_reaction_map
    ):

        def _sanitize_for_json(x):
            # None / pandas NA
            if x is None or x is pd.NA:
                return None

            # numpy scalars
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.bool_,)):
                return bool(x)
            if isinstance(x, (np.floating,)):
                x = float(x)
                return None if not math.isfinite(x) else x

            # python floats
            if isinstance(x, float):
                return None if not math.isfinite(x) else x

            # containers
            if isinstance(x, dict):
                return {str(k): _sanitize_for_json(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_sanitize_for_json(v) for v in x]

            # leave strings/ints/bools alone; stringify unknown objects
            if isinstance(x, (str, int, bool)):
                return x

            return str(x)

        

        def _json_default(o):
            if o is None:
                return None
            if o is pd.NA:
                return None
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                x = float(o)
                return None if not np.isfinite(x) else x
            if isinstance(o, (np.bool_,)):
                return bool(o)
            return str(o)

        json_path = f"{output_prefix}_{basename}_modules.json"
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

        want_cols = ["module_equation", "module_probability_before", "module_probability_after"]
        if "module_best_path_kos" in modules_df.columns:
            want_cols.append("module_best_path_kos")

        mod_prob = modules_df.set_index("module")[want_cols].to_dict(orient="index")



        df_nodes = df.copy()
        df_nodes["KO id"] = df_nodes["KO id"].astype(str).str.strip().str.upper()
        df_nodes = df_nodes[df_nodes["Modules"].notna()].copy()  # guard NaNs

        df_nodes["modules_present"] = (
            df_nodes["Modules"]
            .astype(str)
            .str.split(",")
            .apply(lambda xs: [s.strip() for s in xs if s and s.strip()])
        )

        # keep your existing explode workflow for grouping-by-module
        df_nodes = df_nodes.assign(module_list=df_nodes["Modules"].astype(str).str.split(","))
        df_nodes = df_nodes.explode("module_list")
        df_nodes["module_list"] = df_nodes["module_list"].astype(str).str.strip()
        node_fields = [
            "KO id","target name","E-value","score",
            "overlapgroup_winner","overlapgroup_winner_score",
            "overlapgroup_winner_hit_conf","overlap_relative_position_confidence",
            "kofam_score_threshold","hit_conf","flag_is_dubious","is_outcompeted",
            "flag_is_below_kofam_threshold","KO_freq","Dk","Dk_Neighbor",
            "KO_Neighbors","KO_Neighbor_Count", "buddy_stats","modules_present",
        ]
        present = [c for c in node_fields if c in df_nodes.columns]

        aggregated = {}


        # 
        steps_pack_by_module: dict[str, dict] = {}

        if steps_df is not None and not steps_df.empty:
            rename_map = {
                "module_id": "module",
                "step_no": "step",
                "step_equation": "equation",
                "step_prob_before": "p_before",
                "step_prob_after": "p_after",
                # optional if present:
                "best_path_kos": "best_path_kos",
                "best_path_score": "best_path_score",
                "best_path_reactions": "best_path_reactions",

            }
            steps_norm = steps_df.rename(
                columns={k: v for k, v in rename_map.items() if k in steps_df.columns}
            ).copy()

            required = {"module", "step", "equation", "p_before", "p_after"}
            missing = sorted(required - set(steps_norm.columns))
            if missing:
                raise ValueError(f"steps_df missing required columns: {missing}. Have: {list(steps_norm.columns)}")

            steps_norm["module"]   = steps_norm["module"].astype(str).str.strip()
            steps_norm["step"] = steps_norm["step"].astype(str).str.strip()
            steps_norm["step"] = pd.to_numeric(steps_norm["step"], errors="coerce")

            steps_norm["equation"] = steps_norm["equation"].astype(str)
            steps_norm["p_before"] = pd.to_numeric(steps_norm["p_before"], errors="coerce")
            steps_norm["p_after"]  = pd.to_numeric(steps_norm["p_after"], errors="coerce")

            # optional cols
            if "best_path_kos" in steps_norm.columns:
                steps_norm["best_path_kos"] = steps_norm["best_path_kos"].astype(str).str.strip()

            if "best_path_score" in steps_norm.columns:
                steps_norm["best_path_score"] = pd.to_numeric(steps_norm["best_path_score"], errors="coerce")
            if "best_path_reactions" in steps_norm.columns:
                steps_norm["best_path_reactions"] = steps_norm["best_path_reactions"].astype(str).fillna("").str.strip()


            def _split_step_id(x: float) -> tuple[int, float]:
                if not np.isfinite(x):
                    return (1, float("nan"))
                line = int(np.floor(x))
                inner = round((x - line) * 10, 6)  # .1 -> 1.0, .2 -> 2.0
                return (line, inner)

            for mod_id, sub in steps_norm.groupby("module", sort=False):
                sub = sub.dropna(subset=["step"]).sort_values("step")

                # detect multiline: any non-integer step id (like 1.1) OR multiple integer "lines"
                has_decimal = bool(((sub["step"] % 1) != 0).any())

                if has_decimal:
                    # build lines[]
                    sub = sub.copy()
                    sub[["line", "inner_step"]] = sub["step"].apply(lambda v: pd.Series(_split_step_id(float(v))))
                    lines_out = []
                    for line_no, gline in sub.groupby("line", sort=True):
                        gline = gline.sort_values("inner_step")
                        steps_list = []
                        for _, row in gline.iterrows():
                            step_dict = {
                                "step": float(row["inner_step"]),     # inner step within the line (e.g., 1.0)
                                "equation": row["equation"],
                                "p_before": float(row["p_before"]) if np.isfinite(row["p_before"]) else None,
                                "p_after":  float(row["p_after"]) if np.isfinite(row["p_after"]) else None,
                                
                            }
                            if "best_path_kos" in gline.columns:
                                step_dict["best_path_kos"] = row.get("best_path_kos", None)

                            if "best_path_reactions" in gline.columns:
                                step_dict["best_path_reactions"] = row.get("best_path_reactions", "") or ""

                            
                            if "best_path_score" in gline.columns:
                                v = row.get("best_path_score", None)
                                step_dict["best_path_score"] = float(v) if (v is not None and np.isfinite(v)) else None
                            steps_list.append(step_dict)

                        # optional pretty string
                        steps_inline = "; ".join([f"step {int(s['step']) if s['step']%1==0 else s['step']}: {s['equation']}" for s in steps_list])

                        lines_out.append({
                            "line": int(line_no),
                            "steps": steps_list,
                            "steps_inline": steps_inline
                        })

                    steps_pack_by_module[mod_id] = {"lines": lines_out}

                else:
                    # single-line: keep steps[] like before
                    steps_list = []
                    for _, row in sub.iterrows():
                        step_dict = {
                            "step": float(row["step"]),
                            "equation": row["equation"],
                            "p_before": float(row["p_before"]) if np.isfinite(row["p_before"]) else None,
                            "p_after":  float(row["p_after"]) if np.isfinite(row["p_after"]) else None,
                        }
                        if "best_path_kos" in sub.columns:
                            step_dict["best_path_kos"] = row.get("best_path_kos", None)

                        if "best_path_reactions" in sub.columns:
                            step_dict["best_path_reactions"] = row.get("best_path_reactions", "") or ""

                        if "best_path_score" in sub.columns:
                            v = row.get("best_path_score", None)
                            step_dict["best_path_score"] = float(v) if (v is not None and np.isfinite(v)) else None
                        steps_list.append(step_dict)

                    steps_pack_by_module[mod_id] = {"steps": steps_list}
        

        # Pre-group nodes by module id (so we can write modules that have zero nodes too)
        nodes_by_module: dict[str, pd.DataFrame] = {}
        if "module_list" in df_nodes.columns:
            for mod_id, grp in df_nodes.groupby("module_list", sort=False):
                mod_id = str(mod_id).strip()
                if mod_id:
                    nodes_by_module[mod_id] = grp

        # Write all modules that have module-level metadata OR step info
        all_mod_ids = set(mod_prob.keys())
        all_mod_ids |= set(steps_pack_by_module.keys())  # if you used my earlier name

        for mod_id in sorted(all_mod_ids):
            if not mod_id or mod_id not in mod_prob:
                # If you truly want ONLY modules that exist in modules_df, keep this guard.
                # Otherwise you can allow step-only modules too.
                continue

            meta = mod_prob.get(mod_id, {
                "module_equation": "",
                "module_probability_before": None,
                "module_probability_after": None,
                "module_best_path_kos": None,
            })

            grp = nodes_by_module.get(mod_id, None)

            # base nodes from df (limited to columns actually present)
            if grp is not None and not grp.empty:
                nodes = grp[present].to_dict(orient="records")
            else:
                nodes = []  # will become START/SINK only

            # ---- START/SINK rows (same as your code) ----
            _base_defaults = {
                "KO id": None,
                "target name": "NA",
                "E-value": None,
                "score": None,
                "overlapgroup_winner": "NA",
                "overlapgroup_winner_score": None,
                "overlapgroup_winner_hit_conf": None,
                "overlap_relative_position_confidence": None,
                "kofam_score_threshold": None,
                "hit_conf": None,
                "flag_is_dubious": False,
                "is_outcompeted": False,
                "flag_is_below_kofam_threshold": False,
                "KO_freq": None,
                "Dk": None,
                "Dk_Neighbor": None,
                "KO_Neighbors": "NA",
                "KO_Neighbor_Count": 0,
                "buddy_stats": None,
            }

            _start = _base_defaults.copy()
            _start.update({"KO id": "START", "Dk": 1.0, "Dk_Neighbor": 1.0})
            start_row = {k: v for k, v in _start.items() if k in present}

            _sink = _base_defaults.copy()
            _sink.update({"KO id": "SINK", "Dk": 1.0, "Dk_Neighbor": 1.0})
            sink_row = {k: v for k, v in _sink.items() if k in present}

            nodes.insert(0, start_row)
            nodes.append(sink_row)
            # --------------------------------------------

            entry = {
                "module_equation": meta.get("module_equation", ""),
                "module_probability_before": meta.get("module_probability_before", None),
                "module_probability_after": meta.get("module_probability_after", None),
                "best_path": meta.get("module_best_path_kos", None),  # if you added this
                "nodes": nodes,
            }

            pack = steps_pack_by_module.get(mod_id)  # contains {"steps":...} or {"lines":...}
            if pack:
                entry.update(pack)

            aggregated[mod_id] = entry



        missing_modules = sorted(set(modules_df["module"]) - set(aggregated.keys()))
        if missing_modules:
            print(f"{len(missing_modules)} modules omitted (not present in df or no nodes):")
            print(", ".join(missing_modules[:50]))

        aggregated = _sanitize_for_json(aggregated)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False, allow_nan=False)


        print(f"[Done.] Wrote {json_path} with {len(aggregated)} modules.")
        return json_path


## Main Callers

class BlimmpPipeline:
    def __init__(self, cfg: RunConfig, paths: Paths):
        self.cfg = cfg
        self.paths = paths

    def run(self):

        #File loaders
        sample_name = os.path.basename(self.cfg.input_file).split('.')[0]
        logging.info(f"Processing sample {sample_name} with sigma={self.cfg.sigma}")
        if not (0.0 <= self.cfg.sigma <= 1.0):
            raise ValueError("--sigma must be between 0 and 1")
        
        # taxonomy-driven paths
        counts_tsv, onehop_json, twohop_json, all_neighbor_json, tag = File_Helpers.lineage_paths(self.cfg.taxonomy, self.paths)
        logging.info(f"Taxonomic level chosen: {self.cfg.taxonomy}")

        ko_occ = File_Helpers.read_ko_occurrence(str(counts_tsv))
        if self.cfg.verbose:
            logging.debug("KO occurrence head:\n%s", ko_occ.head())

        #Ignore, warming up JIT
        _ = _assign_groups_numba(np.array([0.,1.]), np.array([1.,2.]), 0.6)

        hmm_hits = HMMParsers.process_domtblout(self.cfg.input_file)
        logging.info("Clustering overlapping annotations...")

        hmm_groups = Overlap.assign_overlap_groups(hmm_hits)
        # KO-level de-dup (best score per KO)
        hmm_groups_dedup = (hmm_groups.sort_values('score', ascending=False).drop_duplicates(subset='KO id', keep='first').reset_index(drop=True))

        # Reporting values
        counts = hmm_groups_dedup[["overlap_group", "KO id"]].agg(pd.Series.nunique)
        logging.info(f"Number of overlapping annotations detected: {counts['overlap_group']}")
        logging.info(f"Number of unique KEGG annotations on dataframe: {counts['KO id']}")



        #Loading the kofamdb file
        positionscored = PositionScores.winner_info_and_flags(hmm_groups_dedup, self.paths.kofam_ko_list_path)
        if self.cfg.verbose:
            logging.debug("positionscored head:\n%s", positionscored.head())


        # Build KO universe straight from my module JSONs
        ko_to_modules_str = File_Helpers.modules_to_kos(self.paths.module_json_dir)
        ko_universe = pd.DataFrame({'KO id': sorted(ko_to_modules_str.keys())})


        # Left-join observed annotations; missing KOs get defaults
        positionscored_full = ko_universe.merge(positionscored, on='KO id', how='left')

        # Defaults for KOs not present in HMM hits
        positionscored_full['hit_conf'] = positionscored_full['hit_conf'].fillna(0.0)
        positionscored_full['E-value']  = positionscored_full['E-value'].fillna(1000.0)

        # (Optional, but helps avoid surprises)
        for col in ('score', 'overlap_group'):
            if col in positionscored_full:
                positionscored_full[col] = positionscored_full[col].fillna(0)

        

        #Calculating Dk from positionscored
        dk_calculations = CalculateKOProbabilities.calculate_dk_per_ko(positionscored_full, ko_occ, verbose=self.cfg.verbose)
        logging.info("Dk calculations shape: %s", dk_calculations.shape)
        if self.cfg.verbose:
            logging.debug("Dk calculations head:\n%s", dk_calculations.head())

        #In the future you can choose two/one hop
        neighbor_map, ko_counts = NeighborCalculations.make_neighbor_dictionary(all_neighbor_json)

        #Update the Dk calculations based on the neighbors
        dk_update_calculations, used_neighbors = CalculateKOProbabilities.dk_neighbor_update(dk_calculations,neighbor_map, ko_counts, alpha=0.6, return_used=True,verbose=self.cfg.verbose)
        print("after update max abs diff:", (dk_update_calculations["Dk_Neighbor"] - dk_update_calculations["Dk"]).abs().max())

        # attach neighbor lists/counts per KO
        dk_update_calculations["KO_Neighbors"] = dk_update_calculations["KO id"].map(lambda k: ",".join(used_neighbors.get(k, [])))
        dk_update_calculations["KO_Neighbor_Count"] = dk_update_calculations["KO id"].map(lambda k: len(used_neighbors.get(k, []))).fillna(0).astype(int)

        # Add module details to the new updated df
        ko_to_modules_str = File_Helpers.modules_to_kos(self.paths.module_json_dir)
        # annotate modules and drop KOs not present in any module file
        dk_update_calculations["Modules"] = dk_update_calculations["KO id"].map(ko_to_modules_str)
        dk_update_calculations = dk_update_calculations.dropna(subset=["Modules"])

        module_eq = File_Helpers.load_module_eq(self.paths.module_eq_json)
        module_freq_dict = File_Helpers.load_module_freq(self.paths.module_frequencies)

        single_only = {m:e for m,e in module_eq.items() if e.get("steps")}
        multi_only  = {m:e for m,e in module_eq.items() if e.get("lines")}


        steps_df_single = CalculateModuleProbabilities.evaluate_step_probabilities(single_only,dk_update_calculations,before_col="Dk",after_col="Dk_Neighbor",verbose=self.cfg.verbose)
        #steps_df["multiline"] = False


        modules_df_single = CalculateModuleProbabilities.calculate_module_confidence(
        steps_df=steps_df_single,
        module_dict=single_only,
        genome_completeness=self.cfg.sigma,  
        module_frequencies=module_freq_dict,
        verbose=self.cfg.verbose)


        # 2) multiline (new)
        steps_df_multi = CalculateModuleProbabilities.evaluate_multiline_step_probabilities(
            multi_only, dk_update_calculations,
            before_col="Dk", after_col="Dk_Neighbor",
            step_format="path.step",
            verbose=self.cfg.verbose
        )

        modules_df_multi = CalculateModuleProbabilities.calculate_multiline_module_confidence_from_steps(
            steps_df_multi,
            module_dict_multiline=multi_only,
            genome_completeness=self.cfg.sigma,
            module_frequencies=module_freq_dict,
            verbose=self.cfg.verbose
        )


        best_paths = ModuleBestPath(
            module_eq=module_eq,
            ko_df=dk_update_calculations,
            ko_id_col="KO id",
            ko_prob_col="Dk_Neighbor",
            score_col="score",         
            keep_duplicate_kos="max",
        )


        best_steps_df, best_failures_df = best_paths.run_all()

        best_failures_df.to_csv("best_path_fails.csv", index=False)

        steps_df   = pd.concat([steps_df_single, steps_df_multi], ignore_index=True).sort_values(["module","multiline","step"]).reset_index(drop=True)

        steps_df["step"] = steps_df["step"].astype(str)
        best_steps_df["step"] = best_steps_df["step"].astype(str)

        steps_df = steps_df.merge(
            best_steps_df[["module","multiline","step","best_path_score","best_path_kos"]],
            on=["module","multiline","step"],
            how="left"
        )

        module_reaction_map = File_Helpers.load_module_reactions(self.paths.module_reaction_dir)
        steps_df["best_path_reactions"] = steps_df.apply(lambda row: File_Helpers.reactions_for_module_bestpath(module_reaction_map, row["module"], row["best_path_kos"]),axis=1)


        module_best_df = best_paths.compute_module_best_paths(steps_df)

        modules_df = pd.concat([modules_df_single, modules_df_multi], ignore_index=True).sort_values(["module"]).reset_index(drop=True)
        modules_df = modules_df.merge(module_best_df, on="module", how="left")
        print(modules_df.columns)
        modules_df["module_best_path_reactions"] = modules_df.apply(lambda row: File_Helpers.reactions_for_module_bestpath(module_reaction_map, row["module"], row["module_best_path_kos"]),axis=1)
        #Write files
        FileWriters.write_csv_outputs(dk_update_calculations,steps_df,modules_df,self.cfg.output_prefix,basename="BLIMMP")
        #FileWriters.write_module_json(dk_update_calculations,modules_df,self.cfg.output_prefix,basename="BLIMMP")
        FileWriters.write_module_json(dk_update_calculations, modules_df, steps_df,self.cfg.output_prefix,basename="BLIMMP",
            module_json_dir=str(self.paths.module_json_dir),
            module_reaction_map=module_reaction_map,
        )





def main():
    p = argparse.ArgumentParser(description='BLIMMP class-based pipeline (tbl/domtblout)')
    p.add_argument('file', help='Path to the .tblout or .domtblout file')
    p.add_argument('-f', '--format', choices=['tbl','domtblout'], required=True)
    p.add_argument('-s', '--sigma', type=float, required=True)
    #Easy to add if they want one or two hop/ default is 2 hop
    p.add_argument('-t', '--taxonomy', default="bacteria", metavar="NAME")
    p.add_argument('-o', '--output', required=True, help='Output prefix')
    p.add_argument('-l', '--logfile', action='store_true', help='Generate verbose logfile')
    args = p.parse_args()

    logfile_path = f"{args.output}_debug.log" if args.logfile else None
    logging.getLogger("numba").setLevel(logging.ERROR)

    logger = logging.getLogger()
    logger.handlers.clear()

    if args.logfile:
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(logfile_path, mode="w")
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    else:
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("[%(levelname)s] %(message)s")


        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)


    cfg = RunConfig(input_file=args.file, fmt=args.format, sigma=args.sigma,taxonomy=args.taxonomy, output_prefix=args.output,verbose=args.logfile,logfile_path=logfile_path)
    


    HERE = os.path.dirname(os.path.abspath(__file__))
    GD   = os.path.join(HERE, "Graph_Dependencies")
    DD   = os.path.join(HERE, "Data_Dependencies")

    ZIPS_TO_EXTRACT = {os.path.join(GD, "KEGG_Graphs_Generated_Aug25.zip") : os.path.join(GD, "KEGG_Graphs_Generated_Aug25")}

    for zip_path, extract_to in ZIPS_TO_EXTRACT.items():
        if os.path.isfile(zip_path):                   
            os.makedirs(extract_to, exist_ok=True)
            print(f"Extracting {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_to)
            os.remove(zip_path)                           
            print(f"  Done → {extract_to}")

    paths = Paths(
        counts_dir = Path(DD)/ "ATB_Taxonomy_Frequency",
        onehop_dir = Path(GD)/ "ONE_HOP_NEIGHBOR_DATA",
        twohop_dir = Path(GD)/ "TWO_HOP_NEIGHBOR_DATA",
        module_neighbor_dir   = Path(GD)/ "MODULE_ALL_NEIGHBOR_DATA",
        module_eq_json = Path(GD)/ "KEGG_Module_Equations_Jan26.json",
        module_json_dir = Path(GD)/ "KEGG_Graphs_Generated_Aug25",
        kofam_ko_list_path = Path(DD)/ "ko_list.txt",
        module_frequencies = Path(DD)/ "module_freq.txt",
        module_reaction_dir = Path(GD)/ "module_ko_reaction.json"
    )


    BlimmpPipeline(cfg, paths).run()


if __name__ == "__main__":
    main() 