#!/usr/bin/env python3
"""
Belle1 Configuration Manager

This module is responsible for reading and managing configuration information in Belle1 data, including:
- background_files_merged.dat: Background file mapping
- run_database_energy.dat: Run database energy information

Author: GitHub Copilot
Date: September 10, 2025
"""

import os
import sys
import pickle
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("belle1_mc_gen.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Belle1MCGen")

@dataclass
class BackgroundFileInfo:
    """Background file information class"""
    experiment: int
    run_min: int
    run_max: int
    file_path: str
    
    def is_run_in_range(self, run: int) -> bool:
        """Check if run number is within range"""
        return self.run_min <= run <= self.run_max
    
    def __str__(self) -> str:
        return f"Exp:{self.experiment} Run:{self.run_min}-{self.run_max} File:{self.file_path}"

@dataclass
class RunDatabaseEntry:
    """Run database entry class"""
    experiment: int
    run: int
    ecms: float = 10.58  # Default energy is 10.58 GeV
    ip: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # (x, y, z, sigx, sigy, sigz)
    luminosity: float = 0.0
    # Additional fields for mc generation
    events: int = 0
    luminosity_fraction: float = 0.0
    background_file: str = ""
    dataset: str = ""       # Dataset, such as "4S", "4S_offres", "4S_scan"

class Belle1MCGen:
    """Belle1 configuration manager class, responsible for reading and managing Belle1 configuration information"""
    
    def __init__(self, exps:List[int], dts:List[str], base_dir: str, nevents: int,lsf_q = "s"):
        """
        Initialize the configuration manager
        
        Args:
            base_dir: Base directory containing configuration files
        """
        self.base_dir = base_dir
        self.background_file_db_path = os.path.join(base_dir, "background_files_merged.dat")
        self.run_database_path = os.path.join(base_dir, "run_database_energy.dat")
        
        # Data storage
        self.background_files: List[BackgroundFileInfo] = []
        self.run_database: Dict[Tuple[int, int], RunDatabaseEntry] = {}
        
        # Job management attributes
        self.selected_runs = []  # List of selected runs
        self.total_luminosity = 0.0
        self.job_ids = []  # For tracking submitted job IDs
        
        # Job parameters
        self.total_events = nevents
        self.experiments = exps  # empty means all valid experiments
        self.dataset_types = dts  # empty means all dataset types
        self.process_name = ""
        self.model_file = ""
        self.work_dir = "./test"
        self.script_dir = "./scripts"
        self.queue = lsf_q
        self.dry_run = False
        self.verbose = False

        # marker
        self.evet_distributed = False
        
        # Cache file paths
        self.cache_dir = os.path.join(base_dir, ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.bg_cache_path = os.path.join(self.cache_dir, "background_files.pkl")
        self.run_db_cache_path = os.path.join(self.cache_dir, "run_database.pkl")
        
        # Load configuration
        self.load_background_files()
        self.load_run_database()
        self.load_selected_runs()
        
        logger.info("Belle1 Configuration Manager initialized")
    

    def load_background_files(self) -> None:
        """Load background file information"""
        logger.info(f"Loading background file information: {self.background_file_db_path}")
        
        # Try to load from cache
        if os.path.exists(self.bg_cache_path):
            try:
                with open(self.bg_cache_path, 'rb') as f:
                    self.background_files = pickle.load(f)
                logger.info(f"Loaded {len(self.background_files)} background file records from cache")
                return
            except Exception as e:
                logger.warning(f"Failed to load background file information from cache: {e}")
        
        # If cannot load from cache, load from original file
        if not os.path.exists(self.background_file_db_path):
            logger.warning(f"Background file database does not exist: {self.background_file_db_path}")
            return
        
        try:
            self.background_files = []
            with open(self.background_file_db_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) != 4:
                        logger.warning(f"Invalid background file database line: {line}")
                        continue
                    
                    try:
                        exp = int(parts[0])
                        run_min = int(parts[1])
                        run_max = int(parts[2])
                        bg_file = parts[3]
                        
                        self.background_files.append(BackgroundFileInfo(
                            experiment=exp, 
                            run_min=run_min, 
                            run_max=run_max, 
                            file_path=bg_file
                        ))
                    except ValueError as e:
                        logger.warning(f"Error parsing background file line: {line}, error: {e}")
            
            logger.info(f"Loaded {len(self.background_files)} background file records from file")
            
            # Save to cache
            self._save_to_cache(self.background_files, self.bg_cache_path)
            
        except Exception as e:
            logger.error(f"Error loading background file information: {e}")
    

    def load_run_database(self) -> None:
        """Load run database information"""
        logger.info(f"Loading run database: {self.run_database_path}")
        
        # Try to load from cache
        if os.path.exists(self.run_db_cache_path):
            try:
                with open(self.run_db_cache_path, 'rb') as f:
                    self.run_database = pickle.load(f)
                logger.info(f"Loaded {len(self.run_database)} run database records from cache")
                return
            except Exception as e:
                logger.warning(f"Failed to load run database from cache: {e}")
        
        # If cannot load from cache, load from original file
        if not os.path.exists(self.run_database_path):
            logger.warning(f"Run database does not exist: {self.run_database_path}")
            return
        
        try:
            self.run_database = {}
            with open(self.run_database_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:  # At least 5 fields needed: experiment number, run number, dataset name, luminosity, energy value
                        logger.warning(f"Invalid run database line: {line}")
                        continue
                    
                    try:
                        exp = int(parts[0])
                        run = int(parts[1])
                        dataset_type = parts[2]    # Dataset name (4S, 4S_offres, 4S_scan etc.)
                        luminosity = float(parts[3])  # Luminosity
                        ecms = float(parts[4])     # Energy value
                        
                        # Create basic run database entry
                        entry = RunDatabaseEntry(
                            experiment=exp,
                            run=run,
                            ecms=ecms,
                            luminosity=luminosity,
                            ip=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # All IP information set to 0
                            dataset=dataset_type,  # Directly use dataset type from database
                            background_file=self.get_background_file(exp, run)
                        )
                        
                        self.run_database[(exp, run)] = entry
                    except ValueError as e:
                        logger.warning(f"Error parsing run database line: {line}, error: {e}")
            
            logger.info(f"Loaded {len(self.run_database)} run database records from file")
            
            # Save to cache
            self._save_to_cache(self.run_database, self.run_db_cache_path)
            
        except Exception as e:
            logger.error(f"Error loading run database: {e}")
            self.run_database = {}


    def _save_to_cache(self, data: Any, cache_path: str) -> None:
        """Save data to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Data saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save data to cache: {e}")


    @lru_cache(maxsize=128)
    def get_background_file(self, experiment: int, run: int) -> Optional[str]:
        """
        Get background file for the specified experiment and run
        
        Args:
            experiment: Experiment number
            run: Run number
            
        Returns:
            Background file path, or default background file if not found, or None
        """
        # Search for matching entry in background file database
        for bg_info in self.background_files:
            if bg_info.experiment == experiment and bg_info.is_run_in_range(run):
                if os.path.exists(bg_info.file_path):
                    return bg_info.file_path
                else:
                    logger.warning(f"Corresponding background file does not exist: {bg_info.file_path}")
                    break
        
        logger.error(f"Cannot find background file: Experiment {experiment}, Run {run}")
        return None
    

    def load_selected_runs(self):
        """
        Load runs matching dataset criteria
        
        Returns:
            bool: Whether loading was successful
        """
        logger.info("Loading run data...")
        
        dts_selected = self.dataset_types
        exps_selected = self.experiments
        if len(dts_selected) == 0 and len(exps_selected) == 0:
            self.selected_runs = list(self.run_database.values())
            logger.info("No dataset types or experiments specified, loading all valid runs")
            return True 
        
        # Check if configuration manager has loaded run database
        if not self.run_database:
            logger.error("Failed to load run database")
            return False
        
        try:
            # Clear existing run data
            self.selected_runs = []
            self.total_luminosity = 0.0
            
            # Get run data from configuration manager
            for key, entry in self.run_database.items():
                exp, run_num = key

                if len(exps_selected) != 0 and exp not in exps_selected:
                    continue
                if len(dts_selected) != 0 and entry.dataset not in dts_selected:
                    continue

                # Add to selected run list
                self.selected_runs.append(entry)
                self.total_luminosity += entry.luminosity
            
            logger.info(f"Loaded {len(self.selected_runs)} run records, total luminosity: {self.total_luminosity/1000:.5f} /fb")
            
            return len(self.selected_runs) > 0
            
        except Exception as e:
            logger.error(f"Error loading run data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


    def get_avail_exps(self) -> Set[int]:
        """Get all available experiment numbers"""
        exps = {entry.experiment for entry in self.selected_runs}
        logger.info(f"Available experiments: {sorted(exps)}")
        return exps
    

    def get_avail_exp_runs(self) -> Dict[int, List[int]]:
        """Get all run numbers for the specified experiment"""
        exps = self.get_avail_exps()
        available_runs = {}
        for exp in exps:
            runs = sorted(entry.run for entry in self.selected_runs if entry.experiment == exp)
            # Convert runs to range string like 2:5,7:9,...
            ranges = []
            if runs:
                start = prev = runs[0]
                for r in runs[1:]:
                    if r == prev + 1:
                        prev = r
                    else:
                        ranges.append(f"{start}:{prev}" if start != prev else f"{start}")
                        start = prev = r
                ranges.append(f"{start}:{prev}" if start != prev else f"{start}")
            range_str = ",".join(ranges)
            available_runs[exp] = runs
            logger.info(f"Experiment {exp} runs: {range_str}")
        #logger.info(f"Available runs for experiments: {available_runs}")
        return available_runs 
    

    def get_avail_dts(self) -> Set[str]:
        """Get all available dataset types"""
        dts = {entry.dataset for entry in self.selected_runs}
        logger.info(f"Available dataset types: {sorted(dts)}")
        return dts
    

    def calculate_luminosity_fractions(self):
        """Calculate luminosity fractions for each run"""
        logger.info("Calculating luminosity fractions...")
        
        if not self.selected_runs:
            logger.warning("No run data, cannot calculate luminosity fractions")
            return
        
        # Get total luminosity, if 0 then use average distribution
        total_lumi = sum(run.luminosity for run in self.selected_runs)
        
        if total_lumi <= 0:
            import sys
            sys.exit("Total luminosity is zero, cannot calculate luminosity fractions")
        
        # Calculate luminosity fraction for each run
        for run in self.selected_runs:
            # Directly calculate and store the run's luminosity as a fraction of total luminosity
            run.luminosity_fraction = run.luminosity / total_lumi
        
        # Check if sum of all fractions is close to 1.0 (considering floating point precision)
        total_fraction = sum(run.luminosity_fraction for run in self.selected_runs)
        if abs(total_fraction - 1.0) > 1e-10:
            logger.warning(f"Sum of luminosity fractions ({total_fraction}) does not match expected value (1.0), normalizing")
            # Normalize all fractions to ensure sum is 1.0
            scale_factor = 1.0 / total_fraction
            for run in self.selected_runs:
                run.luminosity_fraction *= scale_factor
    

    def distribute_events(self):
        """
        Distribute events according to luminosity ratio
        
        Args:
            total_events: Total number of events to distribute, if None then use self.total_events
            
        Returns:
            int: Number of events actually distributed
        """
        total_events = self.total_events
            
        logger.info(f"Distributing {total_events} events...")
        
        # Reset all run event counts
        for run in self.selected_runs:
            run.events = 0
        
        # Ensure luminosity fractions have been calculated
        if not hasattr(self.selected_runs[0], 'luminosity_fraction') or self.selected_runs[0].luminosity_fraction == 0:
            self.calculate_luminosity_fractions()
        
        # Record distributed events total, used to track rounding errors
        distributed_events = 0
        
        # Distribute events using luminosity fractions with rounding
        for run in self.selected_runs:
            run.events = round(total_events * run.luminosity_fraction)
            distributed_events += run.events
        
        # Handle rounding errors, ensure total event count is correct
        remaining_events = total_events - distributed_events
        if remaining_events != 0:
            sorted_runs = sorted(self.selected_runs, key=lambda r: r.luminosity_fraction, reverse=True)
            
            # Distribute remaining events (positive or negative) to top runs one by one
            for i in range(abs(remaining_events)):
                if remaining_events > 0:
                    sorted_runs[i % len(sorted_runs)].events += 1
                else:
                    sorted_runs[i % len(sorted_runs)].events -= 1
        
        # Print detailed distribution information
        if logger.isEnabledFor(logging.DEBUG):
            for run in self.selected_runs:
                logger.debug(f"Experiment {run.experiment} Run {run.run}: {run.events} events (luminosity fraction: {run.luminosity_fraction:.4f})")
        
        # Count actual distributed events
        actual_events = sum(run.events for run in self.selected_runs)
        logger.info(f"Distributed {actual_events} events according to luminosity fractions")

        self.evet_distributed = True
        
        return actual_events


    def calculate_events_per_exp(self):
        """
        Calculate number of events per experiment based on already distributed events
        
        Returns:
            dict: Number of events per experiment
        """
        logger.info("Calculating events per experiment summary...")
        if not self.evet_distributed:
            self.distribute_events()
        
        # Group runs by experiment and sum their events
        from collections import defaultdict
        experiment_data = defaultdict(list)
        experiment_luminosity = defaultdict(float)
        events_per_experiment = defaultdict(int)
        
        # Group run data by experiment
        for run in self.selected_runs:
            exp = run.experiment
            experiment_data[exp].append(run)
            experiment_luminosity[exp] += run.luminosity
            events_per_experiment[exp] += run.events
        
        # Log event distribution results for each experiment
        for exp in sorted(events_per_experiment.keys()):
            logger.info(f"Experiment {exp}: {events_per_experiment[exp]} events, {experiment_luminosity[exp]/1000:.5f} /fb")
        
        return events_per_experiment


    
    '''
    def scale_luminosity_by_energy(self):
        """
        Scale luminosity by energy (considering energy dependence of e+e- -> mu+mu- cross section)
        """
        logger.info("Scaling luminosity by energy...")
        
        # Only scale when there is run data
        if not self.selected_runs:
            logger.warning("No run data to scale")
            return
        
        # Scale luminosity proportionally to e+e- -> mu+mu- cross section
        for run in self.selected_runs:
            energy = run.ecms
            xsec = self._xsec_ee_to_mumu(energy)
            
            # Scale luminosity
            ref_energy = 10.58  # Y(4S) energy, used as reference
            ref_xsec = self._xsec_ee_to_mumu(ref_energy)
            
            # Adjust luminosity to be proportional to reference cross section
            run.luminosity *= (xsec / ref_xsec)
        
        # Recalculate total luminosity
        self.total_luminosity = sum(run.luminosity for run in self.selected_runs)
        
        logger.info(f"Scaled total luminosity: {self.total_luminosity:.2f} 1/pb")
    
    def _xsec_ee_to_mumu(self, energy):
        """Calculate e+e- -> mu+mu- cross section at given energy (in nb)"""
        import math
        
        alpha = 1.0 / 137.0
        m_mu = 0.10566  # GeV
        s = energy * energy
        
        sigma = (4.0 * math.pi * alpha * alpha) / (3.0 * s)
        sigma *= (1.0 + 2.0 * m_mu * m_mu / s)
        sigma *= 0.389e6  # Convert to nb
        
        return sigma
    '''
        
    def submit_generation_jobs(self):
        """
        Submit generation jobs - group by experiment and background file
        Each job limited to max 9999 events
        """
        gen_dir = os.path.join(self.work_dir, "gen")
        os.makedirs(gen_dir, exist_ok=True)
        logger.info("Submitting generation jobs (grouped by exp and bg file, max 9999 events per job)...")
        
        # Check if model file exists
        if not os.path.exists(self.model_file):
            logger.error(f"Model file does not exist: {self.model_file}")
            return 0
            
        # If there is no run data, cannot submit jobs
        if not self.selected_runs:
            logger.error("No run data, cannot submit jobs")
            return 0
        
        # Group runs by (experiment, background_file)
        from collections import defaultdict
        grouped = defaultdict(list)
        for run in self.selected_runs:
            if run.events > 0:
                grouped[(run.experiment, run.background_file)].append(run)
        
        # Maximum events per job
        MAX_EVENTS_PER_JOB = 9999 # >= 10000 will cause error
        
        jobs_submitted = 0
        for (exp, bg_file), runs in grouped.items():
            # Sort runs by run number
            runs.sort(key=lambda r: r.run)
            
            # Prepare output directory
            output_dir = gen_dir
            #output_dir = os.path.join(gen_dir, f"e{exp}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Split into multiple jobs if total events > MAX_EVENTS_PER_JOB
            current_job_runs = []
            current_job_events = 0
            job_index = 0
            
            for run in runs:
                # If adding this run would exceed the limit, submit current job and start a new one
                if current_job_events + run.events > MAX_EVENTS_PER_JOB and current_job_runs:
                    # Submit current job
                    min_run = min(r.run for r in current_job_runs)
                    max_run = max(r.run for r in current_job_runs)
                    job_id = f"{min_run}" if min_run == max_run else f"{min_run}_{max_run}"
                    output_file = f"e{exp}_r{job_id}_j{job_index}.dat"
                    log_file = f"e{exp}_r{job_id}_j{job_index}.log"
                    
                    cmd = [
                        "bsub", "-q", self.queue,
                        "-o", os.path.join(output_dir, log_file),
                        "basf2", self.model_file, "--",
                        str(exp), str(min_run), str(current_job_events), bg_file,
                        os.path.join(output_dir, output_file)
                    ]
                    
                    try:
                        logger.info(f"Submitting job: exp={exp}, runs={len(current_job_runs)}, "
                                   f"run_range={min_run}-{max_run}, events={current_job_events}, bg={bg_file}")
                        if not self.dry_run:
                            import subprocess
                            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                            job_info = result.stdout.strip()
                            logger.info(f"Job submitted: {job_info}")
                            self.job_ids.append(job_info.split("<")[1].split(">")[0] if "<" in job_info else "unknown")
                            jobs_submitted += 1
                        else:
                            logger.info(f"[DRY RUN] Would submit command: {' '.join(cmd)}")
                    except Exception as e:
                        logger.error(f"Error submitting job for exp {exp} bg {bg_file}: {e}")
                    
                    # Reset for next job
                    current_job_runs = []
                    current_job_events = 0
                    job_index += 1
                
                # Add current run to the job
                current_job_runs.append(run)
                current_job_events += run.events
            
            # Submit the last job if any runs left
            if current_job_runs:
                min_run = min(r.run for r in current_job_runs)
                max_run = max(r.run for r in current_job_runs)
                job_id = f"{min_run}" if min_run == max_run else f"{min_run}_{max_run}"
                output_file = f"e{exp}_r{job_id}_j{job_index}.dat"
                log_file = f"e{exp}_r{job_id}_j{job_index}.log"
                
                cmd = [
                    "bsub", "-q", self.queue, 
                    "-o", os.path.join(output_dir, log_file),
                    "basf2", self.model_file, "--",
                    str(exp), str(min_run), str(current_job_events), bg_file,
                    os.path.join(output_dir, output_file)
                ]
                
                try:
                    logger.info(f"Submitting job: exp={exp}, runs={len(current_job_runs)}, "
                               f"run_range={min_run}-{max_run}, events={current_job_events}, bg={bg_file}")
                    if not self.dry_run:
                        import subprocess
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        job_info = result.stdout.strip()
                        logger.info(f"Job submitted: {job_info}")
                        self.job_ids.append(job_info.split("<")[1].split(">")[0] if "<" in job_info else "unknown")
                        jobs_submitted += 1
                    else:
                        logger.info(f"[DRY RUN] Would submit command: {' '.join(cmd)}")
                except Exception as e:
                    logger.error(f"Error submitting job for exp {exp} bg {bg_file}: {e}")
        
        logger.info(f"Submitted {jobs_submitted} generation jobs")
        return jobs_submitted
    

    def submit_generation_jobs_by_run(self):
        """
        Submit generation jobs - one job per run (backup method)
        
        Returns:
            int: Number of jobs submitted
        """
        gen_dir = os.path.join(self.work_dir, "gen")
        os.makedirs(gen_dir, exist_ok=True)

        logger.info("Submitting generation jobs (one job per run)...")
        
        # Check if model file exists
        if not os.path.exists(self.model_file):
            logger.error(f"Model file does not exist: {self.model_file}")
            return 0
            
        # If there is no run data, cannot submit jobs
        if not self.selected_runs:
            logger.error("No run data, cannot submit jobs")
            return 0
        
        # Create output directory structure
        os.makedirs(gen_dir, exist_ok=True)
        
        # Get valid runs (event count > 0)
        valid_runs = [run for run in self.selected_runs if run.events > 0]
        if not valid_runs:
            logger.warning("No runs to process (all runs have zero events)")
            return 0
        
        # Group by experiment, only for display information
        from collections import defaultdict
        runs_by_exp = defaultdict(list)
        for run in valid_runs:
            runs_by_exp[run.experiment].append(run)
        
        # Display run and event count for each experiment
        for exp, runs in sorted(runs_by_exp.items()):
            exp_total_events = sum(run.events for run in runs)
            logger.info(f"Experiment {exp}: {len(runs)} runs, total events: {exp_total_events}")
        
        # Submit job for each run
        jobs_submitted = 0
        for run in valid_runs:
            experiment = run.experiment
            
            # Prepare output directory
            output_dir = os.path.join(gen_dir, f"e{experiment}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if background file exists
            bg_file = run.background_file
            if not bg_file or not os.path.exists(bg_file):
                logger.error(f"Background file does not exist: {bg_file}")
                continue
            
            # Prepare command
            cmd = [
                "bsub", 
                "-q", self.queue,
                "-o",
                f"{output_dir}/e{experiment}_r{run.run}.log",
                "basf2",
                self.model_file,
                "--",
                str(experiment),
                str(run.run),
                str(run.events),
                bg_file,
                f"{output_dir}/e{experiment}_r{run.run}.dat"
            ]
            
            # Submit job
            try:
                logger.info(f"Submitting generation job for experiment {experiment} run {run.run}: {run.events} events")
                if not self.dry_run:
                    import subprocess
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    job_info = result.stdout.strip()
                    logger.info(f"Job submitted: {job_info}")
                    self.job_ids.append(job_info.split("<")[1].split(">")[0] if "<" in job_info else "unknown")
                    jobs_submitted += 1
                else:
                    logger.info(f"[DRY RUN] Would submit command: {' '.join(cmd)}")
            except Exception as e:
                logger.error(f"Error submitting generation job for experiment {experiment} run {run.run}: {e}")
                continue
        
        logger.info(f"Submitted {jobs_submitted} individual generation jobs")
        return jobs_submitted
        

    def submit_generation_exp_run(self, experiment: int, run_number: int, n: int):
        """
        Submit generation job for a specific experiment and run.

        Args:
            experiment (int): Experiment number
            run_number (int): Run number
            n (int): Number of events

        Returns:
            int: Number of jobs submitted (0 or 1)
        """
        MAX_EVENTS_PER_JOB = 9999
        if n > MAX_EVENTS_PER_JOB:
            logger.error(f"Requested event number ({n}) exceeds maximum allowed ({MAX_EVENTS_PER_JOB}) for a single job.")
            return 0

        gen_dir = os.path.join(self.work_dir, "gen")
        os.makedirs(gen_dir, exist_ok=True)

        logger.info(f"Submitting generation job for exp={experiment}, run={run_number}, n={n}")

        # Find background file for this exp/run
        bg_file = self.get_background_file(experiment, run_number)
        if not bg_file or not os.path.exists(bg_file):
            logger.error(f"Background file does not exist for exp={experiment}, run={run_number}: {bg_file}")
            return 0

        # Check model file
        if not os.path.exists(self.model_file):
            logger.error(f"Model file does not exist: {self.model_file}")
            return 0

        output_file = os.path.join(gen_dir, f"e{experiment}_r{run_number}_manual.dat")
        log_file = os.path.join(gen_dir, f"e{experiment}_r{run_number}_manual.log")
        cmd = [
            "bsub", "-q", self.queue,
            "-o", log_file,
            "basf2", self.model_file, "--",
            str(experiment), str(run_number), str(n), bg_file, output_file
        ]

        try:
            logger.info(f"Submitting job: {' '.join(cmd)}")
            if not self.dry_run:
                import subprocess
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                job_info = result.stdout.strip()
                logger.info(f"Job submitted: {job_info}")
                self.job_ids.append(job_info.split("<")[1].split(">")[0] if "<" in job_info else "unknown")
                return 1
            else:
                logger.info(f"[DRY RUN] Would submit command: {' '.join(cmd)}")
                return 1
        except Exception as e:
            logger.error(f"Error submitting job for exp={experiment}, run={run_number}: {e}")
            return 0


    def submit_generation_phokhara_bin_batch(self):
        """
        generation flat for phokhara bin batch jobs

        Returns:
            int: Number of jobs submitted (0 or 1)
        """

        gen_dir = os.path.join(self.work_dir, "gen")
        os.makedirs(gen_dir, exist_ok=True)

        self.experiments = [51]  # only use experiment 51 and 4S
        self.dts = ["4S"]
        self.load_selected_runs()
        runs_map = self.get_avail_exp_runs()
        runs = runs_map[51][:200]
        #runs = [5, 56, 66, 75, 82]
        print(len(runs))

        i = 0
        jobs_submitted = 0
        for run in runs:
            logger.info(f"submitting phokhara bin batch for run {run}")
            bg_file = self.get_background_file(51, run)
            cmd = [
                "bsub", 
                "-q", self.queue,
                "-o",
                f"{gen_dir}/e{51}_r{run}.log",
                "basf2",
                self.model_file,
                "--",
                str(51),
                str(run),
                str(5000),
                bg_file,
                f"{gen_dir}/e{51}_r{run}.dat",
                str(i)
            ]
            i += 1
            try:
                logger.info(f"Submitting generation job for experiment {51} run {run}, bin {i}")
                if not self.dry_run:
                    import subprocess
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    job_info = result.stdout.strip()
                    logger.info(f"Job submitted: {job_info}")
                    self.job_ids.append(job_info.split("<")[1].split(">")[0] if "<" in job_info else "unknown")
                    jobs_submitted += 1
                else:
                    logger.info(f"[DRY RUN] Would submit command: {' '.join(cmd)}")
            except Exception as e:
                logger.error(f"Error submitting generation job for experiment {51} run {run}: {e}")
                continue
        
        logger.info(f"Submitted {jobs_submitted} individual generation jobs")
        return jobs_submitted

    
    def submit_gsim_jobs(self):
        """
        Submit GSIM jobs by finding generated dat files and creating specific scripts for each run.
        
        Returns:
            int: Number of jobs submitted
        """
        sim_dir = os.path.join(self.work_dir, "sim")
        gen_dir = os.path.join(self.work_dir, "gen")
        work_base_dir = os.path.join(self.work_dir, "sim_script")
        os.makedirs(sim_dir, exist_ok=True)
        os.makedirs(work_base_dir, exist_ok=True)
        
        logger.info("Submitting GSIM jobs by discovering input files from: %s", gen_dir)
        
        if not os.path.isdir(gen_dir):
            logger.error(f"Generation directory does not exist, cannot find input for GSIM: {gen_dir}")
            return 0

        import glob
        import re
        import subprocess
        
        jobs_submitted = 0
        
        # Find all potential input dat files from the generation step
        input_files = glob.glob(os.path.join(gen_dir, "**", "e*_r*.dat"), recursive=True)
        
        if not input_files:
            logger.warning(f"No input .dat files found in {gen_dir}. Nothing to simulate.")
            return 0
            
        logger.info(f"Found {len(input_files)} potential input files for GSIM.")

        # Create a quick lookup map for run data
        run_map = {(r.experiment, r.run): r for r in self.selected_runs}

        for dst_path in input_files:
            filename = os.path.basename(dst_path)
            # Try to extract experiment and run number from filename
            match = re.match(r'e(\d+)_r(\d+)', filename)
            if not match:
                logger.warning(f"Could not parse experiment and run number from filename, skipping: {filename}")
                continue
            
            experiment = int(match.group(1))
            run_number = int(match.group(2))

            # Find the corresponding run data to get background file etc.
            run_data = run_map.get((experiment, run_number))
            if not run_data:
                logger.warning(f"No run database entry found for exp {experiment}, run {run_number}. Skipping file {filename}.")
                continue

            # --- Create GSIM files inside run_work_dir ---

            # 1. Create GSIM input file (gsim.dat)
            gsim_input_file = os.path.join(work_base_dir, f"e{experiment}_r{run_number}_gsim.dat")
            self._create_gsim_input_file(gsim_input_file, experiment, run_number)

            # 2. Create BASF script file (gsim.basf)
            output_mdst_file = os.path.join(sim_dir, f"e{experiment}_r{run_number}.mdst")
            basf_script_file = os.path.join(work_base_dir, f"e{experiment}_r{run_number}_gsim.basf")
            self._create_gsim_basf_script(basf_script_file, experiment, run_number, 
                                         output_mdst_file, dst_path)

            # 3. Create a robust wrapper script that sets env and runs basf
            wrapper_script_file = os.path.join(work_base_dir, f"e{experiment}_r{run_number}_gsim_run.sh")
            self._create_gsim_wrapper_script(wrapper_script_file, experiment, run_data.background_file, basf_script_file)

            # 4. Submit the wrapper script directly
            log_file = os.path.join(sim_dir, f"e{experiment}_r{run_number}_gsim.log")
            cmd = [
                "bsub",
                "-q", self.queue,
                "-oo", log_file,
                os.path.abspath(wrapper_script_file),
            ]
            
            # Submit job
            try:
                logger.info(f"Submitting GSIM job for exp={experiment} run={run_number} with input {filename}")
                if not self.dry_run:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    job_info = result.stdout.strip()
                    logger.info(f"Job submitted: {job_info}")
                    self.job_ids.append(job_info.split("<")[1].split(">")[0] if "<" in job_info else "unknown")
                    jobs_submitted += 1
                else:
                    logger.info(f"[DRY RUN] Would submit command: {' '.join(cmd)}")
            except Exception as e:
                logger.error(f"Error submitting GSIM job for exp={experiment} run={run_number}: {e}")
                logger.error(f"Failed command: {' '.join(cmd)}")
                continue
        
        logger.info(f"Submitted {jobs_submitted} GSIM jobs")
        return jobs_submitted
    
    
    def _create_gsim_input_file(self, filename, experiment, run_number):
        """Create GSIM input configuration file"""
        with open(filename, 'w') as f:
            f.write("LIST\n")
            f.write("DEBU 0 0 0 (main)\n")
            f.write("SWIT 1 3 0 0 0 0 1 0\n")
            f.write("OPTI 2\n")
            f.write("TRIGGER 2\n")
            f.write("CUTS 0.0001 0.0001   0.01   0.001   0.001  0.0001  0.0001\n")
            f.write("VRTX  0.0     0.0     0.0      100.0E-04   5.0E-04   0.37\n")
            f.write("KINE 3\n")
            f.write("HADR 4\n")
            f.write("SVD 1 (geometry) 1 (Hit record)\n")
            f.write("CDC 1 (geometry) 1 (Hit record)\n")
            f.write("ECL 1 (geometry) 1 (Hit record)\n")
            f.write("EFC 1 (geometry) 1 (Hit record)\n")
            f.write("ACC 1 (geometry) 1 (Hit record)\n")
            f.write("TOF 1 (geometry) 1 (Hit record)\n")
            f.write("KLM 1 (geometry) 1 (Hit record)\n")
            f.write("BP  1 (Beam pipe)\n")
            f.write("CRY 1 (Cryostat)\n")
            f.write("MAG 1 (Coil)\n")
            f.write("TRG 0 (Trigger Hit record)\n")
            f.write("BFLD 21\n")
            f.write(f"EXPN {experiment}\n")
            f.write(f"RUNG {run_number} 1\n")
            f.write("WSAG 1\n")
            f.write("XTFN 2\n")
            f.write("CRES 1\n")
            f.write("RSC2 1.0\n")
            f.write("TANL 1\n")
            f.write("RNDM  9876  54321\n")
            f.write("TIME 20000 0 0       (TIME LEFT FOR UGLAST)\n")
            f.write("END\n")


    def _create_gsim_basf_script(self, filename, experiment, run, output_file, input_dst, tsim = False):
        """Create BASF script for GSIM simulation - based on mcprod5-e07.basf"""
        with open(filename, 'w') as f:
            if experiment > 30:
                f.write("path add_module main ProcessHeader\n")
                f.write("path add_module main cdctable genunpak\n")
                f.write("path add_module main gsim acc_mc calsvd addbg\n")
            else:
                f.write("path add_module main cdctable genunpak\n")
                f.write("path add_module main bpsmear gsim acc_mc calsvd addbg\n")
            
            if experiment > 30:
                f.write("path add_module main tof_datT0TS tsimtof calcdc l4 evtime l0svd tsimsvd\n")
            else:
                f.write("path add_module main tof_datT0TS tsimtof calcdc l4 evtime l0svd\n")
            
            f.write("path add_module main reccdc recsvd\n")
            f.write("path add_module main trasan TOFt0 trak trkmgr AnadEdx ext\n")
            f.write("path add_module main rectof rececl_cf rececl_match rececl_gamma rececl_pi0\n")
            f.write("path add_module main rec_acc muid_set muid_dec klid efcclust\n")
            f.write("path add_module main v0finder rec2mdst evtvtx evtcls\n")
            
            if tsim:
                f.write("path add_module main tsimskin rectrg\n")
            f.write("\n")

            # GSIM input file
            f.write(f"module put_parameter gsim GSIM_INPUT\\e{experiment}_r{run}_gsim.dat\n")
            f.write("module put_parameter addbg LUMDEP\\1\n")
            
            if experiment > 30 and experiment != 73:
                f.write("module put_parameter addbg SEQRD\\0\n")
                f.write("module put_parameter addbg RANRD\\1\n")
                f.write("\n")
            
            # add bpsmear IP parameters ,tempolarily using fixed values
            f.write("module put_parameter bpsmear ip_nominal_x\0.04596")
            f.write("module put_parameter bpsmear ip_nominal_y\0.04777")
            f.write("module put_parameter bpsmear ip_nominal_z\-0.2282")
            f.write("module put_parameter bpsmear sigma_ip_x\0.007375")
            f.write("module put_parameter bpsmear sigma_ip_y\0.0004463")
            f.write("module put_parameter bpsmear sigma_ip_z\0.3190 ")
            
            if tsim:
                f.write("module put_parameter rectrg Time_window_lo\\-1.0\n")
                f.write("module put_parameter rectrg Time_window_hi\\1.0\n")
                f.write("\n")

            f.write("module put_parameter evtcls classification_level\\0\n")
            f.write("module put_parameter l4 debug\\0\n")
            f.write("\n")
            f.write("initialize\n")
            f.write("\n")
            
            # Table save commands - from mcprod5-e07.basf
            f.write("table save mdst_all\n")
            f.write("table save evtcls_all\n")
            f.write("table save evtvtx_all\n")
            f.write("table save gsim_rand\n")
            f.write("table save hepevt_all\n")
            f.write("table save mctype\n")
            f.write("table save level4_all\n")
            f.write("\n")
            f.write("table save bgtbl_info\n")
            f.write("table save dattof_trgl0\n")
            f.write("table save reccdc_timing\n")
            
            if tsim:
                f.write("table save trg_all\n")
                f.write("table save mid_all\n")
                f.write("table save tsimgdl_all\n")

            f.write(f"\noutput open {os.path.abspath(output_file)}\n")
            f.write(f"process_event {os.path.abspath(input_dst)}\n")
            f.write("terminate\n")


    def _create_gsim_wrapper_script(self, filename, experiment, background_file, basf_script_path):
        """Create wrapper script that sets up environment and runs BASF - based on rec.csh"""
        basf_script_path = os.path.abspath(basf_script_path)
        env_dir = os.path.dirname(basf_script_path)

        # Belle level selection based on rec.csh logic
        if experiment <= 27:
            belle_level = "b20030807_1600"  # SVD1 period
        else:
            belle_level = "b20090127_0910"  # SVD2 period

        with open(filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# GSIM wrapper script for experiment {experiment}\n")
            f.write("# Based on rec.csh logic\n")
            f.write("\n")
            f.write("# Set Belle environment variables BEFORE sourcing bashrc_general\n")
            f.write(f"export BELLE_LEVEL={belle_level}\n")
            f.write("export BELLE_DEBUG=opt\n")
            
            # CERN_LEVEL only for SVD2 and later (exp >= 31, per rec.csh)
            if experiment >= 31:
                f.write("export CERN_LEVEL=2006\n")
            
            f.write("export USE_GRAND_REPROCESS_DATA=1\n")
            f.write("\n")
            f.write("# Source Belle environment\n")
            f.write("source /sw/belle/local/etc/bashrc_general\n")
            f.write("\n")
            f.write("# Set BASF parameters (after sourcing)\n")
            f.write("export BASF_NPROCESS=1\n")  
            #f.write("export BASF_USER_INIT=geant_init.so\n")
            #f.write("export BASF_USER_IF=basfsh.so\n")
            f.write(f'export ADDBG_DAT="{background_file}"\n')
            f.write("\n")
            f.write("# Change to script directory\n")
            f.write(f"cd {env_dir}\n")
            f.write("\n")
            f.write("date\n")
            f.write(f'echo "Starting GSIM for experiment {experiment}"\n')
            f.write(f'echo "BELLE_LEVEL: {belle_level}"\n')
            f.write(f'echo "Working directory: $(pwd)"\n')
            f.write(f'echo "Background file: {background_file}"\n')
            f.write("\n")
            f.write("# Run BASF and filter out empty lines and '0: ' lines\n")
            f.write(f"basf {basf_script_path} 2>&1 | grep -v -E '^$|^0: *$'\n")
            f.write("EXIT_CODE=${PIPESTATUS[0]}\n")
            f.write("\n")
            f.write("date\n")
            f.write('echo "GSIM completed with exit code: $EXIT_CODE"\n')
            f.write("\n")
            f.write("exit $EXIT_CODE\n")

        os.chmod(filename, 0o755)
    

    
def main():
    parser = argparse.ArgumentParser(description="Belle1 MC generation")

    # Info queries
    parser.add_argument("--ava_exps", action="store_true", help="list available experiments")
    parser.add_argument("--ava_dts", action="store_true", help="list available dataset types")
    parser.add_argument("--ava_run", action="store_true", help="list available exp,run")
    parser.add_argument("--exp_events", action="store_true", help="show events per experiment")

    # Optional parameters
    parser.add_argument("--base_dir", type=str, default="/gpfs/group/belle2/users2022/wangz/SignalMC_belle1/database", help="database directory")
    parser.add_argument("--queue", type=str, default="s", help="LSF queue name")
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument("--dry_run", action="store_true", help="print commands only")

    # MC generation
    parser.add_argument("--gen", action="store_true", help="submit generation jobs")
    parser.add_argument("--gen_by_run", action="store_true", help="submit generation jobs (one per run)")
    parser.add_argument("--sim", action="store_true", help="submit simulation jobs")
    parser.add_argument("--top_dir", type=str, help="parent directory for gen, sim, work")
    parser.add_argument("--model_file", type=str, help="generator model file")
    parser.add_argument("--n", type=int, default=10000, help="total events to generate")
    parser.add_argument("--dts", type=str, default=[], nargs='+', help="dataset types (4S, 4S_offres, etc.)")
    parser.add_argument("--exps", type=int, default=[], nargs='+', help="experiment numbers (7, 9, 11, etc.)")

    # Add CLI for specific exp/run generation
    parser.add_argument("--gen_exp_run", action="store_true", help="submit generation job for specific experiment and run")
    parser.add_argument("--exp", type=int, help="experiment number for --gen_exp_run")
    parser.add_argument("--run", type=int, help="run number for --gen_exp_run")
    parser.add_argument("--n_run", type=int, default=7500, help="number of events for --gen_exp_run")
    
    parser.add_argument("--batch_bin_production", action="store_true", help="test generation flat for phokhara bin batch jobs")

    args = parser.parse_args()
    
    # Simple validation
    if args.gen and (not args.top_dir or not args.model_file):
        parser.error("--gen requires --top_dir and --model_file")
    if args.sim and not args.top_dir:
        parser.error("--sim requires --top_dir")
    if args.batch_bin_production and (not args.top_dir or not args.model_file):
        parser.error("--gen requires --top_dir and --model_file")

    # Initialize
    MCGen = Belle1MCGen(base_dir=args.base_dir, dts=args.dts, exps=args.exps, nevents=args.n, lsf_q=args.queue)
    MCGen.dry_run = args.dry_run
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    MCGen.distribute_events()
    # Handle info queries
    if args.ava_exps:
        MCGen.get_avail_exps()
    if args.ava_dts:
        MCGen.get_avail_dts()
    if args.ava_run:
        MCGen.get_avail_exp_runs()
    if args.exp_events:
        MCGen.calculate_events_per_exp()
    
    # Handle MC operations
    if args.gen or args.gen_by_run:
        MCGen.distribute_events()
        MCGen.model_file = args.model_file
        MCGen.work_dir = args.top_dir
        
        if args.gen:
            MCGen.submit_generation_jobs()
        if args.gen_by_run:
            MCGen.submit_generation_jobs_by_run()
        
    if args.sim:
        MCGen.work_dir = args.top_dir
        MCGen.submit_gsim_jobs()

    if args.gen_exp_run:
        MCGen.model_file = args.model_file
        MCGen.work_dir = args.top_dir
        MCGen.submit_generation_exp_run(args.exp, args.run, args.n_run)
    
    if args.batch_bin_production:
        MCGen.model_file = args.model_file
        MCGen.work_dir = args.top_dir
        MCGen.submit_generation_phokhara_bin_batch()


if __name__ == "__main__":
    main()
