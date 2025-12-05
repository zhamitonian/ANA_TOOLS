#!/usr/bin/env python3
"""
Belle Run Database Manager

Standalone module for managing Belle experiment run information.
Handles loading and querying run database from run_database_energy.dat.

Author: GitHub Copilot
Date: December 4, 2025
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict


@dataclass
class RunEntry:
    """Belle run database entry"""
    experiment: int
    run: int
    dataset: str
    luminosity: float
    energy: float
    
    def __str__(self) -> str:
        return f"Exp:{self.experiment} Run:{self.run} Dataset:{self.dataset} Lumi:{self.luminosity:.3f} Energy:{self.energy:.4f}"


class BelleRunManager:
    """
    Belle experiment run information manager
    
    Responsible for:
    - Loading run database from run_database_energy.dat
    - Querying and filtering runs by experiment/dataset/energy
    - Grouping runs for batch processing
    - Calculating luminosity statistics
    """
    
    def __init__(self, database_path: str):
        """
        Initialize run manager
        
        Args:
            database_path: Path to run_database_energy.dat file
        """
        self.database_path = database_path
        self.run_database: Dict[Tuple[int, int], RunEntry] = {}
        
    def load_database(self) -> None:
        """Load run database from file"""
        if not os.path.exists(self.database_path):
            raise FileNotFoundError(f"Run database file not found: {self.database_path}")
        
        self.run_database.clear()
        
        with open(self.database_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    print(f"Warning: Skipping invalid line {line_num}: {line}")
                    continue
                
                try:
                    exp = int(parts[0])
                    run = int(parts[1])
                    dataset = parts[2]
                    lumi = float(parts[3])
                    energy = float(parts[4])
                    
                    entry = RunEntry(
                        experiment=exp,
                        run=run,
                        dataset=dataset,
                        luminosity=lumi,
                        energy=energy
                    )
                    self.run_database[(exp, run)] = entry
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing line {line_num}: {line} ({e})")
                    continue
        
        print(f"Loaded {len(self.run_database)} run entries from database")
    
    def get_all_runs(self) -> List[RunEntry]:
        """Get all run entries"""
        return list(self.run_database.values())
    
    def get_run(self, experiment: int, run: int) -> Optional[RunEntry]:
        """Get specific run entry"""
        return self.run_database.get((experiment, run))
    
    def filter_runs(self, 
                   experiments: Optional[Set[int]] = None,
                   datasets: Optional[Set[str]] = None,
                   energy_range: Optional[Tuple[float, float]] = None) -> List[RunEntry]:
        """
        Filter runs by criteria
        
        Args:
            experiments: Set of experiment numbers to include (None for all)
            datasets: Set of dataset types to include (None for all)
            energy_range: (min_energy, max_energy) tuple (None for all)
            
        Returns:
            List of matching RunEntry objects
        """
        filtered = []
        
        for entry in self.run_database.values():
            # Filter by experiment
            if experiments is not None and entry.experiment not in experiments:
                continue
            
            # Filter by dataset
            if datasets is not None and entry.dataset not in datasets:
                continue
            
            # Filter by energy range
            if energy_range is not None:
                min_e, max_e = energy_range
                if not (min_e <= entry.energy <= max_e):
                    continue
            
            filtered.append(entry)
        
        return filtered
    
    def group_by_experiment(self, runs: List[RunEntry]) -> Dict[int, List[RunEntry]]:
        """Group runs by experiment number"""
        grouped = defaultdict(list)
        for entry in runs:
            grouped[entry.experiment].append(entry)
        return dict(grouped)
    
    def group_by_dataset(self, runs: List[RunEntry]) -> Dict[str, List[RunEntry]]:
        """Group runs by dataset type"""
        grouped = defaultdict(list)
        for entry in runs:
            grouped[entry.dataset].append(entry)
        return dict(grouped)
    
    def group_by_exp_dataset(self, runs: List[RunEntry]) -> Dict[Tuple[int, str], List[int]]:
        """
        Group run numbers by (experiment, dataset)
        
        Returns:
            Dictionary with (exp, dataset) as key and sorted list of run numbers as value
        """
        grouped = defaultdict(list)
        
        for entry in runs:
            grouped[(entry.experiment, entry.dataset)].append(entry.run)
        
        # Sort run numbers for each group
        for key in grouped:
            grouped[key].sort()
        
        return dict(grouped)
    
    def get_available_experiments(self) -> Set[int]:
        """Get all available experiment numbers"""
        return {entry.experiment for entry in self.run_database.values()}
    
    def get_available_datasets(self) -> Set[str]:
        """Get all available dataset types"""
        return {entry.dataset for entry in self.run_database.values()}
    
    def get_available_energies(self) -> Set[float]:
        """Get all available beam energies"""
        return {entry.energy for entry in self.run_database.values()}
    
    def calculate_total_luminosity(self, runs: List[RunEntry]) -> float:
        """Calculate total integrated luminosity for given runs"""
        return sum(entry.luminosity for entry in runs)
    
    def get_statistics(self, runs: Optional[List[RunEntry]] = None) -> Dict[str, any]:
        """
        Get statistics for runs
        
        Args:
            runs: List of runs to analyze (None for all runs)
            
        Returns:
            Dictionary with statistics
        """
        if runs is None:
            runs = self.get_all_runs()
        
        if not runs:
            return {
                'total_runs': 0,
                'total_luminosity': 0.0,
                'experiments': set(),
                'datasets': set(),
                'energies': set()
            }
        
        return {
            'total_runs': len(runs),
            'total_luminosity': self.calculate_total_luminosity(runs),
            'experiments': {r.experiment for r in runs},
            'datasets': {r.dataset for r in runs},
            'energies': {r.energy for r in runs},
            'luminosity_by_exp': {
                exp: self.calculate_total_luminosity(runs_list)
                for exp, runs_list in self.group_by_experiment(runs).items()
            },
            'luminosity_by_dataset': {
                ds: self.calculate_total_luminosity(runs_list)
                for ds, runs_list in self.group_by_dataset(runs).items()
            }
        }
    
    def print_summary(self, runs: Optional[List[RunEntry]] = None) -> None:
        """Print summary of runs"""
        stats = self.get_statistics(runs)
        
        print("\n" + "="*60)
        print("Belle Run Database Summary")
        print("="*60)
        print(f"Total runs: {stats['total_runs']}")
        print(f"Total luminosity: {stats['total_luminosity']:.3f} pb⁻¹ ({stats['total_luminosity']/1000:.3f} fb⁻¹)")
        
        print(f"\nExperiments: {sorted(stats['experiments'])}")
        print(f"Datasets: {sorted(stats['datasets'])}")
        print(f"Energies: {sorted(stats['energies'])}")
        
        if 'luminosity_by_exp' in stats:
            print("\nLuminosity by experiment:")
            for exp in sorted(stats['luminosity_by_exp'].keys()):
                lumi = stats['luminosity_by_exp'][exp]
                print(f"  Exp {exp:2d}: {lumi:8.3f} pb⁻¹ ({lumi/1000:6.3f} fb⁻¹)")
        
        if 'luminosity_by_dataset' in stats:
            print("\nLuminosity by dataset:")
            for ds in sorted(stats['luminosity_by_dataset'].keys()):
                lumi = stats['luminosity_by_dataset'][ds]
                print(f"  {ds:15s}: {lumi:8.3f} pb⁻¹ ({lumi/1000:6.3f} fb⁻¹)")
        
        print("="*60 + "\n")


def main():
    """Command line interface for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Belle Run Database Manager")
    parser.add_argument("database", nargs='?', 
                       default="/gpfs/home/belle2/wangz/.config/belle/run_database_energy.dat",
                       help="Path to run_database_energy.dat (default: %(default)s)")
    parser.add_argument("--exp", type=int, nargs='+', help="Filter by experiment(s)")
    parser.add_argument("--dataset", type=str, nargs='+', help="Filter by dataset(s)")
    parser.add_argument("--summary", action="store_true", help="Print summary")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = BelleRunManager(args.database)
    manager.load_database()
    
    # Filter runs if requested
    runs = manager.get_all_runs()
    if args.exp or args.dataset:
        exps = set(args.exp) if args.exp else None
        dss = set(args.dataset) if args.dataset else None
        runs = manager.filter_runs(experiments=exps, datasets=dss)
    
    # Print summary
    if args.summary:
        manager.print_summary(runs)
    else:
        for run in runs[:10]:  # Print first 10
            print(run)
        if len(runs) > 10:
            print(f"... and {len(runs) - 10} more runs")


if __name__ == "__main__":
    main()
