#!/usr/bin/env python3

import ROOT as R
import argparse

def check(rootFile_path, treename, n):
    """Print branch values for the first n events in a ROOT TTree"""
    rootFile = R.TFile.Open(rootFile_path, 'READ')
    
    if not rootFile or rootFile.IsZombie():
        print(f"Error: Cannot open file {rootFile_path}")
        return
    
    tree = rootFile.Get(treename)
    if not tree:
        print(f"Error: Cannot find tree '{treename}' in {rootFile_path}")
        print(f"Available keys: {[key.GetName() for key in rootFile.GetListOfKeys()]}")
        rootFile.Close()
        return
    
    total_entries = tree.GetEntries()
    n = min(n, total_entries)
    
    print(f"\nTree: {treename}")
    print(f"Total entries: {total_entries}")
    print(f"Printing first {n} events\n")
    
    for i in range(n):
        tree.GetEntry(i)
        print("*", 25*"-", f"Event {i}/{total_entries-1}", 25*"-", "*")
        
        for branch in tree.GetListOfBranches():
            name = branch.GetName()
            value = getattr(tree, name)
            
            # Handle different types of branch values
            if hasattr(value, '__len__') and not isinstance(value, str):
                # Array or vector
                print(f"  {name:30s}: [{', '.join(str(v) for v in value)}]")
            else:
                # Scalar value
                print(f"  {name:30s}: {value}")
        print()
    
    rootFile.Close()

def main():
    parser = argparse.ArgumentParser(
        description="Print branch values from ROOT TTree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print first 5 events from 'event' tree
  %(prog)s data.root
  
  # Print first 10 events from 'mytree' tree
  %(prog)s data.root -t mytree -n 10
  
  # Print only 1 event to check structure
  %(prog)s data.root -n 1
        """
    )
    parser.add_argument("input", help="Input ROOT file path")
    parser.add_argument("--tree", "-t", default="event", 
                       help="Name of the TTree to process (default: event)")
    parser.add_argument("--n", "-n", type=int, default=5, 
                       help="Number of events to print (default: 5)")

    args = parser.parse_args()
    check(args.input, args.tree, args.n)

if __name__ == "__main__":
    main()