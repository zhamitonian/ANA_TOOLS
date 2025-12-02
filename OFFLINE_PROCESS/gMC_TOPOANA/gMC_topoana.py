import ROOT
import sys
import os
import re
from typing import Optional, List , Tuple
import subprocess

def find_decay_indices(file_path:str, search_final:str, search_mediate:Optional[str] = None, exclude_mediate:Optional[str] = None) -> List[int]:
    if search_mediate is None:
        search_mediate = search_final
    with open(file_path, 'r') as file:
        paragraphs = file.read().split('\n\n')
    indexes = []
    for paragraph in paragraphs:
        lines = paragraph.split('\n')
        if search_final.strip() in lines[-1]:
            if exclude_mediate and any(exclude_mediate.strip() in line for line in lines):
                continue
            if search_mediate and not any(search_mediate.strip() in line for line in lines):
                continue
            match = re.search(r'iDcyTr:\s*(\d+)', lines[0])
            if match :
                index = match.group(1) 
                indexes.append(index)
    return indexes

def gMC_topoana(input_rootFile:str, tree_name:str, channel_to_filter:Optional[List[Tuple[str, str]]]= None )-> str:
    """
    Run gMC_topoana on the specified input ROOT file and tree.
    
    Args:
        input_rootFile (str): Path to the input ROOT file.
        tree_name (str): Name of the tree to process.
        channel_to_filter (Optional[List[Tuple[str, str]]]): [(mediate_state, final_state)].
    Returns:
        str: the root file path being processed (topo and filter specific channel).
            such as: [(" phi --> K+ K- " , " e+ e- ---> K+ K+ K- K- "),...]
    """

    # Calculate the output path based on input path
    input_dir = os.path.dirname(input_rootFile)
    output_prefix = os.path.join(input_dir, "topoana")

    # Update the topo_info.card with the input file and output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    topo_card_path = os.path.join(script_dir, "topo_info.card")
    with open(topo_card_path, "r") as f:
        lines = f.readlines()
    # The input file path is on line 9 (index 8)
    # The output file path(only need prefix) is on line 16 (index 15)
    lines[8] = f"     {input_rootFile}\n"
    lines[15] = f"    {output_prefix}\n"
    lines[20] = f"    {tree_name}\n"  
        
    with open("topo_info.card", "w") as f:
        f.writelines(lines)
        
    # Run the topology analysis
    result = subprocess.run(["topoana.exe", "topo_info.card"], 
                           capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: Topology analysis failed")
        print(result.stderr)
        sys.exit(1)


    if channel_to_filter is not None:
        decay_indexes = []
        for mediate_state, final_state in channel_to_filter:
            decay_indexes += find_decay_indices(f"{output_prefix}.txt", final_state,  mediate_state)
            print(decay_indexes)
        
        # Handle case when no matching decays are found
        if not decay_indexes:
            print("Warning: No matching decay channels found to filter. Returning original topoana output.")
            return f"{output_prefix}.root"
        
        cut_string = " "
        for index in decay_indexes:
            cut_string += f'&& (iDcyTr != {index}) '
        cut_string = cut_string[3:]  # Remove leading '&& '
        print(f"Cut string: {cut_string}")

        df = ROOT.RDataFrame(tree_name, f"{output_prefix}.root")
        df = df.Filter(cut_string, "Filter by decay indices")
        df.Report().Print()

        filtered_rootPath = f"{input_dir}/Background.root"
        df.Snapshot("event", filtered_rootPath)
        return filtered_rootPath
    else:
        return f"{output_prefix}.root"

        
# Example usage:
#gMC_topoana("/gpfs/group/belle2/users2022/wangz/data_gMC/tagged_ISRphiKK_MC/gMC_3Cfit_4S_hadron/only_4S/processed_temp.root","event",  [( " phi --> K+ K- " , " e+ e- ---> K+ K+ K- K- ")])
#gMC_topoana("/gpfs/group/belle2/users2022/wangz/data_gMC/tagged_ISRphiKK_MC/gMC_3Cfit_4S_hadron/only_4S/processed.root","event")




