#!/usr/bin/env python3
"""
Compare merged graphs (ligand + protein) with NO cross edges:
- One with covalent protein connectivity
- One with spatial protein connectivity
"""

import os
import sys
import torch
import subprocess

# Add the preprocess directory to path
sys.path.append('preprocess')
from single_graph_processing_pdb import process_protein_pdb_ligand_style as process_protein_covalent, process_ligand_sdf
from single_graph_processing_pdb_protein import process_protein_pdb_ligand_style as process_protein_spatial

def create_merged_no_cross_graphs():
    """Create merged graphs with NO cross edges - just ligand + protein side by side."""
    test_dir = "test_data"
    comparison_dir = os.path.join(test_dir, "merged_no_cross")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Process both ligand-protein pairs
    protein_files = {
        "aaaa": "test_data/data/aaaa_pocket.pdb",
        "3tf7": "test_data/data/3tf7_pocket.pdb"
    }
    
    ligand_files = {
        "aaaa": "test_data/data/aaaa_ligand.sdf",
        "3tf7": "test_data/data/3tf7_ligand.sdf"
    }
    
    for name, pdb_path in protein_files.items():
        ligand_path = ligand_files[name]
        
        if os.path.exists(pdb_path) and os.path.exists(ligand_path):
            try:
                print(f"Processing {name.upper()} complex...")
                
                # Process ligand (same for both)
                print(f"  Loading ligand...")
                ligand_graph = process_ligand_sdf(ligand_path)
                
                # Process protein with COVALENT bonds
                print(f"  Processing protein with COVALENT bonds...")
                protein_covalent = process_protein_covalent(pdb_path)
                
                # Process protein with SPATIAL connectivity
                print(f"  Processing protein with SPATIAL connectivity...")
                protein_spatial = process_protein_spatial(pdb_path, r_cut=6.0)
                
                # Create merged graphs (NO cross edges)
                print(f"  Creating merged graphs (NO cross edges)...")
                
                # Merged with covalent protein
                merged_covalent = merge_ligand_protein_no_cross(ligand_graph, protein_covalent, name)
                covalent_path = os.path.join(comparison_dir, f"merged_covalent_{name}.pt")
                torch.save(merged_covalent, covalent_path)
                print(f"  Saved merged covalent: {name}")
                
                # Merged with spatial protein
                merged_spatial = merge_ligand_protein_no_cross(ligand_graph, protein_spatial, name)
                spatial_path = os.path.join(comparison_dir, f"merged_spatial_{name}.pt")
                torch.save(merged_spatial, spatial_path)
                print(f"  Saved merged spatial: {name}")
                
                # Print comparison stats
                print(f"\n{name.upper()} MERGED COMPARISON (NO CROSS EDGES):")
                print(f"  Covalent merged: {merged_covalent.x.shape[0]} nodes, {merged_covalent.edge_index.shape[1]} edges")
                print(f"  Spatial merged:  {merged_spatial.x.shape[0]} nodes, {merged_spatial.edge_index.shape[1]} edges")
                print(f"  Edge difference: {merged_spatial.edge_index.shape[1] - merged_covalent.edge_index.shape[1]} more edges in spatial")
                print()
                
            except Exception as e:
                print(f"Failed to process {name}: {e}")
        else:
            print(f"Files not found for {name}: PDB={os.path.exists(pdb_path)}, SDF={os.path.exists(ligand_path)}")
    
    return comparison_dir

def merge_ligand_protein_no_cross(ligand_graph, protein_graph, name):
    """Merge ligand and protein graphs with NO cross edges - just side by side."""
    from torch_geometric.data import Data
    
    # Get dimensions
    ligand_nodes = ligand_graph.x.shape[0]
    protein_nodes = protein_graph.x.shape[0]
    total_nodes = ligand_nodes + protein_nodes
    
    # Combine node features
    # Add origin flag: 0 for ligand, 1 for protein
    ligand_x = torch.cat([ligand_graph.x, torch.zeros(ligand_nodes, 1)], dim=1)
    protein_x = torch.cat([protein_graph.x, torch.ones(protein_nodes, 1)], dim=1)
    combined_x = torch.cat([ligand_x, protein_x], dim=0)
    
    # Combine positions
    combined_pos = torch.cat([ligand_graph.pos, protein_graph.pos], dim=0)
    
    # Combine edge indices (offset protein edges by ligand node count)
    ligand_edges = ligand_graph.edge_index
    protein_edges = protein_graph.edge_index + ligand_nodes
    
    # Combine edge attributes
    combined_edge_attr = torch.cat([ligand_graph.edge_attr, protein_graph.edge_attr], dim=0)
    
    # Create origin tracking for edges
    ligand_edge_origin = torch.zeros(ligand_edges.shape[1], dtype=torch.long)
    protein_edge_origin = torch.ones(protein_edges.shape[1], dtype=torch.long)
    combined_edge_origin = torch.cat([ligand_edge_origin, protein_edge_origin], dim=0)
    
    # Combine all edges (NO cross edges)
    combined_edge_index = torch.cat([ligand_edges, protein_edges], dim=1)
    
    # Create origin tracking for nodes
    origin_nodes = torch.cat([torch.zeros(ligand_nodes, dtype=torch.long), 
                             torch.ones(protein_nodes, dtype=torch.long)], dim=0)
    
    # Create the merged graph
    merged_graph = Data(
        x=combined_x,
        pos=combined_pos,
        edge_index=combined_edge_index,
        edge_attr=combined_edge_attr,
        origin_nodes=origin_nodes,
        origin_edges=combined_edge_origin,
        name=f"merged_{name}"
    )
    
    return merged_graph

def visualize_merged_no_cross(comparison_dir):
    """Use the existing visualize.py to visualize merged graphs."""
    print("\n" + "="*60)
    print("VISUALIZING MERGED GRAPHS (NO CROSS EDGES): COVALENT vs SPATIAL")
    print("="*60)
    
    # Find all .pt files in the comparison directory
    pt_files = [f for f in os.listdir(comparison_dir) if f.endswith('.pt')]
    
    if not pt_files:
        print("No .pt files found in comparison directory!")
        return
    
    print(f"Found {len(pt_files)} merged graphs to visualize:")
    for file in sorted(pt_files):
        print(f"  - {file}")
    
    # Visualize each graph using the existing visualize.py
    for file in sorted(pt_files):
        file_path = os.path.join(comparison_dir, file)
        print(f"\n--- Visualizing {file} ---")
        
        try:
            # Run 3D visualization
            print("3D visualization:")
            subprocess.run(['python', 'visualize.py', '--path', file_path, '--mode', '3d'], check=True)
            
            # Run 2D visualization
            print("2D visualization:")
            subprocess.run(['python', 'visualize.py', '--path', file_path, '--mode', '2d'], check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to visualize {file}: {e}")
        except Exception as e:
            print(f"Error visualizing {file}: {e}")

def main():
    """Main function."""
    print("Creating merged graphs (NO CROSS EDGES) with covalent vs spatial protein processing...")
    comparison_dir = create_merged_no_cross_graphs()
    
    print(f"\nMerged graphs (NO CROSS EDGES) saved to: {comparison_dir}")
    
    # Visualize the merged graphs
    visualize_merged_no_cross(comparison_dir)

if __name__ == "__main__":
    main()
