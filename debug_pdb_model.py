#!/usr/bin/env python3
"""
Clean debug script for ETNN PDB model testing.
Run with: python debug_pdb_model.py experiment=standard dataset=pdb_general_experiments
"""

import os
import pandas as pd
import hydra
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

import utils
from etnn.pdbbind.pdbbind import PDBBindCC

def create_test_index():
    """Create a test index CSV that points to our test data"""
    df = pd.DataFrame({
        'Target ChEMBLID': ['target_aaaa', 'target_3tf7'],
        'Molecule ChEMBLID': ['mol_aaaa', 'mol_3tf7'],
        '-logAffi': [7.5, 8.2],
    })
    
    os.makedirs('test_data', exist_ok=True)
    csv_path = 'test_data/index_debug.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Created test index: {csv_path}")
    return csv_path

def create_test_graphs():
    """Create individual ligand and protein graphs from our test data"""
    from preprocess.single_graph_processing_pdb import process_ligand_sdf, process_protein_pdb_ligand_style
    
    print("🛠️  Creating test graphs...")
    
    # Create directories
    os.makedirs('test_data/debug_graphs/ligand', exist_ok=True)
    os.makedirs('test_data/debug_graphs/protein', exist_ok=True)
    
    # Process test data files
    test_files = [
        ('target_aaaa_mol_aaaa', 'test_data/ligand.sdf', 'test_data/pocket_6A.pdb'),
        ('target_3tf7_mol_3tf7', 'test_data/ligand.sdf', 'test_data/pocket_6A.pdb')
    ]
    
    for target_id, ligand_path, protein_path in test_files:
        # Process ligand
        ligand_graph = process_ligand_sdf(ligand_path)
        ligand_graph.id = target_id
        ligand_output = f'test_data/debug_graphs/ligand/{target_id}.pt'
        torch.save(ligand_graph, ligand_output)
        
        # Process protein
        protein_graph = process_protein_pdb_ligand_style(protein_path)
        protein_graph.id = target_id
        protein_output = f'test_data/debug_graphs/protein/{target_id}.pt'
        torch.save(protein_graph, protein_output)
        
        print(f"   Processing {target_id}...")
        print(f"     ✅ {ligand_graph.x.shape[0]} ligand + {protein_graph.x.shape[0]} protein = {ligand_graph.x.shape[0] + protein_graph.x.shape[0]} total atoms")

@hydra.main(version_base=None, config_path="conf/conf_pdb", config_name="config")
def main(cfg: DictConfig) -> None:
    print("============================================================")
    print("🔬 PDB ETNN MODEL DEBUG")
    print("============================================================")
    print(f"Dataset config loaded")
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create test data
    create_test_index()
    create_test_graphs()
    
    # Load dataset configuration  
    cfg_dataset = cfg.dataset
    
    print(f"\n🔧 Dataset Configuration:")
    print(f"   connectivity: {cfg_dataset.connectivity}")
    print(f"   neighbor_types: {cfg_dataset.neighbor_types}")
    print(f"   lifters: {cfg_dataset.lifters}")
    print(f"   r_cut: {getattr(cfg_dataset, 'r_cut', 'N/A')}")
    print(f"   connect_cross: {getattr(cfg_dataset, 'connect_cross', 'N/A')}")
    
    # Create dataset
    print(f"\n📁 Loading dataset...")
    # Get r_cut value from config
    r_cut_value = getattr(cfg_dataset, 'r_cut', 5.0)
    print(f"   Using r_cut for lifters: {r_cut_value}")
    
    train_dataset = PDBBindCC(
        root="test_data/debug_pdbbind",
        index="test_data/index_debug.csv",
        lifters=cfg_dataset.lifters,
        neighbor_types=cfg_dataset.neighbor_types,
        connectivity=cfg_dataset.connectivity,
        connect_cross=getattr(cfg_dataset, 'connect_cross', False),
        r_cut=r_cut_value,
        merge_graphs=True,
        single_graphs_path="test_data/debug_graphs",
        force_reload=True
        # NOTE: r_cut is now automatically passed to lifters via lifter_kwargs
    )
    
    print(f"✅ Dataset created with {len(train_dataset)} samples")
    
    # Analyze first sample
    print(f"\n📊 Sample Analysis:")
    sample = train_dataset[0]
    print(f"   Sample ID: {getattr(sample, 'id', 'N/A')}")
    print(f"   Target value: {sample.y.item():.3f}")
    
    # Data structure verification
    print(f"\n🔍 Data Structure Verification:")
    print(f"   Sample type: {type(sample).__name__}")
    print(f"   Has pos: {hasattr(sample, 'pos')} -> {sample.pos.shape if hasattr(sample, 'pos') else 'N/A'}")
    print(f"   Has origin_nodes: {hasattr(sample, 'origin_nodes')}")
    if hasattr(sample, 'origin_nodes'):
        origin = sample.origin_nodes
        ligand_count = (origin == 0).sum().item()
        protein_count = (origin == 1).sum().item()
        print(f"     Ligand atoms (0): {ligand_count}")
        print(f"     Protein atoms (1): {protein_count}")
        print(f"     Total atoms: {ligand_count + protein_count}")
    
    # Feature dimensions
    print(f"\n🎯 Feature Dimensions:")
    for i in range(3):  # Check ranks 0, 1, 2
        if hasattr(sample, f'x_{i}'):
            x_tensor = getattr(sample, f'x_{i}')
            print(f"   Rank {i}: {x_tensor.shape} -> {x_tensor.shape[1]} features per cell")
            
            # Show sample features for verification
            if i == 0 and x_tensor.shape[0] > 0:  # Atom features
                print(f"     Sample atom features (first 3 atoms):")
                for j in range(min(3, x_tensor.shape[0])):
                    feat = x_tensor[j]
                    print(f"       Atom {j}: [{feat[0]:.1f}, {feat[1]:.1f}, {feat[2]:.1f}, {feat[3]:.3f}, ...] (element, valence, charge, mass, ...)")
            
            elif i == 1 and x_tensor.shape[0] > 0:  # Bond features
                print(f"     Sample bond features (first 3 bonds):")
                for j in range(min(3, x_tensor.shape[0])):
                    feat = x_tensor[j]
                    # Show first 5 and last 5 features
                    first_5 = feat[:5].tolist()
                    last_5 = feat[-5:].tolist()
                    print(f"       Bond {j}: [{', '.join(f'{x:.2f}' for x in first_5)}, ..., {', '.join(f'{x:.2f}' for x in last_5)}]")
            
            elif i == 2 and x_tensor.shape[0] > 0:  # Cross-interaction features
                print(f"     Sample cross-interaction features (first 3):")
                for j in range(min(3, x_tensor.shape[0])):
                    feat = x_tensor[j]
                    # Show first 5 and last 5 features (RBF + interaction types)
                    first_5 = feat[:5].tolist()
                    last_5 = feat[-5:].tolist()
                    print(f"       Cross {j}: [{', '.join(f'{x:.3f}' for x in first_5)}, ..., {', '.join(f'{x:.1f}' for x in last_5)}] (RBF, ..., interaction_flags)")
    
    # Cell counts
    print(f"\n🔗 Cell Counts:")
    for i in range(3):  # Check ranks 0, 1, 2
        if hasattr(sample, f'x_{i}'):
            x_tensor = getattr(sample, f'x_{i}')
            print(f"   Rank {i}: {x_tensor.shape[0]} cells")
    
    # Cell list verification
    print(f"\n🔬 Cell List Verification:")
    for i in range(3):
        if hasattr(sample, f'x_{i}'):
            try:
                cell_list = sample.cell_list(i)
                print(f"   Rank {i} cells: {len(cell_list)} cells")
                if len(cell_list) > 0:
                    # Show first few cell compositions
                    for j in range(min(3, len(cell_list))):
                        cell = cell_list[j]
                        if i == 0:  # Atoms
                            print(f"     Cell {j}: atom {cell.tolist()}")
                        elif i == 1:  # Bonds  
                            print(f"     Cell {j}: bond between atoms {cell.tolist()}")
                        elif i == 2:  # Cross-interactions
                            print(f"     Cell {j}: cross-interaction between atoms {cell.tolist()}")
            except Exception as e:
                print(f"   Rank {i}: Error getting cell list - {e}")
    
    # Adjacency matrices - the key focus
    print(f"\n🔄 Adjacency Matrices:")
    print(f"   Configuration: connectivity={cfg_dataset.connectivity}, neighbor_types={cfg_dataset.neighbor_types}")
    
    # Check all possible adjacency types
    adjacency_found = []
    total_connections = 0
    
    for i in range(3):  # ranks 0-2
        for j in range(3):  # ranks 0-2
            # Check basic adjacency i_j
            adj_attr = f"adj_{i}_{j}"
            if hasattr(sample, adj_attr):
                adj_matrix = getattr(sample, adj_attr)
                if adj_matrix is not None and hasattr(adj_matrix, 'shape'):
                    connections = adj_matrix.shape[1] if len(adj_matrix.shape) == 2 else 0
                    total_connections += connections
                    adjacency_found.append((adj_attr, adj_matrix.shape, connections))
                    print(f"   ✅ {adj_attr}: {adj_matrix.shape} ({connections:,} connections)")
            
            # Check neighbor-type adjacency i_j_k
            for k in range(3):
                adj_attr = f"adj_{i}_{j}_{k}"
                if hasattr(sample, adj_attr):
                    adj_matrix = getattr(sample, adj_attr)
                    if adj_matrix is not None and hasattr(adj_matrix, 'shape'):
                        connections = adj_matrix.shape[1] if len(adj_matrix.shape) == 2 else 0
                        total_connections += connections
                        adjacency_found.append((adj_attr, adj_matrix.shape, connections))
                        print(f"   ✅ {adj_attr}: {adj_matrix.shape} ({connections:,} connections)")
    
    if not adjacency_found:
        print("   ❌ No adjacency matrices found")
    else:
        print(f"   📊 Total connections: {total_connections:,}")
    
    # Detailed rank 2 adjacency analysis
    if hasattr(sample, 'x_2'):
        print(f"\n🔗 Rank 2 Adjacency Details:")
        print(f"   Rank 2 cells: {sample.x_2.shape[0]} cells with {sample.x_2.shape[1]} features each")
        
        # Show rank 2 to rank 0 connections (atoms)
        if hasattr(sample, 'adj_2_0'):
            adj_2_0 = getattr(sample, 'adj_2_0')
            print(f"   Rank 2 → Rank 0 (atoms): {adj_2_0.shape[1]} connections")
            print(f"     Each rank 2 cell connects to multiple atoms")
            print(f"     Connection matrix shape: {adj_2_0.shape}")
            
            # Show first few connections
            if adj_2_0.shape[1] > 0:
                print(f"     First 5 connections: {adj_2_0[:, :5].tolist()}")
        
        # Show rank 2 to rank 1 connections (bonds)
        if hasattr(sample, 'adj_2_1'):
            adj_2_1 = getattr(sample, 'adj_2_1')
            print(f"   Rank 2 → Rank 1 (bonds): {adj_2_1.shape[1]} connections")
            print(f"     Each rank 2 cell connects to multiple bonds")
            print(f"     Connection matrix shape: {adj_2_1.shape}")
            
            # Show first few connections
            if adj_2_1.shape[1] > 0:
                print(f"     First 5 connections: {adj_2_1[:, :5].tolist()}")
        
        # Show rank 0 to rank 2 connections (atoms to higher-order)
        if hasattr(sample, 'adj_0_2'):
            adj_0_2 = getattr(sample, 'adj_0_2')
            print(f"   Rank 0 → Rank 2 (atoms to higher-order): {adj_0_2.shape[1]} connections")
            print(f"     Each atom can participate in multiple higher-order structures")
            print(f"     Connection matrix shape: {adj_0_2.shape}")
            
            # Show first few connections
            if adj_0_2.shape[1] > 0:
                print(f"     First 5 connections: {adj_0_2[:, :5].tolist()}")
        
        # Show rank 1 to rank 2 connections (bonds to higher-order)
        if hasattr(sample, 'adj_1_2'):
            adj_1_2 = getattr(sample, 'adj_1_2')
            print(f"   Rank 1 → Rank 2 (bonds to higher-order): {adj_1_2.shape[1]} connections")
            print(f"     Each bond can participate in multiple higher-order structures")
            print(f"     Connection matrix shape: {adj_1_2.shape}")
            
            # Show first few connections
            if adj_1_2.shape[1] > 0:
                print(f"     First 5 connections: {adj_1_2[:, :5].tolist()}")
        
        # Show the actual rank 2 cell contents
        print(f"\n🧬 Rank 2 Cell Contents:")
        cell_list_2 = sample.cell_list(2)
        for i, cell in enumerate(cell_list_2):
            print(f"   Cell {i}: {len(cell)} atoms - indices {cell.tolist()}")
            print(f"     Features: {sample.x_2[i].tolist()}")
            
            # Show which atoms this cell contains
            atom_symbols = []
            for atom_idx in cell:
                if atom_idx < len(sample.x):
                    # Extract element from atom features (assuming first feature is element)
                    element_val = sample.x[atom_idx, 0].item()
                    if element_val == 6: atom_symbols.append("C")
                    elif element_val == 7: atom_symbols.append("N")
                    elif element_val == 8: atom_symbols.append("O")
                    elif element_val == 1: atom_symbols.append("H")
                    else: atom_symbols.append(f"E{int(element_val)}")
                else:
                    atom_symbols.append("?")
            print(f"     Atoms: {atom_symbols}")
            print()
    
    # Debug cell format for invariants computation
    
    # Model creation
    print(f"\n🤖 Model Creation:")
    model = utils.get_model(cfg, train_dataset)
    param_count = sum(p.numel() for p in model.parameters())
    model = model.to(device)
    print(f"   ✅ Model created successfully")
    print(f"   Parameters: {param_count:,}")
    
    # Forward pass test
    print(f"\n🚀 Forward Pass Test:")
    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        batch = next(iter(test_loader)).to(device)
        
        pred = model(batch)
        
        print(f"   ✅ Forward pass successful!")
        print(f"   Batch shape: {batch.y.shape}")
        print(f"   Prediction shape: {pred.shape}")
        print(f"   Raw prediction: {pred[0].item():.3f}")
        print(f"   Target: {batch.y[0].item():.3f}")
        print(f"   Absolute error: {abs(pred[0].item() - batch.y[0].item()):.3f}")
    
    print(f"\n🎉 DEBUG COMPLETED SUCCESSFULLY!")
    print(f"📝 Summary:")
    print(f"   • Configuration: {cfg_dataset.connectivity} connectivity, {cfg_dataset.neighbor_types} neighbors")
    print(f"   • Adjacency matrices: {len(adjacency_found)} found, {total_connections:,} total connections")
    print(f"   • Model parameters: {param_count:,}")
    print(f"   • Forward pass: ✅ Working")

if __name__ == "__main__":
    main()