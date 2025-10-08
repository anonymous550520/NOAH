import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_results(results, metadata, save_heatmaps=False, output_dir=None):
    metadata_dict = {item['id']: item for item in metadata}

    total_result = {
        'CHR': 0,
        'EHR': 0,
        'COR': 0,
        'EOR': 0,
        'IEOR': 0,
    }

    # Initialize heatmap data structures
    heatmap_ehr_results = {
        "low-start": 0, "low-center": 0, "low-end": 0,
        "mid-start": 0, "mid-center": 0, "mid-end": 0,
        "high-start": 0, "high-center": 0, "high-end": 0,
    }
    
    heatmap_eor_results = {
        "low-start": 0, "low-center": 0, "low-end": 0,
        "mid-start": 0, "mid-center": 0, "mid-end": 0,
        "high-start": 0, "high-center": 0, "high-end": 0,
    }
    
    heatmap_ieor_results = {
        "low-start": 0, "low-center": 0, "low-end": 0,
        "mid-start": 0, "mid-center": 0, "mid-end": 0,
        "high-start": 0, "high-center": 0, "high-end": 0,
    }
    
    segment_counts = {
        "low-start": 0, "low-center": 0, "low-end": 0,
        "mid-start": 0, "mid-center": 0, "mid-end": 0,
        "high-start": 0, "high-center": 0, "high-end": 0,
    }

    for result in results:
        metadata_item = metadata_dict[result['id']]
        pos = metadata_item['position']
        sim = metadata_item['similarity']
        segment = f"{sim}-{pos}"
        segment_counts[segment] += 1

        # HR
        if result['hallucination']['extracted_events_count'] > 0:
            ehr = result['hallucination']['hallucination_count'] / result['hallucination']['extracted_events_count']
            total_result['EHR'] += ehr
            c_hr = 1 if result['hallucination']['hallucination_count'] > 0 else 0
            total_result['CHR'] += c_hr
            
            # Add to heatmap data
            heatmap_ehr_results[segment] += ehr
        else:
            ehr = 0
            c_hr = 0

        if result['omission']['inserted_omission_count'] > 0:
            eor = (result['omission']['total_omission_count'] -1) / (result['omission']['ground_truth_events_count'] - 1)
            total_result['EOR'] += eor
            cor = 1 if result['omission']['total_omission_count'] > 0 else 0
            total_result['COR'] += cor
            total_result['IEOR'] += 1
            
            # Add to heatmap data
            heatmap_eor_results[segment] += eor
            heatmap_ieor_results[segment] += 1
        else:   
            eor = (result['omission']['total_omission_count']) / (result['omission']['ground_truth_events_count'] - 1)
            total_result['EOR'] += eor
            cor = 1 if result['omission']['total_omission_count'] > 0 else 0
            total_result['COR'] += cor
            
            # Add to heatmap data
            heatmap_eor_results[segment] += eor

    # Calculate averages
    for key, value in total_result.items():
        total_result[key] = value / len(results)
    
    # Calculate heatmap averages
    for segment in heatmap_ehr_results:
        if 1000 > 0:
            heatmap_ehr_results[segment] /= 1000
            heatmap_eor_results[segment] /= 1000
            heatmap_ieor_results[segment] /= 1000
        
    print(total_result)
    
    # Generate heatmaps if requested
    if save_heatmaps and output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save EHR heatmap
        save_heatmap(heatmap_ehr_results, 
                     'ehr', 
                     'Expected Hallucination Rate', 
                     'Reds',
                     os.path.join(output_dir, 'heatmap_ehr.png'))
        
        # Save EOR heatmap
        save_heatmap(heatmap_eor_results, 
                     'eor', 
                     'Expected Omission Rate', 
                     'Blues',
                     os.path.join(output_dir, 'heatmap_eor.png'))
        
        # Save IEOR heatmap
        save_heatmap(heatmap_ieor_results, 
                     'ieor', 
                     'Insertion Expected Omission Rate', 
                     'Greens',
                     os.path.join(output_dir, 'heatmap_ieor.png'))


def create_heatmap_data(data_dict):
    """Convert dictionary data to 3x3 matrix for heatmap"""
    sim_levels = ['low', 'mid', 'high']
    positions = ['start', 'center', 'end']
    
    matrix = np.zeros((3, 3))
    for i, sim in enumerate(sim_levels):
        for j, pos in enumerate(positions):
            key = f"{sim}-{pos}"
            matrix[i, j] = data_dict.get(key, 0)
    
    return matrix


def save_heatmap(data_dict, metric_name, title, cmap, save_path):
    """Save individual heatmap with consistent styling"""
    data = create_heatmap_data(data_dict)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Labels
    sim_labels = ['Low', 'Medium', 'High']
    pos_labels = ['Start', 'Middle', 'End']
    
    # Create heatmap
    sns.heatmap(data, 
               annot=True, 
               fmt='.3f',
               cmap=cmap,
               xticklabels=pos_labels,
               yticklabels=sim_labels,
               ax=ax,
               cbar=False,
               square=True,
               annot_kws={'size': 24})
    
    # Increase tick labels font size
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    # Save heatmap
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {metric_name} heatmap to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation results')
    parser.add_argument('--result-path', '-r', required=True, help='Path to evaluation results JSON file')
    parser.add_argument('--metadata-path', '-m', default='path/to/metadata.json', help='Path to metadata JSON file')
    parser.add_argument('--save-heatmaps', '-s', action='store_true', help='Generate and save heatmap visualizations')
    parser.add_argument('--output-dir', '-o', help='Output directory for heatmaps (required if --save-heatmaps is used)')
    
    args = parser.parse_args()
    
    print(f"Input file: {args.result_path}")
    
    if args.save_heatmaps and not args.output_dir:
        print("Error: --output-dir is required when using --save-heatmaps")
        return

    results = load_json(args.result_path)
    metadata = load_json(args.metadata_path)

    analyze_results(results, metadata, save_heatmaps=args.save_heatmaps, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
