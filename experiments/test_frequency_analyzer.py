import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from experiments.infer import detect_forgery
from utils.frequency_analysis import FrequencyAnalyzer

# Test configuration
MODEL_PATH = "outputs/checkpoints/best_accuracy_model.pt"  # Use the best accuracy model
OUTPUT_DIR = "outputs/test_frequency_results"
TEST_DATA_DIR = "data/CoMoFoD_small_v2/test"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_pair(original_path, forged_path):
    """Test detection on original vs forged pair and save results"""
    
    print(f"\nTesting on: {os.path.basename(original_path)} vs {os.path.basename(forged_path)}")
    
    # Test on original image (should detect no forgery)
    print(f"Processing original image: {os.path.basename(original_path)}...")
    start_time = time.time()
    original_result = detect_forgery(
        original_path, 
        model_path=MODEL_PATH,
        patch_size=64, 
        stride=32, 
        threshold=0.55,
        output_dir=OUTPUT_DIR
    )
    original_time = time.time() - start_time
    
    # Test on forged image (should detect forgery)
    print(f"Processing forged image: {os.path.basename(forged_path)}...")
    start_time = time.time()
    forged_result = detect_forgery(
        forged_path, 
        model_path=MODEL_PATH,
        patch_size=64, 
        stride=32, 
        threshold=0.55,
        output_dir=OUTPUT_DIR
    )
    forged_time = time.time() - start_time
    
    return {
        'original': {
            'path': original_path,
            'result': original_result,
            'time': original_time
        },
        'forged': {
            'path': forged_path,
            'result': forged_result,
            'time': forged_time
        }
    }

def run_test(test_folders=None, max_tests=5):
    """Run tests on specified folders or the first N test folders"""
    
    # Find all test folders
    all_test_folders = [f for f in os.listdir(TEST_DATA_DIR) 
                       if os.path.isdir(os.path.join(TEST_DATA_DIR, f))]
    all_test_folders.sort()
    
    # Use specified folders or first N folders
    if test_folders is None:
        test_folders = all_test_folders[:max_tests]
    
    results = []
    
    for folder in tqdm(test_folders, desc="Processing test folders"):
        folder_path = os.path.join(TEST_DATA_DIR, folder)
        
        # Find original and forged images
        files = os.listdir(folder_path)
        original_file = next((f for f in files if f.endswith('_O.png')), None)
        forged_file = next((f for f in files if f.endswith('_F.png')), None)
        
        if not original_file or not forged_file:
            print(f"Warning: Missing original or forged image in {folder}")
            continue
            
        original_path = os.path.join(folder_path, original_file)
        forged_path = os.path.join(folder_path, forged_file)
        
        # Run test on this pair
        pair_result = test_pair(original_path, forged_path)
        results.append(pair_result)
    
    return results

def analyze_results(results):
    """Analyze and report test results"""
    
    # Initialize counters and metrics
    true_positives = 0  # Correctly detected forgery in forged image
    false_positives = 0  # Incorrectly detected forgery in original image
    true_negatives = 0  # Correctly identified no forgery in original image
    false_negatives = 0  # Failed to detect forgery in forged image
    
    forged_probabilities = []
    original_probabilities = []
    
    forged_times = []
    original_times = []
    
    # Process results
    for pair in results:
        # Original image results (expect no forgery)
        if pair['original']['result']['forgery_detected']:
            false_positives += 1
        else:
            true_negatives += 1
        
        # Forged image results (expect forgery)
        if pair['forged']['result']['forgery_detected']:
            true_positives += 1
        else:
            false_negatives += 1
        
        # Record probabilities and times
        original_probabilities.append(pair['original']['result']['forgery_probability'])
        forged_probabilities.append(pair['forged']['result']['forgery_probability'])
        
        original_times.append(pair['original']['time'])
        forged_times.append(pair['forged']['time'])
    
    # Calculate metrics
    total_samples = len(results) * 2  # 2 images per test (original + forged)
    num_forged = len(results)  # Number of forged images
    num_original = len(results)  # Number of original images
    
    # Accuracy metrics
    accuracy = (true_positives + true_negatives) / total_samples
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print report
    print("\n---------- TEST RESULTS SUMMARY ----------")
    print(f"Total test pairs: {len(results)}")
    print(f"True Positives: {true_positives}/{num_forged} ({true_positives/num_forged:.2%})")
    print(f"True Negatives: {true_negatives}/{num_original} ({true_negatives/num_original:.2%})")
    print(f"False Positives: {false_positives}/{num_original} ({false_positives/num_original:.2%})")
    print(f"False Negatives: {false_negatives}/{num_forged} ({false_negatives/num_forged:.2%})")
    print("\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print("\nAverage Probabilities:")
    print(f"Original Images: {np.mean(original_probabilities):.4f}")
    print(f"Forged Images: {np.mean(forged_probabilities):.4f}")
    print(f"Probability Separation: {np.mean(forged_probabilities) - np.mean(original_probabilities):.4f}")
    print("\nProcessing Time:")
    print(f"Average (Original): {np.mean(original_times):.2f} seconds")
    print(f"Average (Forged): {np.mean(forged_times):.2f} seconds")
    print("----------------------------------------")
    
    # Plot probability distributions
    plt.figure(figsize=(10, 6))
    plt.hist(original_probabilities, alpha=0.5, label='Original Images', bins=10, range=(0, 1))
    plt.hist(forged_probabilities, alpha=0.5, label='Forged Images', bins=10, range=(0, 1))
    plt.axvline(x=0.3, color='r', linestyle='--', label='Decision Threshold (0.3)')
    plt.xlabel('Forgery Probability')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Forgery Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIR, 'probability_distribution.png'))
    print(f"Saved probability distribution plot to {os.path.join(OUTPUT_DIR, 'probability_distribution.png')}")
    
    # Return metrics for further analysis if needed
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall, 
        'f1_score': f1_score,
        'avg_original_prob': np.mean(original_probabilities),
        'avg_forged_prob': np.mean(forged_probabilities),
        'original_probabilities': original_probabilities,
        'forged_probabilities': forged_probabilities
    }

# Example test folders - choose specific test folders to analyze
TEST_FOLDERS = [
    '004',  # Simple copy-move
    '012',  # Multiple objects copied
    '020',  # Rotated copy-move
    '040',  # Scaled copy-move
    '043',  # Complex scene
    '049',  # Textured background
    '077',  # Multiple objects
    '082',  # Small copied region
    '100',  # High detail scene
]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Copy-Move Forgery Detection with Frequency Analysis")
    parser.add_argument('--folders', nargs='+', help="Specific test folders to use")
    parser.add_argument('--all', action='store_true', help="Test on all available test folders")
    parser.add_argument('--max', type=int, default=5, help="Maximum number of test folders if not specified")
    parser.add_argument('--model', type=str, default=MODEL_PATH, help="Path to the model checkpoint")
    
    args = parser.parse_args()
    
    if args.model:
        MODEL_PATH = args.model
        
    # Determine which folders to test
    if args.all:
        test_folders = None  # Will use all available
        max_tests = 1000  # Large number to effectively use all
    elif args.folders:
        test_folders = args.folders
        max_tests = len(test_folders)
    else:
        # Use default selection of interesting test cases
        test_folders = TEST_FOLDERS
        max_tests = len(test_folders)
    
    print(f"Testing with model: {MODEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    if test_folders:
        print(f"Testing on specific folders: {test_folders}")
    else:
        print(f"Testing on up to {max_tests} folders")
        
    # Run tests
    results = run_test(test_folders, max_tests=max_tests)
    
    # Analyze and report results
    metrics = analyze_results(results)
    
    # Generate detailed frequency analysis report for each test case
    print("\nGenerating detailed frequency analysis for test cases...")
    
    for pair in results:
        original_path = pair['original']['path']
        forged_path = pair['forged']['path']
        
        folder_name = Path(original_path).parent.name
        output_folder = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Load images
        original_img = cv2.imread(original_path)
        forged_img = cv2.imread(forged_path)
        
        # If forged result has suspicious pairs, analyze them with frequency analyzer
        if pair['forged']['result'].get('suspicious_pairs', []):
            analyzer = FrequencyAnalyzer()
            
            # Run frequency band analysis
            bands_result = analyzer.analyze_frequency_bands(
                forged_img, 
                pair['forged']['result']['suspicious_pairs'], 
                patch_size=64, 
                num_bands=4
            )
            
            # Run block artifact analysis
            blocks_result = analyzer.analyze_block_artifacts(forged_img, patch_size=64)
            
            # Save visualizations
            if bands_result.get('band_visualization') is not None:
                cv2.imwrite(os.path.join(output_folder, f"{folder_name}_bands.jpg"), 
                           bands_result['band_visualization'])
                
            if blocks_result.get('visualization') is not None:
                cv2.imwrite(os.path.join(output_folder, f"{folder_name}_blocks.jpg"), 
                           blocks_result['visualization'])
                
            # Save metrics to text file
            with open(os.path.join(output_folder, f"{folder_name}_metrics.txt"), 'w') as f:
                f.write(f"Folder: {folder_name}\n")
                f.write(f"Original file: {os.path.basename(original_path)}\n")
                f.write(f"Forged file: {os.path.basename(forged_path)}\n\n")
                
                f.write("Detection Results:\n")
                f.write(f"Original - Probability: {pair['original']['result']['forgery_probability']:.4f}, "
                       f"Detected: {pair['original']['result']['forgery_detected']}\n")
                f.write(f"Forged - Probability: {pair['forged']['result']['forgery_probability']:.4f}, "
                       f"Detected: {pair['forged']['result']['forgery_detected']}\n\n")
                
                f.write("Frequency Analysis Metrics:\n")
                f.write(f"Block Strength: {blocks_result.get('block_strength', 0):.4f}\n")
                f.write(f"Block Periodicity: {blocks_result.get('block_periodicity', 0):.4f}\n")
                f.write(f"Est. Compression Level: {blocks_result.get('compression_level', 0):.4f}\n\n")
                
                # Write band similarities if available
                if 'band_similarities' in bands_result and bands_result['band_similarities']:
                    f.write("Frequency Band Analysis:\n")
                    for i, band_sim in enumerate(bands_result['band_similarities']):
                        f.write(f"Pair {i+1}: {band_sim}\n")
    
    print(f"Testing complete. Results saved to {OUTPUT_DIR}")