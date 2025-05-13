"""
Test script for the AI Forgery Detective model.
This script tests the model on sample images to verify it's working correctly.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model(image_path=None, model_path=None):
    """
    Test the model on a sample image
    
    Args:
        image_path: Path to the image to test (or None to use default)
        model_path: Path to the model to use (or None to use default)
    """
    # Use default paths if none provided
    if image_path is None:
        # Try to use one of the sample images if available
        if os.path.exists('fake.png'):
            image_path = 'fake.png'
        elif os.path.exists('real.png'):
            image_path = 'real.png'
        else:
            print("No sample image found. Please provide an image path.")
            return False
    
    if model_path is None:
        model_path = os.path.join('outputs', 'checkpoints', 'most_accuracy_model.pt')
        # Check if model exists, if not try the Colab model
        if not os.path.exists(model_path):
            model_path = 'final_casia_finetuned_model.pth'
            if not os.path.exists(model_path):
                print("No model found. Please provide a model path.")
                return False
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return False
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return False
    
    print(f"Testing model: {model_path}")
    print(f"Testing image: {image_path}")
    
    # Try to import the inference module
    try:
        try:
            # First try the new module
            from experiments.infer_new import detect_forgery
            print("Using ViT+Siamese model for inference")
        except ImportError:
            # Fall back to old module
            from experiments.infer import detect_forgery
            print("Using legacy model for inference")
        
        # Create output directory
        output_dir = os.path.join('outputs', 'test_results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Run detection
        result = detect_forgery(image_path, model_path, output_dir)
        
        # Display results
        print("\nResults:")
        print(f"Forgery probability: {result['forgery_probability']:.4f}")
        print(f"Forgery detected: {result['forgery_detected']}")
        print(f"Number of suspicious regions: {len(result['suspicious_regions'])}")
        
        if 'heatmap_path' in result and result['heatmap_path']:
            print(f"Heatmap saved to: {result['heatmap_path']}")
            
            # Display the heatmap
            try:
                img = Image.open(result['heatmap_path'])
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Forgery Probability: {result['forgery_probability']:.2%}")
                plt.show()
            except Exception as e:
                print(f"Error displaying heatmap: {e}")
        
        return True
    
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the AI Forgery Detective model")
    parser.add_argument('--image', type=str, help="Path to the image to test")
    parser.add_argument('--model', type=str, help="Path to the model to use")
    args = parser.parse_args()
    
    success = test_model(args.image, args.model)
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed. Please check the error messages above.")
