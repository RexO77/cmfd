#!/usr/bin/env python3
"""
Test script for AI Forgery Detective to verify the application is working properly.
This performs a basic check with sample images to ensure the model works correctly.
"""

import os
import sys
import tempfile
import shutil
import cv2
import numpy as np
import time

def print_status(message):
    """Print status message with timestamp"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def copy_region(image, src_x, src_y, dst_x, dst_y, size):
    """Copy a region from source to destination coordinates"""
    region = image[src_y:src_y+size, src_x:src_x+size].copy()
    image[dst_y:dst_y+size, dst_x:dst_x+size] = region
    return image

def create_test_image(width=800, height=600, forgery=True):
    """Create a test image with optional forgery"""
    # Create random noise image
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some structure (gradients, patterns)
    for i in range(height):
        for j in range(width):
            # Gradient
            image[i, j, 0] = (i * 255) // height
            image[i, j, 1] = (j * 255) // width
            image[i, j, 2] = ((i+j) * 255) // (height+width)
    
    # Add some shapes
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(image, (400, 300), 50, (0, 255, 0), -1)
    cv2.line(image, (0, 0), (width, height), (0, 0, 255), 5)
    
    # If forgery is requested, copy a region
    if forgery:
        src_x, src_y = 100, 100
        dst_x, dst_y = 500, 400
        size = 100
        image = copy_region(image, src_x, src_y, dst_x, dst_y, size)
    
    return image

def test_forgery_detection():
    """Test the forgery detection functionality"""
    print_status("Starting forgery detection test")
    
    # Create test directory
    test_dir = tempfile.mkdtemp()
    print_status(f"Created test directory: {test_dir}")
    
    try:
        # Create sample images
        forged_path = os.path.join(test_dir, "forged.jpg")
        original_path = os.path.join(test_dir, "original.jpg")
        
        # Generate and save images
        forged_img = create_test_image(forgery=True)
        original_img = create_test_image(forgery=False)
        
        cv2.imwrite(forged_path, forged_img)
        cv2.imwrite(original_path, original_img)
        print_status(f"Created test images: {forged_path}, {original_path}")
        
        # Import the detector function
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from experiments.infer import detect_forgery
            print_status("Successfully imported detection module")
        except ImportError as e:
            print_status(f"Error importing detection module: {e}")
            return False
        
        # Set up output directory
        output_dir = os.path.join(test_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to find model
        model_paths = [
            "outputs/checkpoints/most_accuracy_model.pt",
            "outputs/checkpoints/best_model.pt",
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                print_status(f"Using model: {model_path}")
                break
        
        if model_path is None:
            print_status("Error: No model found. Please train or download a model first.")
            return False
        
        # Set environment variable for CPU
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Test forged image
        print_status("Testing forged image detection...")
        start_time = time.time()
        forged_result = detect_forgery(
            image_path=forged_path,
            model_path=model_path,
            patch_size=64,
            stride=32,
            threshold=0.6,
            output_dir=output_dir
        )
        forged_time = time.time() - start_time
        print_status(f"Forged image test completed in {forged_time:.2f} seconds")
        print_status(f"Forgery detected: {forged_result['forgery_detected']}")
        print_status(f"Forgery probability: {forged_result['forgery_probability']:.2%}")
        print_status(f"Suspicious pairs: {len(forged_result['suspicious_pairs'])}")
        
        # Test original image
        print_status("Testing original image detection...")
        start_time = time.time()
        original_result = detect_forgery(
            image_path=original_path,
            model_path=model_path,
            patch_size=64,
            stride=32,
            threshold=0.6,
            output_dir=output_dir
        )
        original_time = time.time() - start_time
        print_status(f"Original image test completed in {original_time:.2f} seconds")
        print_status(f"Forgery detected: {original_result['forgery_detected']}")
        print_status(f"Forgery probability: {original_result['forgery_probability']:.2%}")
        print_status(f"Suspicious pairs: {len(original_result['suspicious_pairs'])}")
        
        # Check if the results make sense
        success = True
        warnings = []
        
        if not forged_result['forgery_detected']:
            warnings.append("WARNING: Failed to detect forgery in forged image")
            success = False
        
        if original_result['forgery_detected']:
            warnings.append("WARNING: False positive in original image")
            success = False
        
        if forged_result['forgery_probability'] < 0.3:
            warnings.append(f"WARNING: Low forgery probability ({forged_result['forgery_probability']:.2%}) in forged image")
        
        if len(forged_result['suspicious_pairs']) < 1:
            warnings.append("WARNING: No suspicious pairs found in forged image")
            success = False
        
        # Print warnings
        for warning in warnings:
            print_status(warning)
        
        if success:
            print_status("SUCCESS: Forgery detection test passed!")
        else:
            print_status("ERROR: Forgery detection test failed")
        
        return success
    
    except Exception as e:
        print_status(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        try:
            shutil.rmtree(test_dir)
            print_status(f"Cleaned up test directory: {test_dir}")
        except:
            print_status(f"Failed to clean up test directory: {test_dir}")

if __name__ == "__main__":
    print_status("=== AI Forgery Detective Test ===")
    success = test_forgery_detection()
    
    if success:
        print_status("All tests passed! The system is working correctly.")
        sys.exit(0)
    else:
        print_status("Tests failed. Please check the errors above.")
        sys.exit(1) 