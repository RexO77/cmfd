"""
Model conversion utility to rename and place the Google Colab model 
in the correct location for the application
"""

import os
import sys
import shutil
import torch

def setup_model(source_model_path="final_casia_finetuned_model.pth", 
                target_path="outputs/checkpoints/most_accuracy_model.pt"):
    """
    Copy and convert the model from Google Colab to the format expected by the application
    
    Args:
        source_model_path: Path to the source model from Google Colab
        target_path: Path where the model should be saved for the application
    """
    # Check if source model exists
    if not os.path.exists(source_model_path):
        print(f"Error: Source model {source_model_path} not found!")
        print("Please ensure you have downloaded the model from Google Colab.")
        return False
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Check if we need to convert the model or just copy it
    try:
        # Try to load the model to verify it
        model_state = torch.load(source_model_path, map_location="cpu")
        
        # Save model to target path
        torch.save(model_state, target_path)
        print(f"Model successfully converted and saved to {target_path}")
        
        return True
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        
        # Try a simple file copy as fallback
        try:
            shutil.copy(source_model_path, target_path)
            print(f"Model copied to {target_path} (without conversion)")
            return True
        except Exception as copy_error:
            print(f"Error copying model: {str(copy_error)}")
            return False

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        source_path = sys.argv[1]
        print(f"Using provided source model path: {source_path}")
    else:
        source_path = "final_casia_finetuned_model.pth"
        print(f"Using default source model path: {source_path}")
    
    # Setup the model
    success = setup_model(source_path)
    
    if success:
        print("Model setup completed successfully!")
        print("You can now run the application with: python run_app.py")
    else:
        print("Model setup failed. Please check the error messages above.")
