# Using the Google Colab Model with AI Forgery Detective

This guide explains how to set up and use the Google Colab-trained model with the AI Forgery Detective application.

## Step 1: Download the Model

Ensure you have downloaded the `final_casia_finetuned_model.pth` file from Google Colab and placed it in the root directory of the project.

## Step 2: Set Up the Model

Run the model setup script to prepare the model for use with the application:

```bash
python setup_model.py
```

This script will:
1. Check if the model file exists
2. Convert or copy it to the correct location
3. Verify it can be loaded correctly

## Step 3: Test the Model

To verify that the model is working correctly, run the test script:

```bash
python test_model.py
```

This will test the model on a sample image and display the results. If you want to test on a specific image, run:

```bash
python test_model.py --image /path/to/your/image.jpg
```

## Step 4: Run the Application

Once the model is set up and tested, you can run the full application:

```bash
python run_app.py
```

This will start the Streamlit web interface where you can:
- Upload images for forgery detection
- View detailed results with visualizations
- Compare two images for similarity

## Troubleshooting

If you encounter any issues:

### Model Not Found

If you see an error about the model not being found:
1. Make sure the `final_casia_finetuned_model.pth` file is in the project root directory
2. Run `python setup_model.py` to set up the model

### Import Errors

If you see import errors related to `timm` or other libraries:
1. Make sure you've installed all dependencies: `pip install -r requirements.txt`
2. Activate your virtual environment if you're using one

### Memory Issues

If you encounter memory errors:
1. Try processing smaller images
2. Close other applications to free up memory
3. If using a GPU, try switching to CPU by setting `CUDA_VISIBLE_DEVICES=-1`

### Apple Silicon Issues

If you're using a Mac with Apple Silicon (M1, M2, etc.) and encounter errors:
1. Make sure you're using the `run_app.py` script which sets the correct environment variables
2. Try installing PyTorch with CPU support only if you continue to have issues
