import os

def abs_model_path():
    """
    Returns the absolute paths for the CountVectorizer and model files.
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)
    
    # Navigate to the parent directory
    out_dir = os.path.dirname(current_dir)
    
    # Navigate to the 'model' subdirectory
    out_dir_2 = os.path.join(out_dir, 'model')
    
    # Construct the paths for the model and vectorizer files
    file_path = os.path.join(out_dir_2, 'randomforest.joblib')
    cv_path = os.path.join(out_dir_2, 'vectorizer.pkl')
    
    return file_path, cv_path

def abs_image_path(img_name):
    """
    Returns the absolute path for an image file.
    
    Args:
        img_name (str): The name of the image file.
    
    Returns:
        str: The absolute path to the image file.
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)
    
    # Navigate to the parent directory
    out_dir = os.path.dirname(current_dir)
    
    # Navigate to the 'assets' subdirectory
    out_dir_2 = os.path.join(out_dir, 'assets')
    
    # Construct the path for the image file
    img_path = os.path.join(out_dir_2, img_name)
    
    return img_path

if __name__ == "__main__":
    # Test the abs_image_path function with an example image
    image = abs_image_path('report.png')
    print(f"Absolute path to the image: {image}")
    
    # Test the abs_model_path function
    model_path, vectorizer_path = abs_model_path()
    print(f"Absolute path to the model: {model_path}")
    print(f"Absolute path to the vectorizer: {vectorizer_path}")
