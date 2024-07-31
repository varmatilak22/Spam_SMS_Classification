import os

def abs_model_path():
    """
    retruns the absolute path on countvectorizer and model
    """
    #Paths
    current_dir = os.path.dirname(__file__)
    out_dir = os.path.dirname(current_dir)
    out_dir_2 = os.path.join(out_dir, 'model')
    file_path = os.path.join(out_dir_2, 'randomforest.joblib')
    cv_path = os.path.join(out_dir_2, 'vectorizer.pkl')
    
    return file_path,cv_path

def abs_image_path(img_name):
    #current_dir = os.path.dirname(__file__)
    #out_dir = os.path.dirname(current_dir)
    #out_dir_2 = os.path.join(out_dir, 'assests')
    img_path = os.path.join('\mount\src\spam_sms_classification\assests', img_name)
    return img_path

if __name__=="__main__":
    image=abs_image_path('report.png')
    print(image)
