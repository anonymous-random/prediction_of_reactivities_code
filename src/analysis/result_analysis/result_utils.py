import os
import shutil

def supplementary_result_structurer(root_dir, filename, output_dir):
    """
    This function is used to collect the given filenames from their result folders to upload them
    in a separate folder in the osf.
    E.g., it collects all "lin_model_coefficients.json", mirrors the order structure of the ml_results_processed
    and dump the results in an output diredtory named "lasso_coefs"

    Args:
        root_dir (str): The root directory from where the function starts collecting
        filename (str): The name of the file to be collected, e.g. "shap_values.json"
        output_dir (str): The name of the output dir that is created
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Current working directory:", os.getcwd())
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if filename in filenames:
            relative_path = os.path.relpath(dirpath, root_dir)
            new_dir = os.path.join(output_dir, relative_path)
            os.makedirs(new_dir, exist_ok=True)
            src_file = os.path.join(dirpath, filename)
            dst_file = os.path.join(new_dir, filename)
            shutil.copy2(src_file, dst_file)
