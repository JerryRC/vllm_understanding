import os
import json
import numpy as np
import math
import glob

def calculate_metrics(predictions, ground_truths):
    """Calculates MAE, RMSE, and Pearson's R."""
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    num_samples = len(predictions)

    if len(ground_truths) != num_samples or num_samples == 0:
        return None, None, None, 0

    mae = np.mean(np.abs(predictions - ground_truths))
    rmse = np.sqrt(np.mean((predictions - ground_truths)**2))

    # Calculate Pearson's R, handle cases with zero std deviation or < 2 samples
    if num_samples < 2 or np.std(predictions) == 0 or np.std(ground_truths) == 0:
        pearson_r = np.nan # Pearson R is undefined for < 2 points or zero variance
    else:
        corr_matrix = np.corrcoef(predictions, ground_truths)
        # Ensure corr_matrix is the expected 2x2 shape before indexing
        if corr_matrix.shape == (2, 2):
            pearson_r = corr_matrix[0, 1]
        else:
             # Should not happen if num_samples >= 2 and std devs are non-zero, but added as safety
            pearson_r = np.nan

    return mae, rmse, pearson_r, num_samples

def process_results_file(file_path):
    """Reads a jsonl file and extracts predictions and ground truths."""
    predictions = []
    ground_truths = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Ensure both keys exist and values are numbers
                    if 'predicted_count' in data and 'ground_truth_number' in data and \
                       isinstance(data['predicted_count'], (int, float)):
                        # Convert ground truth to integer if it's a string
                        if isinstance(data['ground_truth_number'], str):
                            try:
                                ground_truth = int(data['ground_truth_number'])
                            except ValueError:
                                print(f"Skipping invalid ground truth number in {file_path}: {data['ground_truth_number']}")
                                continue
                        else:
                            ground_truth = data['ground_truth_number']
                            
                        predictions.append(data['predicted_count'])
                        ground_truths.append(ground_truth)
                    else:
                        print(f"Skipping invalid line in {file_path}: {line.strip()}")
                except json.JSONDecodeError:
                    print(f"Skipping non-JSON line in {file_path}: {line.strip()}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None
    return predictions, ground_truths

def main():
    analysis_dir = "analysis_results_repcount"
    results_files = glob.glob(os.path.join(analysis_dir, "*", "all_results.jsonl"))

    all_metrics = []

    for file_path in sorted(results_files):
        model_name = os.path.basename(os.path.dirname(file_path))
        print(f"Processing: {file_path} (Model: {model_name})")

        predictions, ground_truths = process_results_file(file_path)

        if predictions is not None and ground_truths is not None:
            mae, rmse, pearson_r, num_samples = calculate_metrics(predictions, ground_truths)

            if num_samples > 0:
                all_metrics.append({
                    "model": model_name,
                    "mae": mae,
                    "rmse": rmse,
                    "pearson_r": pearson_r if not math.isnan(pearson_r) else 'N/A',
                    "num_samples": num_samples
                })
                print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, Pearson's R: {f'{pearson_r:.4f}' if not math.isnan(pearson_r) else 'N/A'}, Samples: {num_samples}")
            else:
                print(f"  No valid samples found.")
        else:
             print(f"  Skipping due to read errors or no data.")

    # Optional: Print results as a Markdown table
    if all_metrics:
        print("\n--- Results Summary ---")
        # Sort by a metric if desired, e.g., MAE
        # all_metrics.sort(key=lambda x: x['mae'])

        print("| Model                           |   MAE |  RMSE | Pearson's R |   Samples |")
        print("| :------------------------------ | ----: | ----: | ----------: | --------: |")
        for metrics in all_metrics:
            r_val = f"{metrics['pearson_r']:.4f}" if isinstance(metrics['pearson_r'], (int, float)) and not math.isnan(metrics['pearson_r']) else 'N/A'
            print(f"| {metrics['model']:<31} | {metrics['mae']:>5.2f} | {metrics['rmse']:>5.2f} | {r_val:>11} | {metrics['num_samples']:>9} |")

if __name__ == "__main__":
    main() 