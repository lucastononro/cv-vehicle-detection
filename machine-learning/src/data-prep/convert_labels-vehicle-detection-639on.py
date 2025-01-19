# This file will join both the labels license and licenseplate from the original dataset
import os
import glob


def convert_labels(directory):
    # Find all txt files in the labels directory
    label_files = glob.glob(os.path.join(directory, "**", "labels", "*.txt"), recursive=True)
    
    print(f"Found {len(label_files)} label files in {directory}")
    
    for label_file in label_files:
        # Read the file
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Process each line
        modified = False
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0] == '1':  # If first number is 1
                parts[0] = '0'  # Change to 0
                modified = True
            new_lines.append(' '.join(parts) + '\n')
        
        # Only write back if we made changes
        if modified:
            with open(label_file, 'w') as f:
                f.writelines(new_lines)
            print(f"Modified: {os.path.basename(label_file)}")

if __name__ == "__main__":
    base_dir = "../data/vehicle-detection-639on"
    
    # Process train, valid, and test directories
    for subset in ["train", "valid", "test"]:
        subset_dir = os.path.join(base_dir, subset)
        if os.path.exists(subset_dir):
            print(f"\nProcessing {subset} directory...")
            convert_labels(subset_dir)
        else:
            print(f"Warning: {subset} directory not found at {subset_dir}")
    
    print("\nDone! All label files have been processed.") 