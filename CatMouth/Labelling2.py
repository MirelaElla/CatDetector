from psychopy import visual, core, event, data, logging
import csv
import os
import shutil
from datetime import datetime

# Set up the Window
win = visual.Window(fullscr=True)

# Set up the image stimulus
image_stim = visual.ImageStim(win)

# Path to your images
image_folder = 'Y:/cats_approach_training_2024_05_19/'

# List of image files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Initialize the responses dictionary
responses = {file: None for file in image_files}

# Check if the results file already exists and load it
results_file = 'labels.csv'
savepoints_dir = 'savepoints'  # Directory for savepoints
os.makedirs(savepoints_dir, exist_ok=True)  # Ensure the savepoints directory exists

try:
    # Attempt to open the file to check if it's writable
    with open(results_file, 'a', newline='') as file:
        pass
except IOError:
    logging.error("File not writable!")
    win.close()  # Close the window before quitting
    core.quit()  # Exit the script immediately
else:
    if os.path.exists(results_file):
        with open(results_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and len(row) == 2:  # Assuming each row contains an image and a response
                    responses[row[0]] = row[1]

    # Find the first unresponded image or start from the beginning
    current_index = next((index for index, file in enumerate(image_files) if responses[file] is None), 0)

    # Trial loop
    while current_index < len(image_files):
        trial = image_files[current_index]

        try:
            image_stim.image = os.path.join(image_folder, trial)
            image_stim.draw()
            win.flip()

            # Wait for response or navigation
            keys = event.waitKeys(keyList=['y', 'n', 'p', 'x', 'q', 'left', 'right', 'return'])
            if 'left' in keys:
                if current_index > 0:
                    current_index -= 1  # Go back one image
                continue
            elif 'right' in keys:
                if current_index < len(image_files) - 1:
                    current_index += 1  # Go forward one image
                continue
            elif 'q' in keys:
                break  # Exit loop if 'q' is pressed
            elif 'return' in keys:
                # Save to CSV when 'enter' is pressed and clear responses dictionary
                with open(results_file, 'w', newline='') as file:
                    writer = csv.writer(file, delimiter=',')
                    for img, resp in responses.items():
                        if resp is not None:
                            writer.writerow([img, resp])
                # Create a timestamped backup copy of the labels file
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                backup_file_path = os.path.join(savepoints_dir, f'labels_{timestamp}.csv')
                shutil.copyfile(results_file, backup_file_path)                            
                continue    
            elif keys[0] in ['y', 'n', 'p', 'x']:
                responses[trial] = keys[0]  # Update response in dictionary
                current_index += 1  # Move to next image

        except Exception as e:
            logging.error(f'Failed to load image {trial}: {e}')
            responses[trial] = 'E'  # Indicate an error occurred
            current_index += 1

    # Rewrite the CSV file with updated responses
    with open(results_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for img, resp in responses.items():
            if resp is not None:
                writer.writerow([img, resp])

    # Clean up
    win.close()
    core.quit()
