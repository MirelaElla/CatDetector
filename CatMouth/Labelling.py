from psychopy import visual, core, event, data, logging
import csv
import os

# Set up the Window
win = visual.Window(fullscr=True)

# Set up the image stimulus
image_stim = visual.ImageStim(win)

# Path to your images
image_folder = 'Y:/cats/'

# List of image files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Check if the results file already exists and find the last image shown
last_image = None
results_file = 'labels.csv'
if os.path.exists(results_file):
    with open(results_file, 'r') as file:
        reader = csv.reader(file)
        last_row = None
        for row in reader:
            last_row = row
        if last_row is not None:
            last_image = last_row[0]

# Find the index of the last image shown
start_index = 0
if last_image in image_files:
    start_index = image_files.index(last_image) + 1

# Trial handler
trials = data.TrialHandler(image_files[start_index:], nReps=1, method='sequential')

# Data file
dataFile = open(results_file, 'a', newline='')  # Open file in append mode
writer = csv.writer(dataFile, delimiter=',')

# Experiment loop
for trial in trials:
    image_stim.image = os.path.join(image_folder, trial)
    image_stim.draw()
    win.flip()

    # Wait for response
    keys = event.waitKeys(keyList=['y', 'n', 'p', 'q'])

    # Write data
    writer.writerow([trial, keys[0]])

    # Check if 'q' was pressed to quit
    if 'q' in keys:
        break

# Close the data file
dataFile.close()

# Clean up
win.close()
core.quit()


