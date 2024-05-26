import matplotlib.pyplot as plt
import random

from yolov7.configuration import get_config

config = get_config()

import os
import json


def extract_results(detections_dir, labels_dir, classes=None):
    # Get the list of detection files
    detection_files = os.listdir(detections_dir)

    is_labeled = os.path.isdir(labels_dir)

    # Initialize dictionaries to store the results and confidences
    results = {}
    confidences = {}
    bboxes = {}

    # Read the detection files and populate the results dictionary
    for file in detection_files:
        detection_path = f"{detections_dir}/{file}"
        label_path = f"{labels_dir}/{file}"

        # Read detection files
        with open(detection_path, 'r') as pred_file:
            detections = pred_file.read().splitlines()

        # Remove file extension from the filename
        filename = os.path.splitext(file)[0]

        # Initialize the results and confidences for the current file
        results[filename] = {}
        confidences[filename] = {}
        bboxes[filename] = {}

        # Extract classes_ids and confidences from detections
        for detection in detections:
            if len(detection.split()) == 6:
                class_id, confidence, min_x, min_y, max_x, max_y = detection.split()
                confidences[filename][class_id] = float(confidence)
                bboxes[filename][class_id] = {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}

        if is_labeled:
            with open(label_path, 'r') as label_file:
                labels = label_file.read().splitlines()
            # Compare labels with detections and populate the results dictionary
            for label in labels:
                if len(label.split()) == 5:
                    class_id, min_x, min_y, max_x, max_y = label.split()

                    if class_id in results[filename]:
                        continue  # Skip duplicate labels if present

                    if class_id in confidences[filename]:
                        # Object is present and detected
                        results[filename][class_id] = 1
                    else:
                        # Object is present but not detected
                        results[filename][class_id] = 0
        else:  # No labels provided, just record if the object was detected (1) or no (None)
            if classes is None:
                # TODO find a way to extract the number of classes from the model for this case
                pass
            else:
                class_ids = [str(class_id) for class_id in range(len(classes))]
            for class_id in class_ids:
                if class_id in confidences[filename]:
                    # Object detected
                    results[filename][class_id] = 1
                # else leave as value None to live place blank
                else:
                    results[filename][class_id] = 0

    return results, confidences, bboxes, detection_files, is_labeled


def create_gannt_diagram(results, filenames, problematic_classes=[],
                         class_names=None, path='./', model_id=None,
                         dataset_type="", save_plot=True, show_plot=False, is_labeled=False):
    class_colors = {}
    # Extract class_ids from the results dictionary
    if class_names is not None:
        classes = set()
        for filename in results:
            classes.update(results[filename].keys())
    else:
        pass  # TODO find a way to extract number of classes from the model, then classes = [str(class_id) for class_id in range(nb_of_classes)]

    # Sort the class_ids
    classes = sorted(classes)
    # clear the plots
    plt.clf()
    # Plot the Gantt diagram
    fig, ax = plt.subplots(figsize=(12, 7))

    # Set the x and y-axis ticks and labels
    ax.set_yticks(range(len(classes)))

    if class_names is not None:
        ax.set_yticklabels(class_names)
    else:
        ax.set_yticklabels(classes)

    # ax.set_xticks(range(len(filenames)))
    ax.set_xticks([tick + 0.5 for tick in range(len(filenames))])
    ax.set_xticklabels([os.path.splitext(file)[0] for file in filenames], rotation=90, ha='center')

    # Set the x and y -axis label
    ax.set_xlabel('Images/frames')
    ax.set_ylabel('Classes')

    present_classes = set()
    # Iterate over the results and plot the Gantt bars
    for i, filename in enumerate(results):
        for j, label in enumerate(classes):
            detected = results[filename].get(label)
            # check if color for the class exists and if no, generate a random one
            if label not in class_colors.keys():
                class_colors[label] = generate_random_class_color()
            if detected is not None:
                if detected:
                    present_classes.update(label)
                    color = class_colors[label]
                    ax.barh(j, 1, left=i, color=color)
                # If there were labels provided we have 3 classes Detected (1), Not Detected (0) and Not present (None),
                # therefore in this situations Not Detected cases are marked with color with higher transparency
                elif is_labeled:
                    color = class_colors[label] + '35'  # Increase transparency
                    ax.barh(j, 1, left=i, color=color)

    # add empty white barplots for the classes that have 0 detections
    for j, class_id in enumerate(classes):
        if str(class_id) not in present_classes:
            ax.barh(j, 1, left=0, color='white')

            # Adjust the plot layout
    plt.tight_layout()
    # Add grid lines
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

    if is_labeled:
        plt.figtext(0.5, 0.01,
                    "GANNT Diagram of detections, the bright color (higher transparency) means missed detection",
                    wrap=True, horizontalalignment='center')

    # Change the color of y-axis labels
    for label in ax.get_yticklabels():
        if label.get_text() in problematic_classes:
            label.set_color('red')
            label.set_weight('bold')

    if save_plot:
        save_filepath = f"{path}/gannt_diagram_{dataset_type}{f'_{model_id}' if model_id is not None else ''}.png"
        plt.savefig(save_filepath)

    if show_plot:
        plt.show()

    return save_filepath


def generate_random_class_color():
    color = '#' + ''.join(random.choices('0123456789ABCDEF', k=6))
    return color


def combine_results(results, confidences, bboxes, save_dir=None, dataset_type=None, class_names=None):
    combined_results = {}

    for img in results:
        preds = []
        for class_id in results[img]:
            detection = bool(results[img][class_id])
            if class_names is not None:
                class_name = class_names[int(class_id)]
            else:
                class_name = class_id
            result = {class_name: {'detection': detection}}
            if detection:
                result[class_name]["bbox"] = bboxes[img][class_id]
                result[class_name]["confidence"] = confidences[img][class_id]
            preds.append(result)
        combined_results[img] = preds

    if save_dir is not None:
        save_path = f"{save_dir}/test_results{f'_{dataset_type}' if dataset_type is not None else ''}.json"
        with open(save_path, 'w') as f:
            json.dump(combined_results, f)

    return combined_results


def get_problematic_classes(combined_results):
    problematic_classes = set()
    problematic_classes.update(_get_never_detected_classes(combined_results))
    problematic_classes.update(_get_classes_not_detected_in_n_consecutive(combined_results, config.get_config(
        'not_detected_frames_threshold', 3)))
    # TODO add new 'problematic classes' cases

    return problematic_classes


def _get_never_detected_classes(eval_results):
    detected_classes = set()
    all_classes = set()

    for img in eval_results:
        for class_data in eval_results[img]:
            for class_name, class_info in class_data.items():
                all_classes.add(class_name)
                if class_info.get('detection'):
                    detected_classes.add(class_name)

    undetected_classes = all_classes - detected_classes

    return undetected_classes


def _get_classes_not_detected_in_n_consecutive(data, n):
    classes_not_detected_in_n_consecutive = set()
    consecutive_misses = {}

    for img in data:
        for class_data in data[img]:
            for class_name, class_info in class_data.items():
                if not class_info.get('detection'):
                    consecutive_misses.setdefault(class_name, 0)
                    consecutive_misses[class_name] += 1
                else:
                    consecutive_misses[class_name] = 0

                if consecutive_misses[class_name] >= n:
                    classes_not_detected_in_n_consecutive.add(class_name)

    return classes_not_detected_in_n_consecutive

### TEST for Gannt diagram
# detections_dir = 'C:\\Users\\synth\\Documents\\develop\\synthavo-models-v2\\test_gannt\\preds'
# labels_dir = 'C:\\Users\\synth\\Documents\\develop\\synthavo-models-v2\\test_gannt\\labels'
# labels_dir_non_exist='C:\\Users\\synth\\Documents\\develop\\synthavo-models-v2\\test_gannt\\non_existent'

# class_names = ['class1', 'class2', 'class3', 'class4', 'class5']

# # results, confidences, bboxes, filenames, is_labeled = extract_results(detections_dir, labels_dir)
# results, confidences, bboxes, filenames, is_labeled = extract_results(detections_dir, labels_dir_non_exist, classes=class_names)
# combined_results = combine_results(results, confidences, bboxes, save_dir='./', dataset_type='images', class_names=class_names)
# problematic_classes = get_problematic_classes(combined_results)
# gannt_filepath = create_gannt_diagram(results, filenames, class_names=class_names, problematic_classes=problematic_classes, model_id=10, show_plot=True, is_labeled=is_labeled)


##### ----------------------------------