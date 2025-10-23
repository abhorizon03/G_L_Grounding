import torch
from scipy import ndimage
from skimage.filters import threshold_multiotsu


def nanvar(tensor):
    tensor_mean = tensor.nanmean()
    output = (tensor - tensor_mean).square().nanmean()
    return output


def largest_connected_component(binary_image):
    # Label all connected components
    labeled_image, num_features = ndimage.label(binary_image)
    
    # Measure the size of each component
    component_sizes = ndimage.sum(binary_image, labeled_image, range(num_features + 1))
    
    # Get the label of the largest component
    largest_component_label = component_sizes.argmax()
    
    # Create a new binary image where only the largest component is set to 1 (True)
    largest_component = (labeled_image == largest_component_label)
    
    return largest_component

def get_bb_from_largest_component(largest_component):
    """
    Returns the predicted bounding box in (x, y, w, h) format
    """

    # Get the slice (bounding box) for the largest component
    bounding_box_slice = ndimage.find_objects(largest_component)[0]
    y_min, y_max = bounding_box_slice[0].start, bounding_box_slice[0].stop
    x_min, x_max = bounding_box_slice[1].start, bounding_box_slice[1].stop
    
    return x_min, y_min, x_max - x_min, y_max - y_min

def get_bb_and_largest_component_from_diff(diff):
    thresholds = threshold_multiotsu(diff, classes = 2)
    diff_binary = diff > thresholds
    largest_comp = largest_connected_component(diff_binary)
    bounding_box = get_bb_from_largest_component(largest_comp)
    return largest_comp, bounding_box