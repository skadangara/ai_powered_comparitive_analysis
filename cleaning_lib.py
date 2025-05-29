import json


def clean_zkp_predicted_label(zkp_predicted_label):
    label_split = zkp_predicted_label.split(",")
    if (len(label_split) == 0):
        label_split = zkp_predicted_label.split(":")
        if(len(label_split) == 0):
            curated_label = "na"
        else:
            curated_label = label_split[0].strip().lower()
    else:
        curated_label = label_split[0].strip().lower()

    return curated_label


def curate_extracted_dimensions(comparison_dimensions, project_name, dimensions):
    project_dim = {}
    data_json = json.loads(dimensions)
    dim_data = data_json['dimensions']
    parsed_data = {entry['dimension']: entry['details'] for entry in dim_data}
    project_dim["project_name"] = project_name
    for i in comparison_dimensions:
        if(i in parsed_data):
            project_dim[i] = parsed_data[i]
    return project_dim
