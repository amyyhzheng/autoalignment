def get_i16_file_dict(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        return {}

    # Get all files in the folder that end with .I16
    file_paths = [f for f in os.listdir(folder_path) if f.endswith('.I16')]

    # Regex pattern to extract the number after 'Z' and before '.I16'
    pattern = re.compile(r'_Z(\d+)\.I16$')

    # Dictionary to store extracted numbers (as integers) and file paths
    file_dict = {}
    file_list = []

    for file in file_paths:
        match = pattern.search(file)
        if match:
            z_number = int(match.group(1))  # Ensure it's stored as an integer
            file_dict[z_number] = os.path.join(folder_path, file)  # Store full path
            print(f"Added key: {z_number}, Type: {type(z_number)}")  # Debugging line
            file_list.append(z_number)

    # Explicitly enforce integer sorting
    sorted_keys = sorted(file_dict.keys())  # Ensure keys are sorted as integers
    sorted_dict = {k: file_dict[k] for k in sorted_keys}

    # Print sorted keys and their corresponding values
    print("\nSorted Dictionary Output:")
    for key in sorted_dict:
        print(f"{key}: {sorted_dict[key]}")  # This should show sorted integer keys

    return sorted_dict.values()
