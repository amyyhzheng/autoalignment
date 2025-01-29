
    filetype_str = '.I16'
    
    # Get all .I16 files
    files = [f for f in os.listdir(path_) if f.endswith(filetype_str)]
    
    # Extract (prefix, Z number) pairs
    file_tuples = []
    pattern = re.compile(r'^(.*_Z)(\d+)(\.I16)$')

    for filename in files:
        match = pattern.match(filename)
        if match:
            prefix, z_num, ext = match.groups()
            file_tuples.append((prefix, int(z_num), ext))  # Store Z-number as an integer

    # Sort files by (prefix, Z-number)
    file_tuples.sort(key=lambda x: (x[0], x[1]))

    # Reconstruct filenames in correct order
    ordered_file_list = [f"{prefix}{z_num}{ext}" for prefix, z_num, ext in file_tuples]

    print(ordered_file_list)
    return ordered_file_list
