class TypeToColor:
    """
    A manager for handling multiple type-to-color mappings. General manager 
    """
    _mappings = {}

    @classmethod
    def add_mapping(cls, mapping_name, mapping):
        """
        Add a new type-to-color mapping.
        """
        if not isinstance(mapping, dict):
            raise TypeError("Mapping must be a dictionary")
        cls._mappings[mapping_name] = mapping

    @classmethod
    def get_mapping(cls, mapping_name):
        """
        Retrieve a specific type-to-color mapping.
        """
        return cls._mappings.get(mapping_name, {})

    @classmethod
    def update_mapping(cls, mapping_name, key, value):
        """
        Update a specific type-to-color mapping with a new key-value pair.
        """
        if mapping_name not in cls._mappings:
            raise ValueError(f"Mapping '{mapping_name}' does not exist")
        cls._mappings[mapping_name][key] = value

    @classmethod
    def list_mappings(cls):
        """
        List all available mapping names.
        """
        return list(cls._mappings.keys())
    
    @classmethod
    def map_types_to_colors(cls, types, mapping_name):
        """
        Map types to colors using the specified type-to-color mapping.
        """
        mapping = cls.get_mapping(mapping_name)
        valid_colors = []
        for t in types:
            color = mapping.get(t, 'black')
            if color.startswith('#'):  # Convert hex to normalized RGBA
                color = tuple(int(color.lstrip('#')[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            valid_colors.append(color)
        return valid_colors
