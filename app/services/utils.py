def validate_input(data):
    """
    Validates the input data for predicting house prices.

    Args:
        data (dict): The input JSON data received from the client.

    Returns:
        bool: True if the input is valid, False otherwise.
    """
    required_types = {
        "area_m2": (int, float),  
        "bedrooms": int,         
        "bathrooms": int,        
        "parking": int,          
        "stratum": int,          
        "year_built": int,       
        "neighborhood": str      
    }

    if not isinstance(data, dict):
        return False

    for feature, expected_type in required_types.items():
        if feature not in data:
            return False
        if not isinstance(data[feature], expected_type):
            return False
        if isinstance(data[feature], (int, float)) and data[feature] < 0:
            return False
        if feature=='area_m2' and data[feature] == 0:
            return False
        if feature=='stratum' and (data[feature] < 1 or data[feature] > 6):
            return False

    return True
