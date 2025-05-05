

def cvu_m_to_ft (l_m : float) -> float:
    """
    Converts lineal meters "m" into lineal feet "ft"
    """
    l_ft = (l_m/0.0254)/12
    
    return l_ft


def cvu_ton_m2_to_kip_ft2 (ton_m2:float) -> float:
    """
    Converts pressure ton per square meter to kip per square feet.
    """

    kip_ft2= ton_m2 * 0.204816144

    return kip_ft2


def cvu_kg_m3_to_kip_ft3 (kg_m3:float) -> float:
    """
    Converts density kg per cubic meter to kip per cubic feet
    """

    kip_ft3 = kg_m3*0.000062428

    return kip_ft3