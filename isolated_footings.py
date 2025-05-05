import pandas as pd
import numpy as np

#Data manipulation


def df_from_etabs_file (file_name : str) -> pd.DataFrame:
    """Takes a xlsx file of the joint base reactions from a etabs model
    since it is a etabs file a sheet name "Joint Reactions" will always be used
    columns C,D,F,G,H,I,J,K corresponding to Unique Name, Output Case, FX, FY, FZ, 
    MX, MY and MZ will always be used. 
    Right hand sign convention will be used. 
    """
    df_etabs_file = pd.read_excel(file_name, sheet_name="Joint Reactions", usecols="C,D,F,G,H,I,J,K", skiprows=[0,2]).set_index(["Unique Name", "Output Case"])

    cols_changed_values = ["FX", "FY", "FZ", "MY", "MZ"]

    df_etabs_file[cols_changed_values] = df_etabs_file[cols_changed_values].apply(lambda x: -x)

    return df_etabs_file


def df_from_etabs_file_st(file) -> pd.DataFrame:
    """
    Toma un archivo Excel subido desde Streamlit o un archivo local, y devuelve el DataFrame
    de reacciones de apoyo de ETABS usando convención de signos positiva hacia abajo.
    
    `file` puede ser una ruta (str) o un archivo subido (`UploadedFile` de Streamlit).
    """
    df_etabs_file = pd.read_excel(file, sheet_name="Joint Reactions", usecols="C,D,F,G,H,I,J,K", skiprows=[0,2])
    df_etabs_file.set_index(["Unique Name", "Output Case"], inplace=True)

    cols_changed_values = ["FX", "FY", "FZ", "MY", "MZ"]
    df_etabs_file[cols_changed_values] = df_etabs_file[cols_changed_values].apply(lambda x: -x)

    return df_etabs_file


def df_filter_load_cases(df: pd.DataFrame, load_cases_used:list ) -> pd.DataFrame:
    """
    Filters a DataFrame keeping only the list on load cases used in the secundary index ("Output Case")
    eliminates the ones that are on the this list. 
    df : Structured DataFrame with the primary index as "Unique Name", and secundary index "Output Case". 
    load_cases_used: a list of wich load cases where used in a particular model. 
    Returns a new DataFrame with the load cases filtered. 
    If one of the items specified in the list does not exist in the "Output Case" index, a warning message is rasied. 
    """
    
    existing_dataframe_cases = df.index.get_level_values("Output Case").unique()
    
    cases_not_existing = [c for c in load_cases_used if c not in existing_dataframe_cases]   
    
    if cases_not_existing:
        print(f"Warning: Following load cases don't exist in the DataFrame: "
              f"{','.join(cases_not_existing)}", end="")
    
    df_filtered = df[df.index.get_level_values("Output Case").isin(load_cases_used)]
    
    return df_filtered


def filtered_contextual_foundation_service_combos(llrf : float = 1.00, scd: float =0.0, lateral_load_cases: list = None, decimals: int =4) -> dict:
    """
    Returns a dictionary of the AGIES 2024 service load combinations 
    for foundation analysis keyed by load combo name.
    fr_v : live load reduction factor
    scd: equal to (2/3 * scs) also equal to Sds per ASCE.
    """
    AGIES_2024_FOUNDATION_SERVICE_COMBOS = {
    "D+(llrf)L": {"DEAD": 1.0, "LIVE": 1.0 * llrf},
    "(1+Svd)D+(llrf)L+0.70QX": {"DEAD": (1.0 + (0.14 * scd)), "LIVE": 1.0 * llrf, "QX": 0.7},
    "(1+Svd)D+(llrf)L-0.70QX": {"DEAD": (1.0 + (0.14 * scd)), "LIVE": 1.0 * llrf, "QX": -0.7},
    "(1+Svd)D+(llrf)L+0.70QX+AT": {"DEAD": (1.0 + (0.14 * scd)), "LIVE": 1.0 * llrf, "QX+AT": 0.7},
    "(1+Svd)D+(llrf)L-0.70QX+AT": {"DEAD": (1.0 + (0.14 * scd)), "LIVE": 1.0 * llrf, "QX+AT": -0.7},
    "(1+Svd)D+(llrf)L+0.70QX-AT": {"DEAD": (1.0 + (0.14 * scd)), "LIVE": 1.0 * llrf, "QX-AT": 0.7},
    "(1+Svd)D+(llrf)L-0.70QX-AT": {"DEAD": (1.0 + (0.14 * scd)), "LIVE": 1.0 * llrf, "QX-AT": -0.7},
    "(1+Svd)D+(llrf)L+0.70QY": {"DEAD": (1.0 + (0.14 * scd)), "LIVE": 1.0 * llrf, "QY": 0.7},
    "(1+Svd)D+(llrf)L-0.70QY": {"DEAD": (1.0 + (0.14 * scd)), "LIVE": 1.0 * llrf, "QY": -0.7},
    "(1+Svd)D+(llrf)L+0.70QY+AT": {"DEAD": (1.0 + (0.14 * scd)), "LIVE": 1.0 * llrf, "QY+AT": 0.7},
    "(1+Svd)D+(llrf)L-0.70QY+AT": {"DEAD": (1.0 + (0.14 * scd)), "LIVE": 1.0 * llrf, "QY+AT": -0.7},
    "(1+Svd)D+(llrf)L+0.70QY-AT": {"DEAD": (1.0 + (0.14 * scd)), "LIVE": 1.0 * llrf, "QY-AT": 0.7},
    "(1+Svd)D+(llrf)L-0.70QY-AT": {"DEAD": (1.0 + (0.14 * scd)), "LIVE": 1.0 * llrf, "QY-AT": -0.7},
    "D+0.70QX": {"DEAD": 1.0, "QX": 0.7},
    "D-0.70QX": {"DEAD": 1.0, "QX": -0.7},
    "D+0.70QX+AT": {"DEAD": 1.0, "QX+AT": 0.7},
    "D-0.70QX+AT": {"DEAD": 1.0, "QX+AT": -0.7},
    "D+0.70QX-AT": {"DEAD": 1.0, "QX-AT": 0.7},
    "D-0.70QX-AT": {"DEAD": 1.0, "QX-AT": -0.7},
    "D+0.70QY": {"DEAD": 1.0, "QY": 0.7},
    "D-0.70QY": {"DEAD": 1.0, "QY": -0.7},
    "D+0.70QY+AT": {"DEAD": 1.0, "QY+AT": 0.7},
    "D-0.70QY+AT": {"DEAD": 1.0, "QY+AT": -0.7},
    "D+0.70QY-AT": {"DEAD": 1.0, "QY-AT": 0.7},
    "D-0.70QY-AT": {"DEAD": 1.0, "QY-AT": -0.7},
    }

    if lateral_load_cases:
        AGIES_2024_FOUNDATION_SERVICE_COMBOS = {
            combo: factors
            for combo, factors in AGIES_2024_FOUNDATION_SERVICE_COMBOS.items()
            if all(load in lateral_load_cases for load in factors if load not in ["DEAD", "LIVE"])
        }

    def round_values(data):
        if isinstance(data, dict):
            return {k: round_values(v) for k, v in data.items()}
        elif isinstance(data, (int,float)):
            return round(data, decimals) 
        return data
        
    return round_values(AGIES_2024_FOUNDATION_SERVICE_COMBOS)


def filtered_contextual_foundation_strength_combos(llrf : float = 1.0, scd: float = 0.0, kz: float = 1.0, lateral_load_cases: list = None, decimals: int = 4) -> dict:
    """
    Returns a dictionary of the AGIES 2024 strength load combinations 
    for foundation analysis keyed by load combo name.
    fr_v : live load reduction factor
    scd: equal to (2/3 * scs) also equal to Scd per ASCE.
    kz: seismic amplification factor. 1 for sizing, 1.2 for site classified as C, D, E and F, 
    1.40 for site classified as A and B.
    """
    AGIES_2024_FOUNDATION_STRENGTH_COMBOS = {
    "1.2D+(1.6*llrf)L": {"DEAD": 1.2, "LIVE": 1.6 * llrf},
    "(1.2+Svd)D+(llrf)L+(kz)QX": {"DEAD": (1.2 + (0.2 * scd)), "LIVE": 1.0 * llrf, "QX": 1.0 * kz},
    "(1.2+Svd)D+(llrf)L-(kz)QX": {"DEAD": (1.2 + (0.2 * scd)), "LIVE": 1.0 * llrf, "QX": -1.0 * kz},
    "(1.2+Svd)D+(llrf)L+(kz)QX+AT": {"DEAD": (1.2 + (0.2 * scd)), "LIVE": 1.0 * llrf, "QX+AT": 1.0 * kz},
    "(1.2+Svd)D+(llrf)L-(kz)QX+AT": {"DEAD": (1.2 + (0.2 * scd)), "LIVE": 1.0 * llrf, "QX+AT": -1.0 * kz},
    "(1.2+Svd)D+(llrf)L+(kz)QX-AT": {"DEAD": (1.2 + (0.2 * scd)), "LIVE": 1.0 * llrf, "QX-AT": 1.0 * kz},
    "(1.2+Svd)D+(llrf)L-(kz)QX-AT": {"DEAD": (1.2 + (0.2 * scd)), "LIVE": 1.0 * llrf, "QX-AT": -1.0 * kz},
    "(1.2+Svd)D+(llrf)L+(kz)QY": {"DEAD": (1.2 + (0.2 * scd)), "LIVE": 1.0 * llrf, "QY": 1.0 * kz},
    "(1.2+Svd)D+(llrf)L-(kz)QY": {"DEAD": (1.2 + (0.2 * scd)), "LIVE": 1.0 * llrf, "QY": -1.0 * kz},
    "(1.2+Svd)D+(llrf)L+(kz)QY+AT": {"DEAD": (1.2 + (0.2 * scd)), "LIVE": 1.0 * llrf, "QY+AT": 1.0 * kz},
    "(1.2+Svd)D+(llrf)L-(kz)QY+AT": {"DEAD": (1.2 + (0.2 * scd)), "LIVE": 1.0 * llrf, "QY+AT": -1.0 * kz},
    "(1.2+Svd)D+(llrf)L+(kz)QY-AT": {"DEAD": (1.2 + (0.2 * scd)), "LIVE": 1.0 * llrf, "QY-AT": 1.0 * kz},
    "(1.2+Svd)D+(llrf)L-(kz)QY-AT": {"DEAD": (1.2 + (0.2 * scd)), "LIVE": 1.0 * llrf, "QY-AT": -1.0 * kz},
    "(1.0-Svd)D+(kz)QX": {"DEAD": (1.0 - (0.2 * scd)), "QX": 1.0 * kz},
    "(1.0-Svd)D-(kz)QX": {"DEAD": (1.0 - (0.2 * scd)), "QX": -1.0 * kz},
    "(1.0-Svd)D+(kz)QX+AT": {"DEAD": (1.0 - (0.2 * scd)), "QX+AT": 1.0 * kz},
    "(1.0-Svd)D-(kz)QX+AT": {"DEAD": (1.0 - (0.2 * scd)), "QX+AT": -1.0 * kz},
    "(1.0-Svd)D+(kz)QX-AT": {"DEAD": (1.0 - (0.2 * scd)), "QX-AT": 1.0 * kz},
    "(1.0-Svd)D-(kz)QX-AT": {"DEAD": (1.0 - (0.2 * scd)), "QX-AT": -1.0 * kz},
    "(1.0-Svd)D+(kz)QY": {"DEAD": (1.0 - (0.2 * scd)), "QY": 1.0 * kz},
    "(1.0-Svd)D-(kz)QY": {"DEAD": (1.0 - (0.2 * scd)), "QY": -1.0 * kz},
    "(1.0-Svd)D+(kz)QY+AT": {"DEAD": (1.0 - (0.2 * scd)), "QY+AT": 1.0 * kz},
    "(1.0-Svd)D-(kz)QY+AT": {"DEAD": (1.0 - (0.2 * scd)), "QY+AT": -1.0 * kz},
    "(1.0-Svd)D+(kz)QY-AT": {"DEAD": (1.0 - (0.2 * scd)), "QY-AT": 1.0 * kz},
    "(1.0-Svd)D-(kz)QY-AT": {"DEAD": (1.0 - (0.2 * scd)), "QY-AT": -1.0 * kz},
    }

    if lateral_load_cases:
        AGIES_2024_FOUNDATION_STRENGTH_COMBOS = {
            combo: factors
            for combo, factors in AGIES_2024_FOUNDATION_STRENGTH_COMBOS.items()
            if all(load in lateral_load_cases for load in factors if load not in ["DEAD", "LIVE"])
        }

    def round_values(data):
        if isinstance(data, dict):
            return {k: round_values(v) for k, v in data.items()}
        elif isinstance(data, (int,float)):
            return round(data, decimals) 
        return data
        
    return round_values(AGIES_2024_FOUNDATION_STRENGTH_COMBOS)


def base_df_service (filtered_service_combos: dict, filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame representing the results from selected load combinations, in the format in which
    it was used (either ETABS or SAP)
    filtered_service_combos: a dictionary filtered only with the combos with the lateral load cases selected (used). 
    filtered_df: a df filtered only with the load cases (gravity and lateral) that will be used. 
    NOTE: All dictionary keys must be consistent with the names of the index "Output Case" from the filtered_df. 
    """
    
    acc = []

    for combos_name, factors in filtered_service_combos.items():
        df_comb = sum(filtered_df.xs(load, level="Output Case")*factor
                  for load, factor in factors.items())
        df_comb ["Load Combination"] = combos_name
        acc.append(df_comb)

    df_final= pd.concat(acc) 
    df_final.set_index("Load Combination", append=True, inplace=True)

    
    ordered_combos = list(filtered_service_combos.keys())
    df_final = df_final.reorder_levels(["Unique Name", "Load Combination"])
    df_final.index = df_final.index.set_levels(
        [df_final.index.levels[0], pd.CategoricalIndex(df_final.index.levels[1], categories=ordered_combos, ordered=True)]
    )
    df_final.sort_index(level=["Unique Name", "Load Combination"], inplace=True)

    return df_final


def base_df_strength (filtered_strength_combos: dict, filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame representing the results from selected load combinations, in the format in which
    it was used (either ETABS or SAP)
    filtered_strength_combos: a dictionary filtered only with the combos with the lateral load cases selected (used). 
    filtered_df: a df filtered only with the load cases (gravity and lateral) that will be used. 
    NOTE: All dictionary keys must be consistent with the names of the index "Output Case" from the filtered_df. 
    """
    
    acc = []

    for combos_name, factors in filtered_strength_combos.items():
        df_comb = sum(filtered_df.xs(load, level="Output Case")*factor
                  for load, factor in factors.items())
        df_comb ["Load Combination"] = combos_name
        acc.append(df_comb)

    df_final= pd.concat(acc) 
    df_final.set_index("Load Combination", append=True, inplace=True)

    
    ordered_combos = list(filtered_strength_combos.keys())
    df_final = df_final.reorder_levels(["Unique Name", "Load Combination"])
    df_final.index = df_final.index.set_levels(
        [df_final.index.levels[0], pd.CategoricalIndex(df_final.index.levels[1], categories=ordered_combos, ordered=True)]
    )
    df_final.sort_index(level=["Unique Name", "Load Combination"], inplace=True)

    return df_final


#Other calculations


def secmod_about_y (ly:float, lx:float) -> float: 
    """
    Returns section modulus of a rectangular shape about y axis 
    """

    sm_y= (1/6)*ly*lx*lx

    return sm_y


def secmod_about_x (ly:float, lx:float) -> float: 
    """
    Returns section modulus of a rectangular shape about x axis 
    """

    sm_x= (1/6)*ly*ly*lx

    return sm_x


def self_weight_soil_and_concrete (lx: float, ly: float, h1: float, h2: float, d_soil:float, d_concrete:float) -> float: 
    """
    Calculates the selfweithg of the soil over the footing and the footing itself. 
    """

    pt=((lx*ly*h2)*d_soil)+((lx*ly*h1)*d_concrete)

    return pt


def pressure_at_corners(row):
    """
    Returns the equivalente pressure at each corner from axial and flexural demans. 
    P1: bottom left
    P2: upper left
    P3: upper right
    P4: bottom right
    """
    
    sy = np.sign(row["MY_Total"])  
    sx = np.sign(row["MX_Total"])  

    signos = [
        (-1 * sy, -1 * sx),  # P1: inferior izquierda
        (-1 * sy,  1 * sx),  # P2: superior izquierda
        ( 1 * sy,  1 * sx),  # P3: superior derecha
        ( 1 * sy, -1 * sx),  # P4: inferior derecha
    ]

    p0 = row["P/A"]
    my_sy = row["MY/Sy"]  
    mx_sx = row["MX/Sx"]  

    presiones = []
    for s_my, s_mx in signos:
        p = p0 + s_my * my_sy + s_mx * mx_sx
        presiones.append(p)

    return pd.Series(presiones, index=["P1", "P2", "P3", "P4"])


def padm_from_index(row, cs: float = 1.0, fa_cs: float = 1.33) -> float:
    """
    Calculates Admisible pressure, depending on the load combination. If it includes any seismic load
    admisible pressure magnifies. 
    cs: Allowable pressure
    fa_cs: amplification factor for seismic demands
    """
    carga_sismica = ["QX", "QY", "QX+AT", "QX-AT", "QY+AT", "QY-AT"]
    nombre_combo = row.name[1]  # Accede al segundo nivel del índice
    if any(s in nombre_combo for s in carga_sismica):
        return cs * fa_cs
    return cs


def get_df_service(df_service: pd.DataFrame, pt: float, h1: float, h2: float,
                                lx: float, ly: float, sx: float, sy: float,
                                cs: float, fa_cs: float) -> pd.DataFrame:
    """
    Returns the service date frame for bearing pressure check
    """
    df = df_service.copy()

    df["Axial_Total"] = -df["FZ"] + pt
    df["MX_Total"] = df["FY"] * (h1 + h2) + df["MX"]
    df["MY_Total"] = df["FX"] * (h1 + h2) + df["MY"]
    df["P/A"] = df["Axial_Total"] / (lx * ly)
    df["MX/Sx"] = df["MX_Total"].abs() / sx
    df["MY/Sy"] = df["MY_Total"].abs() / sy
    df[["P1", "P2", "P3", "P4"]] = df.apply(pressure_at_corners, axis=1)
    df["Pmin"] = df[["P1", "P2", "P3", "P4"]].min(axis=1)
    df["Pmax"] = df[["P1", "P2", "P3", "P4"]].max(axis=1)
    df["Padm"] = df.apply(padm_from_index, axis=1, cs=cs, fa_cs=fa_cs)
    df["Results"] = np.where(df["Pmax"] <= df["Padm"], "OK", "NOT OK")

    return df