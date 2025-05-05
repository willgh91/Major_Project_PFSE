import pandas as pd
import numpy as np
import isolated_footings as i_f
import conversion_units as cu
import streamlit as st
import matplotlib.pyplot as plt

st.header("Allowable soil pressure for isolated footings")


#User inputs
st.sidebar.subheader("User Input")
st.sidebar.info(
    "ℹ️ **Important**: Before running the analysis, make sure that the selected load cases "
    "match those available in your ETABS Excel file.\n\n"
    "This app only works with kips and ft as units")
uploaded_file = st.sidebar.file_uploader("Upload an ETABS Excel File (.xlsx)", type=["xlsx"])
if st.sidebar.button("▶ Run Analysis"):
    st.session_state.run_analysis = True

with st.sidebar.expander("Soil Properties", expanded=True):
    d_soil_iv= st.number_input("Soil density (kg/m3)", min_value=0.0, format="%.3f", value=1600.00)
    cs_iv= st.number_input("Allowable soil bearing pressure (Ton/m2)", min_value=0.0, format="%.3f", value=20.00)
    fa_cs= st.number_input("Temporal loads amplification factor (seismic)", min_value=0.0, format="%.3f", value=1.333)


with st.sidebar.expander("Geometric Properties", expanded=True):
    lx_c_iv= st.number_input("Column dimmension along x-axis (m)", min_value=0.0, format="%.3f", value=0.80)
    ly_c_iv= st.number_input("Column dimmension along y-axis (m)", min_value=0.0, format="%.3f", value=0.80)
    lx_iv= st.number_input("Footing dimmension along x-axis (m)", min_value=0.0, format="%.3f", value=2.50)
    ly_iv= st.number_input("Footing dimmension along y-axis (m)", min_value=0.0, format="%.3f", value=2.50)
    h1_iv= st.number_input("Footing thickness (m)", min_value=0.0, format="%.3f", value=0.40)
    h2_iv= st.number_input("Soil depth above footing", min_value=0.0, format="%.3f", value=1.00)


with st.sidebar.expander("Material Properties", expanded=True):
    fc= st.number_input("Concrete (psi)", min_value=0.0, format="%.3f", value=4.00)
    fy= st.number_input("Reinforced steel (psi)", min_value=0.0, format="%.3f", value=60.00)
    d_concrete_iv= st.number_input("Concrete density (kg/m3)", min_value=0.0, format="%.3f", value=2400.00)

with st.sidebar.expander("Load Cases Selection", expanded=True):
    st.markdown("**Gravity Load Cases**")
    dead = st.checkbox("DEAD", value=True)
    live = st.checkbox("LIVE", value=True)
    roof = st.checkbox("ROOF", value=False)

    gravity_cases_used = []
    if dead: gravity_cases_used.append("DEAD")
    if live: gravity_cases_used.append("LIVE")
    if roof: gravity_cases_used.append("ROOF")

    st.markdown("---")
    st.markdown("**Lateral Load Cases**")
    qx = st.checkbox("QX", value=True)
    qy = st.checkbox("QY", value=True)
    qx_at = st.checkbox("QX+AT", value=False)
    qx_at_neg = st.checkbox("QX-AT", value=False)
    qy_at = st.checkbox("QY+AT", value=False)
    qy_at_neg = st.checkbox("QY-AT", value=False)

    lateral_cases_used = []
    if qx: lateral_cases_used.append("QX")
    if qy: lateral_cases_used.append("QY")
    if qx_at: lateral_cases_used.append("QX+AT")
    if qy_at: lateral_cases_used.append("QY+AT")
    if qx_at_neg: lateral_cases_used.append("QX-AT")
    if qy_at_neg: lateral_cases_used.append("QY-AT")


with st.sidebar.expander("Other load combination factors", expanded=True):
    llrf= st.number_input("Live load reduction factor", min_value=0.0, format="%.3f", value=1.00)
    scd= st.number_input("Scd", min_value=0.0, format="%.3f", value=0.00)
    kz= st.number_input("Kz", min_value=0.0, format="%.3f", value=1.20)

if "run_analysis" in st.session_state and st.session_state.run_analysis:
    #Unit conversions
    lx_c= cu.cvu_m_to_ft(lx_c_iv)
    ly_c= cu.cvu_m_to_ft(ly_c_iv)
    lx= cu.cvu_m_to_ft(lx_iv)
    ly= cu.cvu_m_to_ft(ly_iv)
    h1= cu.cvu_m_to_ft(h1_iv)
    h2= cu.cvu_m_to_ft(h2_iv)
    cs= cu.cvu_ton_m2_to_kip_ft2(cs_iv)
    d_soil= cu.cvu_kg_m3_to_kip_ft3(d_soil_iv)
    d_concrete= cu.cvu_kg_m3_to_kip_ft3(d_concrete_iv)


    #Internal calculations
    pt = i_f.self_weight_soil_and_concrete(lx,ly,h1,h2,d_soil,d_concrete)
    tlcu= gravity_cases_used + lateral_cases_used
    sx= i_f.secmod_about_x(ly,lx)
    sy= i_f.secmod_about_y(ly,lx)


    #Base_Service_DataFrame
    df_etabs= i_f.df_from_etabs_file_st(uploaded_file)
    df_filtered_cases= i_f.df_filter_load_cases(df_etabs, tlcu)
    service_combos= i_f.filtered_contextual_foundation_service_combos(llrf, scd, lateral_cases_used)
    df_service_base= i_f.base_df_service(service_combos, df_filtered_cases)


    #Service_DataFrame
    df_service=i_f.get_df_service(df_service_base, pt, h1, h2, lx, ly, sx, sy, cs, fa_cs)


    # Selector "Unique Name"
    unique_names = df_service.index.get_level_values("Unique Name").unique()
    selected_name = st.selectbox("Select a Unique Name (column node):", unique_names)

    # Filtrar el DataFrame para el nodo seleccionado
    df_filtered = df_service.loc[selected_name]

    #Show summary DataFrame
    cols_resumen = ["P1", "P2", "P3", "P4", "Pmin", "Pmax", "Padm", "Results"]
    df_summary = df_filtered[cols_resumen].reset_index(level="Load Combination")
    st.dataframe(df_summary.style.hide(axis="index"))

    # Generar un gráfico por cada combinación de carga
    for combinacion, presiones in df_filtered.iterrows():
        # Extraer presiones de las esquinas
        P1 = presiones["P1"]  # inferior izquierda
        P2 = presiones["P2"]  # superior izquierda
        P3 = presiones["P3"]  # superior derecha
        P4 = presiones["P4"]  # inferior derecha

        # Crear grilla
        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 2), np.linspace(0, 1, 2))

        # Interpolación bilineal
        Z = (
            P1 * (1 - grid_x) * (1 - grid_y) +  # (0, 0)
            P2 * (1 - grid_x) * grid_y     +    # (0, 1)
            P3 * grid_x * grid_y           +    # (1, 1)
            P4 * grid_x * (1 - grid_y)          # (1, 0)
        )

        # Crear figura
        fig, ax = plt.subplots()
        c = ax.contourf(grid_x, grid_y, Z, levels=10, cmap='viridis')
        fig.colorbar(c, ax=ax, label="Pressure (kip/ft2)")

        # Anotar esquinas correctamente
        ax.text(0, 0, f'P1\n{P1:.1f}', color='white', ha='left', va='bottom')   # Inferior izquierda
        ax.text(0, 1, f'P2\n{P2:.1f}', color='white', ha='left', va='top')      # Superior izquierda
        ax.text(1, 1, f'P3\n{P3:.1f}', color='white', ha='right', va='top')     # Superior derecha
        ax.text(1, 0, f'P4\n{P4:.1f}', color='white', ha='right', va='bottom')  # Inferior derecha

        ax.set_title(f"Pressure Map – {selected_name} – {combinacion}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        st.pyplot(fig)


