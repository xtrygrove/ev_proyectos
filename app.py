# ============================================
# app.py - Streamlit App (VSCode)
# Evaluación de Proyectos + Sensibilidad + PDF
# ============================================

# --------------------------------------------
# Imports
# --------------------------------------------
import io
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")  # backend headless (ok para Cloud y VSCode)
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from engine import (
    ProjectInputs,
    evaluate_project,
    scenarios,
    monte_carlo,
    sensitivity_npv_vs_discount_rate,
    sensitivity_npv_vs_cf_multiplier,
)

# --------------------------------------------
# Helpers de formato
# --------------------------------------------
def fmt_money(x, currency):
    if x is None:
        return "-"
    return f"{currency} {float(x):,.0f}".replace(",", ".")

def fmt_pct(x):
    if x is None:
        return "-"
    return f"{float(x)*100:.2f}%"

def fmt_years(x):
    if x is None:
        return "-"
    return f"{float(x):.2f}"

# --------------------------------------------
# PDF en memoria (para descarga)
# --------------------------------------------
def build_pdf_bytes(project: ProjectInputs, scen: dict, mc: dict | None) -> bytes:
    styles = getSampleStyleSheet()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    story = []
    story.append(Paragraph("<b>Reporte Ejecutivo – Evaluación de Proyecto</b>", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Proyecto:</b> {project.name}", styles["Normal"]))
    story.append(Paragraph(f"<b>CAPEX:</b> {fmt_money(project.initial_investment, project.currency)}", styles["Normal"]))
    story.append(Paragraph(f"<b>WACC:</b> {fmt_pct(project.discount_rate)}", styles["Normal"]))
    story.append(Paragraph(f"<b>Horizonte:</b> {len(project.cash_flows)} años", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Escenarios</b>", styles["Heading2"]))
    table_data = [["Escenario", "Decisión", "VAN", "TIR", "Payback"]]
    for k, v in scen.items():
        table_data.append([
            k,
            v["Decisión"],
            fmt_money(v["VAN"], project.currency),
            fmt_pct(v["TIR"]),
            fmt_years(v["Payback"]),
        ])

    tbl = Table(table_data, colWidths=[80, 80, 120, 80, 80])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    if mc is not None:
        story.append(Paragraph("<b>Riesgo (Monte Carlo)</b>", styles["Heading2"]))
        story.append(Paragraph(f"VAN Medio: {fmt_money(mc['VAN Medio'], project.currency)}", styles["Normal"]))
        story.append(Paragraph(
            f"P05 / P50 / P95: {fmt_money(mc['P05'], project.currency)} / "
            f"{fmt_money(mc['P50'], project.currency)} / {fmt_money(mc['P95'], project.currency)}",
            styles["Normal"],
        ))
        story.append(Paragraph(f"Prob(VAN>0): {mc['Prob VAN > 0']*100:.1f}%", styles["Normal"]))
        story.append(Spacer(1, 10))

    story.append(Paragraph(
        "Nota: este reporte es un insumo para toma de decisiones y no constituye ejecución ni intermediación financiera.",
        styles["Normal"]
    ))

    doc.build(story)
    return buffer.getvalue()


# --------------------------------------------
# UI Streamlit
# --------------------------------------------
st.set_page_config(page_title="Evaluación de Proyectos (MVP)", layout="wide")
st.title("Evaluación de Proyectos – CFO Digital (MVP)")

with st.sidebar:
    st.header("Inputs del Proyecto")

    name = st.text_input("Nombre del proyecto", value="Nueva Planta Industrial")
    currency = st.selectbox("Moneda", options=["CLP", "USD", "UF"], index=0)

    capex = st.number_input("CAPEX inicial", min_value=0.0, value=300_000_000.0, step=1_000_000.0)
    wacc_pct = st.number_input("WACC (%)", min_value=0.0, value=14.0, step=0.1)
    wacc = wacc_pct / 100.0
    terminal_value = st.number_input("Valor terminal (opcional)", min_value=0.0, value=0.0, step=10_000_000.0)

    st.subheader("Flujos (FCF) por año")
    years = st.slider("Horizonte (años)", min_value=3, max_value=15, value=5)

    default_cfs = [70_000_000, 85_000_000, 95_000_000, 100_000_000, 105_000_000]
    default_cfs = (default_cfs + [default_cfs[-1]] * max(0, years - len(default_cfs)))[:years]

    cf_df = pd.DataFrame({"Año": list(range(1, years + 1)), "FCF": default_cfs})
    cf_df = st.data_editor(cf_df, use_container_width=True, num_rows="fixed")

    st.divider()
    run_mc = st.checkbox("Incluir Monte Carlo", value=True)
    mc_sims = st.slider("Simulaciones", 500, 10000, 3000, 500) if run_mc else 0
    sigma_cf = st.slider("Volatilidad FCF (sigma)", 0.01, 0.50, 0.15, 0.01) if run_mc else 0.0
    sigma_r = st.slider("Volatilidad WACC (sigma)", 0.001, 0.05, 0.01, 0.001) if run_mc else 0.0

    st.divider()
    compute = st.button("Calcular")


# --------------------------------------------
# Cálculo y visualización
# --------------------------------------------
if compute:
    cash_flows = cf_df["FCF"].astype(float).tolist()

    project = ProjectInputs(
        name=name,
        initial_investment=float(capex),
        cash_flows=cash_flows,
        discount_rate=float(wacc),
        terminal_value=float(terminal_value),
        currency=currency
    )

    # ---- Resultados base y escenarios ----
    base = evaluate_project(project)
    scen = scenarios(project)
    scen_df = pd.DataFrame(scen).T

    st.subheader("Resultado Base (Decisión)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decisión", base["Decisión"])
    c2.metric("VAN", fmt_money(base["VAN"], currency))
    c3.metric("TIR", fmt_pct(base["TIR"]))
    c4.metric("Payback", fmt_years(base["Payback"]))

    st.subheader("Escenarios")
    st.dataframe(scen_df, use_container_width=True)

    # ---- Monte Carlo ----
    mc = None
    if run_mc:
        mc = monte_carlo(project, n=mc_sims, sigma_cf=sigma_cf, sigma_r=sigma_r)
        st.subheader("Riesgo (Monte Carlo)")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("VAN Medio", fmt_money(mc["VAN Medio"], currency))
        r2.metric("P05", fmt_money(mc["P05"], currency))
        r3.metric("P50", fmt_money(mc["P50"], currency))
        r4.metric("Prob(VAN>0)", f"{mc['Prob VAN > 0']*100:.1f}%")

    # --------------------------------------------
    # Sensibilidad (gráficos sin sobreingeniería)
    # --------------------------------------------
    st.subheader("Sensibilidad (VAN)")
    tab1, tab2 = st.tabs(["VAN vs WACC", "VAN vs Nivel de Flujos (FCF)"])

    # --- Tab 1: VAN vs WACC ---
    with tab1:
        st.caption("Curva VAN vs WACC para visualizar el punto de quiebre (VAN = 0).")
        a, b, c = st.columns(3)
        wacc_min = a.number_input("WACC mínimo (%)", min_value=0.0, value=max(0.0, wacc_pct - 6.0), step=0.5)
        wacc_max = b.number_input("WACC máximo (%)", min_value=0.0, value=wacc_pct + 6.0, step=0.5)
        wacc_step = c.number_input("Paso (%)", min_value=0.1, value=0.5, step=0.1)

        if wacc_max <= wacc_min:
            st.warning("WACC máximo debe ser mayor al mínimo.")
        else:
            rates = np.arange(wacc_min, wacc_max + 1e-9, wacc_step) / 100.0
            sens_r = sensitivity_npv_vs_discount_rate(project, rates.tolist())
            df_r = pd.DataFrame(sens_r, columns=["WACC", "VAN"])
            df_r["WACC_%"] = df_r["WACC"] * 100

            fig = plt.figure()
            plt.plot(df_r["WACC_%"], df_r["VAN"])
            plt.axhline(0)
            plt.axvline(wacc_pct)
            plt.xlabel("WACC (%)")
            plt.ylabel(f"VAN ({currency})")
            st.pyplot(fig)

            with st.expander("Ver tabla"):
                st.dataframe(df_r[["WACC_%", "VAN"]], use_container_width=True)

    # --- Tab 2: VAN vs Multiplicador FCF ---
    with tab2:
        st.caption("Evalúa cómo cambia el VAN si los flujos suben/bajan (demanda, margen, eficiencia).")
        a, b, c = st.columns(3)
        m_min = a.number_input("Multiplicador mínimo", min_value=0.1, value=0.70, step=0.05)
        m_max = b.number_input("Multiplicador máximo", min_value=0.1, value=1.30, step=0.05)
        m_step = c.number_input("Paso", min_value=0.01, value=0.05, step=0.01)

        if m_max <= m_min:
            st.warning("Multiplicador máximo debe ser mayor al mínimo.")
        else:
            mults = np.arange(m_min, m_max + 1e-9, m_step)
            sens_m = sensitivity_npv_vs_cf_multiplier(project, mults.tolist())
            df_m = pd.DataFrame(sens_m, columns=["Multiplicador_FCF", "VAN"])

            fig = plt.figure()
            plt.plot(df_m["Multiplicador_FCF"], df_m["VAN"])
            plt.axhline(0)
            plt.axvline(1.0)
            plt.xlabel("Multiplicador de FCF")
            plt.ylabel(f"VAN ({currency})")
            st.pyplot(fig)

            with st.expander("Ver tabla"):
                st.dataframe(df_m, use_container_width=True)

    # --------------------------------------------
    # PDF descargable
    # --------------------------------------------
    st.subheader("Reporte PDF")
    pdf_bytes = build_pdf_bytes(project, scen, mc)
    st.download_button(
        label="Descargar reporte PDF",
        data=pdf_bytes,
        file_name="reporte_evaluacion_proyecto.pdf",
        mime="application/pdf"
    )

else:
    st.info("Configura los inputs en la barra lateral y presiona **Calcular**.")
