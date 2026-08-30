import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import linregress
from scipy.interpolate import interp1d
from math import sqrt, degrees, asin
import io
import plotly.io as pio

st.set_page_config(page_title="Contact Angle Analysis", layout="wide")
st.title("Contact Angle Analysis")
st.markdown("---")

FIG_H = 320

# ── Helper functions ──────────────────────────────────────────────────────────
def load_dynamic_ca_data(file_bytes, cycle=1):
    raw_data = pd.read_csv(io.BytesIO(file_bytes), sep='\t', skiprows=3, header=None)
    raw_data = raw_data.apply(pd.to_numeric, errors='coerce')
    start = (cycle - 1) * 4
    data = pd.DataFrame({
        'pos1': raw_data[start],  'wt1': raw_data[start + 1],
        'pos2': raw_data[start + 2], 'wt2': raw_data[start + 3],
    })
    set1 = data[['pos1', 'wt1']].dropna().reset_index(drop=True)
    set2 = data[['pos2', 'wt2']].dropna().reset_index(drop=True)
    return data, set1, set2

def find_touch_point(set1, tolerance=0.005):
    for i in range(len(set1) - 1):
        if abs(set1.iloc[i]['wt1']) <= tolerance and abs(set1.iloc[i+1]['wt1']) > tolerance:
            return set1.iloc[i]['pos1']
    return None

def nondimensionalize(set1, set2, x1, g, p, sft, l_cap):
    set1 = set1.copy(); set2 = set2.copy()
    set1['x1_nd'] = (set1['pos1'] - x1) / (sqrt(2) * l_cap)
    set1['y1_nd'] = 1000 * set1['wt1'] * g / (p * sft)
    set2['x2_nd'] = (set2['pos2'] - x1) / (sqrt(2) * l_cap)
    set2['y2_nd'] = 1000 * set2['wt2'] * g / (p * sft)
    return set1, set2

def find_intersections(x_data, y_data, slope, intercept, tol):
    diff = np.abs(y_data - (slope * x_data + intercept))
    indices = np.where(diff < tol)[0]
    return [{'x': x_data[i], 'y': y_data[i]} for i in indices]

def calculate_ca(x, case):
    val = 1 - x**2
    if not -1 <= val <= 1:
        return None
    return 180 - degrees(asin(val)) if case in ('ACA > 90°', 'RCA > 90°') else degrees(asin(val))

def make_fig(title, xlabel='Non-dimensionalized Position', ylabel='Non-dimensionalized Force'):
    fig = go.Figure()
    fig.update_layout(height=FIG_H, title=title, xaxis_title=xlabel, yaxis_title=ylabel,
                      legend=dict(font=dict(size=9)), margin=dict(l=50, r=20, t=40, b=40),
                      hovermode='closest')
    return fig

# ── File upload (above columns) ───────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload DynamicCA .txt file", type="txt")

if uploaded_file:
    base_name = uploaded_file.name.replace('.txt', '')
    file_bytes = uploaded_file.read()
    cycle = st.radio("Select cycle", [1, 2, 3, 4], horizontal=True)
    data, set1, set2 = load_dynamic_ca_data(file_bytes, cycle=cycle)
    st.success(f"Cycle {cycle} loaded — advancing: {len(set1)} rows, receding: {len(set2)} rows")

    st.markdown("---")

    # ── Two-column layout ─────────────────────────────────────────────────────
    left, right = st.columns(2)

    # ═════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN — Data Prep + Parameters
    # ═════════════════════════════════════════════════════════════════════════
    with left:
        st.header("① Data Preparation")

        # All cycles overview
        st.subheader("All Cycles Overview")
        colors_adv = ['black', 'blue', 'green', 'red']
        colors_rec = ['gray', 'steelblue', 'limegreen', 'salmon']
        fig = make_fig('Raw Data: All Cycles', 'Position (mm)', 'Weight (g)')
        for c in range(1, 5):
            _, s1, s2 = load_dynamic_ca_data(file_bytes, cycle=c)
            fig.add_trace(go.Scatter(x=s1['pos1'], y=s1['wt1'], mode='lines',
                                     name=f'Cycle {c} Adv', line=dict(color=colors_adv[c-1])))
            fig.add_trace(go.Scatter(x=s2['pos2'], y=s2['wt2'], mode='lines',
                                     name=f'Cycle {c} Rec', line=dict(color=colors_rec[c-1], dash='dash')))
        st.plotly_chart(fig, width='stretch')
        html_bytes = fig.to_html().encode('utf-8')
        st.download_button(label="Download Overview", data=html_bytes,
                   file_name=f"{base_name}_all_cycles.html", mime="text/html")

        # Selected cycle
        st.subheader(f"Selected Cycle {cycle}")
        fig = make_fig('Raw Data: Position vs Weight', 'Position (mm)', 'Weight (g)')
        fig.add_trace(go.Scatter(x=set1['pos1'], y=set1['wt1'], mode='lines',
                                 name='Advancing', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=set2['pos2'], y=set2['wt2'], mode='lines',
                                 name='Receding', line=dict(color='red')))
        st.plotly_chart(fig, width='stretch')

        # Touch point
        st.subheader("Touch Point Detection")
        touch_tol = st.number_input("Touch point tolerance (g)", value=0.005, format="%.4f")
        x1 = find_touch_point(set1, tolerance=touch_tol)
        if x1 is None:
            st.error("Touch point not found — try adjusting tolerance.")
            st.stop()
        st.success(f"Touch point: x1 = {x1:.4f} mm")
        fig = make_fig('Touch Point Detection', 'Position (mm)', 'Weight (g)')
        fig.add_trace(go.Scatter(x=set1['pos1'], y=set1['wt1'], mode='lines+markers',
                                 name='Advancing', line=dict(color='blue'), marker=dict(size=3)))
        fig.add_vline(x=x1, line_dash='dash', line_color='red',
                      annotation_text=f'x1={x1:.4f}', annotation_position='top right')
        fig.add_hline(y=0, line_color='black', opacity=0.3)
        st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        st.header("② Parameters & Non-dimensionalized Data")

        c1, c2 = st.columns(2)
        with c1:
            sft = st.number_input("Surface tension (mN/m)", value=110.0)
            den = st.number_input("Density (g/cm³)", value=0.775, format="%.4f")
            g   = st.number_input("Gravity (m/s²)", value=9.81, format="%.3f")
        with c2:
            p    = st.number_input("Perimeter (mm)", value=32.0)
            area = st.number_input("Area (mm²)", value=15.0)

        l_cap    = sqrt(sft / (den * g))
        theo_max = sft * p / 10000
        max_wt2  = set2['wt2'].max()
        st.write(f"**Capillary length:** {l_cap:.4f} mm")
        st.write(f"**Theo. max:** {theo_max:.4f} g  |  **Measured max:** {max_wt2:.4f} g")
        if theo_max < max_wt2:
            st.warning("WARNING: receding CA could be smaller than 0, please check")

        set1_nd, set2_nd = nondimensionalize(set1, set2, x1, g, p, sft, l_cap)
        fig = make_fig('Non-dimensionalized Data')
        fig.add_trace(go.Scatter(x=set1_nd['x1_nd'], y=set1_nd['y1_nd'], mode='lines+markers',
                                 name='Advancing', line=dict(color='blue'), marker=dict(size=3)))
        fig.add_trace(go.Scatter(x=set2_nd['x2_nd'], y=set2_nd['y2_nd'], mode='lines+markers',
                                 name='Receding', line=dict(color='magenta'), marker=dict(size=3)))
        fig.update_xaxes(range=[0, 2.5])
        st.plotly_chart(fig, width='stretch')

    # ═════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN — ACA + RCA & Summary
    # ═════════════════════════════════════════════════════════════════════════
    with right:
        st.header("③ Advancing Contact Angle (ACA)")

        x_adv = set1_nd['x1_nd'].values
        y_adv = set1_nd['y1_nd'].values

        st.subheader("Linear Fit — Advancing")
        c1, c2 = st.columns(2)
        with c1:
            x_min_aca = st.number_input("X min (ACA fit)", value=float(np.min(x_adv)),
                                        format="%.4f", key='xmin_aca')
        with c2:
            x_max_aca = st.number_input("X max (ACA fit)", value=float(np.max(x_adv)),
                                        format="%.4f", key='xmax_aca')

        mask_aca = (x_adv >= x_min_aca) & (x_adv <= x_max_aca)
        slope, intercept = None, None
        if np.sum(mask_aca) >= 2:
            slope, intercept, r_value, _, _ = linregress(x_adv[mask_aca], y_adv[mask_aca])
            x_line = np.linspace(np.min(x_adv), np.max(x_adv), 200)
            fig = make_fig('ACA Linear Fit')
            fig.add_trace(go.Scatter(x=x_adv, y=y_adv, mode='markers', name='Data',
                                     marker=dict(color='blue', size=4)))
            fig.add_trace(go.Scatter(x=x_adv[mask_aca], y=y_adv[mask_aca], mode='markers',
                                     name='Fit window', marker=dict(color='green', size=6)))
            fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode='lines',
                                     name=f'y={slope:.4f}x+{intercept:.4f}',
                                     line=dict(color='red', dash='dash')))
            fig.add_vline(x=x_min_aca, line_dash='dash', line_color='red', opacity=0.4)
            fig.add_vline(x=x_max_aca, line_dash='dash', line_color='red', opacity=0.4)
            st.plotly_chart(fig, width='stretch')
            
            st.write(f"**ACA fit:** y = {slope:.6f}x + {intercept:.6f}  R² = {r_value**2:.6f}")
        else:
            st.warning("Need at least 2 points in the fit window.")

        if slope is not None:
            st.subheader("ACA Intersection Point (x2)")
            tol_aca = st.number_input("Tolerance", value=0.01, format="%.4f", key='tol_aca')
            intersections_aca = find_intersections(x_adv, y_adv, slope, intercept, tol_aca)

            if intersections_aca:
                options = [f"Point {i+1}: x = {pt['x']:.4f}" for i, pt in enumerate(intersections_aca)]
                options.append("Manual entry")
                selected = st.selectbox("Select intersection point", options, key='sel_aca')
                x2 = st.number_input("Enter x2 manually", value=0.0, format="%.6f", key='manual_x2') \
                    if selected == "Manual entry" else intersections_aca[options.index(selected)]['x']
            else:
                st.warning("No intersections found — try increasing tolerance.")
                x2 = st.number_input("Enter x2 manually", value=0.0, format="%.6f", key='manual_x2_only')

            fig = make_fig('ACA Intersection Points')
            x_fit = np.linspace(np.min(x_adv), np.max(x_adv), 200)
            fig.add_trace(go.Scatter(x=x_adv, y=y_adv, mode='lines+markers', name='Data',
                                     line=dict(color='blue'), marker=dict(size=3)))
            fig.add_trace(go.Scatter(x=x_fit, y=slope * x_fit + intercept, mode='lines',
                                     name='Fit', line=dict(color='red', dash='dash')))
            for i, pt in enumerate(intersections_aca):
                if abs(pt['x'] - x2) < 1e-9:
                    fig.add_trace(go.Scatter(x=[pt['x']], y=[pt['y']], mode='markers',
                                             name='Selected', marker=dict(color='green', size=12,
                                             line=dict(color='black', width=2))))
                else:
                    fig.add_trace(go.Scatter(x=[pt['x']], y=[pt['y']], mode='markers',
                                             marker=dict(color='cyan', size=8), showlegend=False))
            st.plotly_chart(fig, width='stretch')

            st.subheader("Calculate ACA")
            case_aca = st.radio("Case", ['ACA > 90°', 'ACA < 90°'], key='case_aca', horizontal=True)
            ca_aca = calculate_ca(x2, case_aca)
            if ca_aca is not None:
                st.success(f"x2 = {x2:.6f} → **ACA = {ca_aca:.2f}°**")
            else:
                st.error(f"x2 = {x2:.4f} is out of valid range.")
                ca_aca = None

        st.markdown("---")
        st.header("④ Receding Contact Angle (RCA) & Summary")

        x_rec = set2_nd['x2_nd'].values
        y_rec = set2_nd['y2_nd'].values

        # Buoyancy check
        st.subheader("Buoyancy Line Check")
        buoyancy_slope = -np.sqrt(2) * area / (p * l_cap)
        buoyancy_tol   = st.number_input("Buoyancy tolerance", value=0.005, format="%.4f", key='btol')
        x_max_rec  = np.max(x_rec)
        range_mask = x_rec >= (x_max_rec - 1.0)
        x_range    = x_rec[range_mask]
        y_range    = y_rec[range_mask]
        close_pts  = np.where(np.abs(y_range - buoyancy_slope * x_range) < buoyancy_tol)[0]

        fig = make_fig('Set 2 vs Buoyancy Line')
        x_line_b = np.linspace(np.min(x_rec), np.max(x_rec), 200)
        fig.add_trace(go.Scatter(x=x_rec, y=y_rec, mode='lines+markers', name='Receding',
                                 line=dict(color='magenta'), marker=dict(size=3)))
        fig.add_trace(go.Scatter(x=x_line_b, y=buoyancy_slope * x_line_b, mode='lines',
                                 name=f'Buoyancy: y={buoyancy_slope:.4f}x',
                                 line=dict(color='green', dash='dash')))
        fig.add_vrect(x0=x_max_rec - 1.0, x1=x_max_rec, fillcolor='yellow',
                      opacity=0.2, layer='below', line_width=0)
        if len(close_pts) > 0:
            fig.add_trace(go.Scatter(x=x_range[close_pts], y=y_range[close_pts], mode='markers',
                                     name='Intersections', marker=dict(color='red', size=8)))
            x_90 = x_range[close_pts[0]]
            st.info(f"x_90 = {x_90:.6f}")
        else:
            x_90 = None
            st.info("No intersection found in search region.")
        st.plotly_chart(fig, width='stretch')

        # RCA linear fit
        st.subheader("Linear Fit — Receding")
        c1, c2 = st.columns(2)
        with c1:
            x_min_rca = st.number_input("X min (RCA fit)", value=float(np.min(x_rec)),
                                        format="%.4f", key='xmin_rca')
        with c2:
            x_max_rca = st.number_input("X max (RCA fit)", value=float(np.max(x_rec)),
                                        format="%.4f", key='xmax_rca')

        mask_rca = (x_rec >= x_min_rca) & (x_rec <= x_max_rca)
        slope_rca, intercept_rca = None, None
        if np.sum(mask_rca) >= 2:
            slope_rca, intercept_rca, r_value_rca, _, _ = linregress(x_rec[mask_rca], y_rec[mask_rca])
            x_line_r = np.linspace(np.min(x_rec), np.max(x_rec), 200)
            fig = make_fig('RCA Linear Fit')
            fig.add_trace(go.Scatter(x=x_rec, y=y_rec, mode='markers', name='Data',
                                     marker=dict(color='magenta', size=4)))
            fig.add_trace(go.Scatter(x=x_rec[mask_rca], y=y_rec[mask_rca], mode='markers',
                                     name='Fit window', marker=dict(color='green', size=6)))
            fig.add_trace(go.Scatter(x=x_line_r, y=slope_rca * x_line_r + intercept_rca, mode='lines',
                                     name=f'y={slope_rca:.4f}x+{intercept_rca:.4f}',
                                     line=dict(color='red', dash='dash')))
            fig.add_vline(x=x_min_rca, line_dash='dash', line_color='red', opacity=0.4)
            fig.add_vline(x=x_max_rca, line_dash='dash', line_color='red', opacity=0.4)
            st.plotly_chart(fig, width='stretch')
            st.write(f"**RCA fit:** y = {slope_rca:.6f}x + {intercept_rca:.6f}  R² = {r_value_rca**2:.6f}")
        else:
            st.warning("Need at least 2 points in the fit window.")

        if slope_rca is not None:
            st.subheader("RCA Intersection Point (x3)")
            tol_rca = st.number_input("Tolerance", value=0.01, format="%.4f", key='tol_rca')
            intersections_rca = find_intersections(x_rec, y_rec, slope_rca, intercept_rca, tol_rca)

            if intersections_rca:
                options_rca = [f"Point {i+1}: x = {pt['x']:.4f}" for i, pt in enumerate(intersections_rca)]
                options_rca.append("Manual entry")
                selected_rca = st.selectbox("Select intersection point", options_rca, key='sel_rca')
                x3 = st.number_input("Enter x3 manually", value=0.0, format="%.6f", key='manual_x3') \
                    if selected_rca == "Manual entry" else intersections_rca[options_rca.index(selected_rca)]['x']
            else:
                st.warning("No intersections found — try increasing tolerance.")
                x3 = st.number_input("Enter x3 manually", value=0.0, format="%.6f", key='manual_x3_only')

            fig = make_fig('RCA Intersection Points')
            x_fit_r = np.linspace(np.min(x_rec), np.max(x_rec), 200)
            fig.add_trace(go.Scatter(x=x_rec, y=y_rec, mode='lines+markers', name='Data',
                                     line=dict(color='magenta'), marker=dict(size=3)))
            fig.add_trace(go.Scatter(x=x_fit_r, y=slope_rca * x_fit_r + intercept_rca,
                                     mode='lines', name='Fit', line=dict(color='red', dash='dash')))
            for i, pt in enumerate(intersections_rca):
                if abs(pt['x'] - x3) < 1e-9:
                    fig.add_trace(go.Scatter(x=[pt['x']], y=[pt['y']], mode='markers',
                                             name='Selected', marker=dict(color='green', size=12,
                                             line=dict(color='black', width=2))))
                else:
                    fig.add_trace(go.Scatter(x=[pt['x']], y=[pt['y']], mode='markers',
                                             marker=dict(color='cyan', size=8), showlegend=False))
            st.plotly_chart(fig, width='stretch')

            st.subheader("Calculate RCA")
            case_rca = st.radio("Case", ['RCA > 90°', 'RCA < 90°', 'Special case'],
                                key='case_rca', horizontal=True)
            rca = None
            if case_rca == 'Special case':
                x_90_val = x_90 if x_90 is not None else st.number_input(
                    "x_90 not found — enter manually", value=0.0, format="%.6f")
                x4  = abs(x3 - x_90_val)
                val = 1 - x4**2
                if -1 <= val <= 1:
                    rca = degrees(asin(val))
                    st.success(f"x3={x3:.6f}, x_90={x_90_val:.6f}, x4={x4:.6f} → **RCA = {rca:.2f}°**")
                else:
                    st.error("x4 out of valid range for asin.")
            else:
                rca = calculate_ca(x3, case_rca)
                if rca is not None:
                    st.success(f"x3 = {x3:.6f} → **RCA = {rca:.2f}°**")

            # Summary
            if ca_aca is not None and rca is not None:
                slope_theoretical = -np.sqrt(2) * area / (p * l_cap)
                deviation = ((slope - slope_theoretical) / slope_theoretical) * 100

                st.markdown("---")
                st.subheader("Summary Plot")
                set1_s = set1_nd.sort_values('x1_nd')
                set2_s = set2_nd.sort_values('x2_nd')
                x_min_c  = max(set1_s['x1_nd'].min(), set2_s['x2_nd'].min())
                x_max_c  = min(set1_s['x1_nd'].max(), set2_s['x2_nd'].max())
                x_common = np.linspace(x_min_c + 1e-9, x_max_c - 1e-9, 1000)
                f_adv = interp1d(set1_s['x1_nd'].values, set1_s['y1_nd'].values,
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
                f_rec = interp1d(set2_s['x2_nd'].values, set2_s['y2_nd'].values,
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
                y_adv_i   = f_adv(x_common)
                y_rec_i   = f_rec(x_common)
                encloarea = np.trapezoid(np.abs(y_adv_i - y_rec_i), x_common)
                mckinley  = encloarea / (set1_nd['x1_nd'].max() - x2)

                x_fit_s = np.linspace(0, 2.5, 500)
                fig = make_fig('Contact Angle Analysis')
                fig.update_layout(height=420)
                fig.add_trace(go.Scatter(x=set1_nd['x1_nd'], y=set1_nd['y1_nd'],
                                         mode='lines+markers', name='Advancing',
                                         line=dict(color='blue'), marker=dict(size=3), opacity=0.6))
                fig.add_trace(go.Scatter(x=set2_nd['x2_nd'], y=set2_nd['y2_nd'],
                                         mode='lines+markers', name='Receding',
                                         line=dict(color='magenta'), marker=dict(size=3), opacity=0.6))
                fig.add_trace(go.Scatter(x=x_fit_s, y=slope * x_fit_s + intercept,
                                         mode='lines', name='ACA fit', line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=x_fit_s, y=slope_rca * x_fit_s + intercept_rca,
                                         mode='lines', name='RCA fit', line=dict(color='red', width=2)))
                fig.add_trace(go.Scatter(x=[x2], y=[slope * x2 + intercept], mode='markers',
                                         name=f'x2={x2:.4f} (ACA={ca_aca:.2f}°)',
                                         marker=dict(color='blue', size=12,
                                                     line=dict(color='black', width=2))))
                fig.add_trace(go.Scatter(x=[x3], y=[slope_rca * x3 + intercept_rca], mode='markers',
                                         name=f'x3={x3:.4f} (RCA={rca:.2f}°)',
                                         marker=dict(color='red', size=12,
                                                     line=dict(color='black', width=2))))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_common, x_common[::-1]]),
                    y=np.concatenate([y_adv_i, y_rec_i[::-1]]),
                    fill='toself', fillcolor='rgba(128,128,128,0.3)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Enclosed area={encloarea:.4f}'))
                fig.update_xaxes(range=[0, 2.5])
                st.plotly_chart(fig, width='stretch')

                st.subheader("Results")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("ACA", f"{ca_aca:.2f}°")
                    st.metric("RCA", f"{rca:.2f}°")
                with m2:
                    st.metric("Classic CAH", f"{ca_aca - rca:.2f}°")
                    st.metric("Slope deviation", f"{deviation:.2f}%")
                with m3:
                    st.metric("Enclosed Area", f"{encloarea:.4f}")
                    st.metric("McKinley Hysteresis", f"{mckinley:.4f}")

                st.markdown(f"""
| Parameter | Value |
|---|---|
| Capillary length | {l_cap:.4f} mm |
| Theoretical slope | {slope_theoretical:.6f} |
| Measured ACA slope | {slope:.6f} (deviation: {deviation:.2f}%) |
| ACA fit | y = {slope:.6f}x + {intercept:.6f} |
| x2 | {x2:.6f} |
| ACA | {ca_aca:.2f}° |
| RCA fit | y = {slope_rca:.6f}x + {intercept_rca:.6f} |
| x3 | {x3:.6f} |
| RCA | {rca:.2f}° |
| Classic CAH | {ca_aca - rca:.2f}° |
| Enclosed area | {encloarea:.6f} |
| McKinley hysteresis | {mckinley:.6f} |
""")
