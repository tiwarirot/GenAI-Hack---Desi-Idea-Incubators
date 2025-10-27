# ---------- Cross page (REPLACE existing Cross block with this) ----------
if menu == "Cross":
    st.header("Strategic Cross-Process Optimization")
    # Try to load a combined historical dataframe (df_all) from a few common filenames
    df_all = None
    for fname in ("df_all.csv", "all_scores.csv", "scores.csv", "combined_scores.csv"):
        df_all = load_csv(fname)
        if df_all is not None:
            break

    # If not found, attempt to synthesize df_all from already-loaded per-process CSVs
    if df_all is None:
        # try to build from individual dataframes if present
        # these variables may exist earlier in the script: df_rg, df_cl, df_q, df_af, df_cross
        df_rg = load_csv("raw_grinding.csv")
        df_cl = load_csv("clinker.csv")
        df_q = load_csv("quality.csv")
        df_af = load_csv("altfuel.csv")
        df_cross = load_csv("cross.csv")

        # If each per-process file contains a score column with expected names, build df_all
        cols_expected = ["raw_score", "clinker_score", "quality_score", "altfuel_score", "cross_score"]
        can_build = True
        # We'll create an empty dict of lists and try to fill using means if column missing
        synth = {}
        try:
            # For each expected score, if there's a direct column in one of the dfs use that column; else attempt heuristics
            # Raw score: try df_rg["raw_score"] or fallback to df_rg['predicted_energy'] etc.
            if df_rg is not None:
                if "raw_score" in df_rg.columns:
                    synth["raw_score"] = df_rg["raw_score"].dropna().values
                else:
                    # heuristic: try 'predicted_energy' or 'predicted' or use numeric column mean
                    for cand in ("predicted_energy", "predicted", "energy", "raw_material_variability"):
                        if cand in df_rg.columns:
                            synth["raw_score"] = df_rg[cand].dropna().values
                            break
            if df_cl is not None:
                if "clinker_score" in df_cl.columns:
                    synth["clinker_score"] = df_cl["clinker_score"].dropna().values
                else:
                    for cand in ("predicted_energy", "kiln_temp", "oxygen_level"):
                        if cand in df_cl.columns:
                            synth["clinker_score"] = df_cl[cand].dropna().values
                            break
            if df_q is not None:
                if "quality_score" in df_q.columns:
                    synth["quality_score"] = df_q["quality_score"].dropna().values
                else:
                    for cand in ("compressive_strength", "strength", "blaine", "blain"):
                        if cand in df_q.columns:
                            synth["quality_score"] = df_q[cand].dropna().values
                            break
            if df_af is not None:
                if "altfuel_score" in df_af.columns:
                    synth["altfuel_score"] = df_af["altfuel_score"].dropna().values
                else:
                    for cand in ("tsr", "fuel_calorific"):
                        if cand in df_af.columns:
                            synth["altfuel_score"] = df_af[cand].dropna().values
                            break
            if df_cross is not None:
                if "cross_score" in df_cross.columns:
                    synth["cross_score"] = df_cross["cross_score"].dropna().values
                else:
                    for cand in ("predicted_energy", "co2"):
                        if cand in df_cross.columns:
                            synth["cross_score"] = df_cross[cand].dropna().values
                            break
        except Exception:
            can_build = False

        # If we were able to synthesize at least the four process arrays and a cross array, make df_all
        required_keys = ["raw_score", "clinker_score", "quality_score", "altfuel_score", "cross_score"]
        if all(k in synth and len(synth[k]) > 0 for k in required_keys):
            # Build DataFrame by aligning lengths (pad/truncate to shortest length for simplicity)
            min_len = min(len(synth[k]) for k in required_keys)
            df_all = pd.DataFrame({
                "raw_score": synth["raw_score"][:min_len],
                "clinker_score": synth["clinker_score"][:min_len],
                "quality_score": synth["quality_score"][:min_len],
                "altfuel_score": synth["altfuel_score"][:min_len],
                "cross_score": synth["cross_score"][:min_len],
            })
        else:
            df_all = None

    if df_all is None:
        st.error(
            "Combined historical score dataframe not found. Please provide a combined CSV (e.g. 'df_all.csv') with columns: "
            "'raw_score', 'clinker_score', 'quality_score', 'altfuel_score', 'cross_score' in the data/ folder."
        )
    else:
        # compute historical means
        raw_mean = float(df_all["raw_score"].mean())
        cl_mean = float(df_all["clinker_score"].mean())
        q_mean = float(df_all["quality_score"].mean())
        af_mean = float(df_all["altfuel_score"].mean())
        cross_mean = float(df_all["cross_score"].mean())

        # Use only the four processes as sources feeding into the single target (Strategic Cross-Process Efficiency)
        source_names = ["Raw & Grinding", "Clinker", "Quality", "Alt Fuel"]
        source_values = np.array([raw_mean, cl_mean, q_mean, af_mean], dtype=float)

        # Normalize contributions to percentage of total (Q2-B -> normalize to % contribution)
        total = float(source_values.sum()) if source_values.sum() != 0 else 1.0
        contributions = (source_values / total) * 100.0  # percentages sum to 100

        # Format labels with percentage appended (you chose label style B)
        labels = [
            f"{source_names[i]} ({contributions[i]:.1f}%)" for i in range(len(source_names))
        ]
        sink_label = f"Strategic Cross-Process Efficiency ({cross_mean:.2f})"

        # Build Sankey nodes and links for Plotly
        # Nodes: all sources followed by single sink
        nodes = source_names + [sink_label]
        # indices
        source_indices = list(range(len(source_names)))
        target_index = len(nodes) - 1

        # For Sankey, each source → sink link has value = contributions[i]
        link_sources = source_indices
        link_targets = [target_index] * len(source_indices)
        link_values = contributions.tolist()

        # Create fig
        sankey_fig = go.Figure(
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                    color=["#4c78a8"] * len(source_names) + ["#0b6efd"],
                ),
                link=dict(
                    source=link_sources,
                    target=link_targets,
                    value=link_values,
                    label=[f"{v:.1f}%" for v in link_values],
                    color=["rgba(12,140,199,0.6)"] * len(link_values),
                ),
            )
        )
        sankey_fig.update_layout(title_text="Contribution to Strategic Cross-Process Efficiency (normalized %)", font_size=12, height=520, template="plotly_white")
        st.plotly_chart(sankey_fig, use_container_width=True)

        st.markdown("**Notes:** contributions are normalized to percentage of total historical mean across processes. The node on the right shows the current cross efficiency value (mean).")
        st.markdown("---")

        # Optional: show small table-like summary (compact)
        st.markdown("#### Source contributions (historical means → normalized %)")
        cols = st.columns(4)
        for i, name in enumerate(source_names):
            with cols[i]:
                st.metric(name, f"{source_values[i]:.2f}", delta=f"{contributions[i]:.1f}%")
