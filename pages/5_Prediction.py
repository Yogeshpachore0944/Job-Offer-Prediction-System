
import pandas as pd
import streamlit as st
import plotly.express as px

from utils.data_loader import (
    load_dataset,
    load_encoders,
    load_feature_names,
    load_model,
    load_named_model,
    load_pipeline,
    load_scaler,
    fix_sklearn_compatibility,
)
from utils.prediction import hr_recommendation
from utils.page_state import activate_page

activate_page("5_Prediction")

st.set_page_config(page_title="Prediction", layout="wide")

st.markdown(
    """
    <style>
    .main { background: #f5f7fb; }
    .block-container { padding: 2rem 3rem; }
    .title { font-size: 38px; font-weight: bold; color: #0E4C92; margin-bottom: 5px; }
    .subtitle { color: #555; font-size: 16px; margin-bottom: 20px; }
    .template-box { background: #eef4fc; border: 1px solid #c7dcfa; padding: 12px 18px; border-radius: 8px; margin-bottom: 15px; }
    .metric-badge { padding: 4px 10px; border-radius: 5px; font-weight: bold; }
    .footer { text-align: center; color: gray; margin-top: 35px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_all_resources():
    pipe = load_pipeline()
    best_m = load_model()
    encs = load_encoders()
    feats = load_feature_names()
    sc = load_scaler()
    models_dict = {
        "Production Pipeline (Tuned XGBoost - Recommended)": ("pipeline", pipe),
        "XGBoost Classifier": ("xgboost.pkl", load_named_model("xgboost.pkl")),
        "Random Forest Classifier": ("random_forest.pkl", load_named_model("random_forest.pkl")),
        "Decision Tree Classifier": ("decision_tree.pkl", load_named_model("decision_tree.pkl")),
        "Logistic Regression": ("logistic.pkl", load_named_model("logistic.pkl")),
    }
    return pipe, best_m, encs, feats, sc, models_dict


pipeline, best_model, encoders, feature_names, scaler, models_dict = load_all_resources()
pipe = pipeline

st.markdown("<div class='title'>Job Offer Acceptance Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Predict whether a candidate is likely to accept or decline a job offer using calibrated ML models.</div>",
    unsafe_allow_html=True,
)

# preset candidate profiles
accept_demo = {
    "age": 30,
    "gender": "Male",
    "country": "USA",
    "years_experience": 6.0,
    "expected_salary_usd": 85000,
    "offered_salary_usd": 100000,
    "remote_preference": "Hybrid",
    "relocation_willingness": "Yes",
    "company_size": "Enterprise",
    "company_rating": 4.6,
}

reject_demo = {
    "age": 25,
    "gender": "Female",
    "country": "India",
    "years_experience": 2.0,
    "expected_salary_usd": 85000,
    "offered_salary_usd": 65000,
    "remote_preference": "Onsite",
    "relocation_willingness": "No",
    "company_size": "Startup",
    "company_rating": 2.3,
}

default_candidate = {
    "age": 30,
    "gender": "Male",
    "country": "USA",
    "years_experience": 6.0,
    "expected_salary_usd": 85000,
    "offered_salary_usd": 100000,
    "remote_preference": "Hybrid",
    "relocation_willingness": "Yes",
    "company_size": "Enterprise",
    "company_rating": 4.6,
}


def apply_candidate_data(data):
    st.session_state["pred_age"] = int(data["age"])
    st.session_state["pred_gender"] = str(data["gender"])
    st.session_state["pred_country"] = str(data["country"])
    st.session_state["pred_experience"] = float(data["years_experience"])
    st.session_state["pred_expected_salary"] = int(data["expected_salary_usd"])
    st.session_state["pred_offered_salary"] = int(data["offered_salary_usd"])
    st.session_state["pred_remote"] = str(data["remote_preference"])
    st.session_state["pred_relocation"] = str(data["relocation_willingness"])
    st.session_state["pred_company_size"] = str(data["company_size"])
    st.session_state["pred_company_rating"] = float(data["company_rating"])


# initialize default session state if not set
if "pred_age" not in st.session_state:
    apply_candidate_data(accept_demo)
    st.session_state["prediction_demo"] = "Sample Candidate-A (High Acceptance Profile)"

with st.sidebar:
    st.image("assets/logo.png", width=500)
    st.title("Navigation")
    st.write("Prediction Module")
    st.markdown("---")
    st.write("### Model Options")
    selected_model_label = st.selectbox(
        "Choose Prediction Algorithm",
        list(models_dict.keys()),
        index=0,
        key="selected_model_option",
    )
    st.markdown("---")
    st.write("### Active Features (14)")
    st.caption("- Candidate Age & Years Experience")
    st.caption("- Expected & Offered Salary ($)")
    st.caption("- Salary Offer Gap & Hike %")
    st.caption("- Experience-to-Age Ratio")
    st.caption("- Work Arrangement & Relocation")
    st.caption("- Company Size & Company Rating")

# mode selector: single candidate vs bulk / batch prediction

tab_single, tab_batch = st.tabs([
    " Single Candidate Prediction",
    " Bulk / Batch Prediction (CSV Upload)"
])

# tab 1: single candidate prediction
with tab_single:
    st.markdown("### Profile Presets & Quick Fill")
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.button("Sample Candidate-A (High Acceptance)", use_container_width=True, type="secondary"):
            apply_candidate_data(accept_demo)
            st.session_state["prediction_demo"] = "Sample Candidate-A (High Acceptance Profile)"
            st.rerun()

    with btn_col2:
        if st.button("Sample Candidate-B (Offer-at-Risk)", use_container_width=True, type="secondary"):
            apply_candidate_data(reject_demo)
            st.session_state["prediction_demo"] = "Sample Candidate-B (Offer-at-Risk Profile)"
            st.rerun()

    with btn_col3:
        if st.button("Reset to Default (Market Standard)", use_container_width=True, type="secondary"):
            apply_candidate_data(default_candidate)
            st.session_state["prediction_demo"] = "Custom Input"
            st.rerun()

    def on_template_change():
        selected = st.session_state.get("prediction_demo")
        if selected == "Sample Candidate-A (High Acceptance Profile)":
            apply_candidate_data(accept_demo)
        elif selected == "Sample Candidate-B (Offer-at-Risk Profile)":
            apply_candidate_data(reject_demo)
        elif selected == "Sample Candidate-C (Market Standard Profile)":
            apply_candidate_data(default_candidate)

    template_options = [
        "Sample Candidate-A (High Acceptance Profile)",
        "Sample Candidate-B (Offer-at-Risk Profile)",
        "Sample Candidate-C (Market Standard Profile)",
        "Custom Input",
    ]

    st.selectbox(
        "Or Choose Candidate Profile Template Dropdown",
        template_options,
        key="prediction_demo",
        on_change=on_template_change,
    )

    gender_options = ["Male", "Female", "Other"]
    country_options = ["USA", "India", "Singapore", "UK", "Germany", "Canada", "Japan", "France", "Australia"]
    remote_options = ["Hybrid", "Remote", "Onsite"]
    reloc_options = ["Yes", "No"]
    size_options = ["Enterprise", "Medium", "Small", "Startup"]

    # form inputs
    st.markdown("### candidate background")
    bg_col1, bg_col2, bg_col3, bg_col4 = st.columns(4)

    with bg_col1:
        age = st.number_input("Age", min_value=18, max_value=65, step=1, key="pred_age")

    with bg_col2:
        gender = st.selectbox("Gender", gender_options, key="pred_gender")

    with bg_col3:
        country = st.selectbox("Country", country_options, key="pred_country")

    with bg_col4:
        experience = st.number_input("Years of Experience", min_value=0.0, max_value=35.0, step=0.5, key="pred_experience")

    st.markdown("### compensation and salary alignment")
    comp_col1, comp_col2, comp_col3 = st.columns(3)

    with comp_col1:
        expected_salary = st.number_input("Expected Salary ($)", min_value=20000, max_value=300000, step=5000, key="pred_expected_salary")

    with comp_col2:
        offered_salary = st.number_input("Offered Salary ($)", min_value=20000, max_value=350000, step=5000, key="pred_offered_salary")

    with comp_col3:
        salary_gap = offered_salary - expected_salary
        hike_pct = (salary_gap / expected_salary * 100) if expected_salary > 0 else 0
        st.metric(
            "Offer Difference vs Expectation",
            f"${salary_gap:+,}",
            f"{hike_pct:+.1f}%",
            delta_color="normal" if salary_gap >= 0 else "inverse"
        )

    st.markdown("### workplace and company conditions")
    work_col1, work_col2, work_col3, work_col4 = st.columns(4)

    with work_col1:
        remote_pref = st.selectbox("Work Arrangement", remote_options, key="pred_remote")

    with work_col2:
        relocation = st.selectbox("Relocation Willingness", reloc_options, key="pred_relocation")

    with work_col3:
        company_size = st.selectbox("Company Size", size_options, key="pred_company_size")

    with work_col4:
        company_rating = st.slider("Company Rating", min_value=1.0, max_value=5.0, step=0.1, key="pred_company_rating")

    st.markdown("---")

    predict = st.button("Predict Offer Acceptance", type="primary", use_container_width=True)

    if predict:
        salary_offer_gap = offered_salary - expected_salary
        offer_ratio = offered_salary / expected_salary if expected_salary > 0 else 1.0
        hike_percentage = (salary_offer_gap / expected_salary * 100) if expected_salary > 0 else 0.0
        exp_age_ratio = experience / age if age > 0 else 0.0

        raw_input_df = pd.DataFrame([{
            "age": int(age),
            "gender": str(gender),
            "country": str(country),
            "years_experience": float(experience),
            "expected_salary_usd": int(expected_salary),
            "offered_salary_usd": int(offered_salary),
            "remote_preference": str(remote_pref),
            "relocation_willingness": str(relocation),
            "company_size": str(company_size),
            "company_rating": float(company_rating),
            "salary_offer_gap_usd": int(salary_offer_gap),
            "offer_to_expected_salary_ratio": float(offer_ratio),
            "salary_hike_pct": float(hike_percentage),
            "exp_to_age_ratio": float(exp_age_ratio),
        }])

        model_kind, model_obj = models_dict[selected_model_label]

        try:
            with st.spinner(f"Scoring with {selected_model_label}..."):
                fix_sklearn_compatibility(model_obj)
                if model_kind == "pipeline":
                    pred = int(model_obj.predict(raw_input_df)[0])
                    probs = model_obj.predict_proba(raw_input_df)[0]
                else:
                    # standalone models preprocessed features
                    preprocessor = pipeline.named_steps["preprocessor"]
                    fix_sklearn_compatibility(preprocessor)
                    X_proc = preprocessor.transform(raw_input_df)
                    pred = int(model_obj.predict(X_proc)[0])
                    probs = model_obj.predict_proba(X_proc)[0]

                reject_prob = float(probs[0] * 100)
                accept_prob = float(probs[1] * 100)
                label = "Accept" if pred == 1 else "Reject"
                confidence = accept_prob if pred == 1 else reject_prob
                recommendation = hr_recommendation(pred, accept_prob)

            st.session_state.candidate = {
                "age": int(age),
                "gender": str(gender),
                "country": str(country),
                "years_experience": float(experience),
                "expected_salary_usd": float(expected_salary),
                "offered_salary_usd": float(offered_salary),
                "salary_offer_gap_usd": float(salary_offer_gap),
                "salary_hike_pct": float(hike_percentage),
                "company_rating": float(company_rating),
                "remote_preference": str(remote_pref),
                "relocation_willingness": str(relocation),
                "company_size": str(company_size),
                "Model_Used": str(selected_model_label),
                "Prediction": int(pred),
                "Prediction_Label": str(label),
                "Acceptance_Probability": float(accept_prob),
                "Rejection_Probability": float(reject_prob),
                "Prediction_Confidence": float(confidence),
                "HR_Recommendation": str(recommendation),
            }

            st.success(f"Prediction generated successfully using **{selected_model_label}**!")
            st.subheader("Prediction Result")
            result_col, accept_col, reject_col = st.columns(3)
            if label == "Accept":
                result_col.success(f"### ACCEPT ({accept_prob:.1f}%)")
            else:
                result_col.error(f"### REJECT ({reject_prob:.1f}%)")
            accept_col.metric("Acceptance Probability", f"{accept_prob:.2f}%")
            reject_col.metric("Rejection Probability", f"{reject_prob:.2f}%")

            st.subheader("Confidence Gauge")
            st.progress(float(confidence) / 100)
            st.write(f"Algorithm Confidence: **{confidence:.2f}%**")

            st.subheader("HR Advisory Recommendation")
            st.info(recommendation)

            # multi model comparison on candidate
            with st.expander("Compare Predictions Across All 5 Model Algorithms on This Candidate", expanded=True):
                comparison_rows = []
                preprocessor = pipeline.named_steps["preprocessor"]
                fix_sklearn_compatibility(preprocessor)
                X_proc_comp = preprocessor.transform(raw_input_df)

                for m_lbl, (m_type, m_inst) in models_dict.items():
                    fix_sklearn_compatibility(m_inst)
                    if m_type == "pipeline":
                        m_p = int(m_inst.predict(raw_input_df)[0])
                        m_prob = m_inst.predict_proba(raw_input_df)[0]
                    else:
                        m_p = int(m_inst.predict(X_proc_comp)[0])
                        m_prob = m_inst.predict_proba(X_proc_comp)[0]

                    comparison_rows.append({
                        "Algorithm": m_lbl,
                        "Verdict": "ACCEPT" if m_p == 1 else "REJECT",
                        "Acceptance Probability (%)": round(float(m_prob[1] * 100), 2),
                        "Rejection Probability (%)": round(float(m_prob[0] * 100), 2),
                    })

                comp_df = pd.DataFrame(comparison_rows)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)

# tab 2: bulk / batch prediction (csv upload)
with tab_batch:
    st.markdown("### Bulk Candidate Offer Acceptance Scoring")
    st.write(
        "Upload a CSV file with multiple candidate offers or generate a random sample "
        "batch to evaluate acceptance probabilities, identify offer-at-risk candidates, "
        "and export the enriched report."
    )

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        # predefined downloadable template
        template_records = [
            {
                "candidate_id": "CAND_SAMPLE_01",
                "age": 29,
                "gender": "Female",
                "country": "USA",
                "years_experience": 5.0,
                "expected_salary_usd": 110000,
                "offered_salary_usd": 125000,
                "remote_preference": "Hybrid",
                "relocation_willingness": "Yes",
                "company_size": "Large",
                "company_rating": 4.5,
            },
            {
                "candidate_id": "CAND_SAMPLE_02",
                "age": 26,
                "gender": "Male",
                "country": "India",
                "years_experience": 2.5,
                "expected_salary_usd": 95000,
                "offered_salary_usd": 75000,
                "remote_preference": "Remote",
                "relocation_willingness": "No",
                "company_size": "Small",
                "company_rating": 2.9,
            },
            {
                "candidate_id": "CAND_SAMPLE_03",
                "age": 34,
                "gender": "Male",
                "country": "Germany",
                "years_experience": 8.0,
                "expected_salary_usd": 130000,
                "offered_salary_usd": 140000,
                "remote_preference": "Onsite",
                "relocation_willingness": "Yes",
                "company_size": "Enterprise",
                "company_rating": 4.1,
            },
        ]
        template_df = pd.DataFrame(template_records)
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="candidate_offers_template.csv",
            mime="text/csv",
            use_container_width=True,
            type="secondary",
        )

    with action_col2:
        if st.button("🎲 Load 30 Random Candidates from Dataset", use_container_width=True, type="secondary"):
            sample_df = load_dataset().sample(30, random_state=42).reset_index(drop=True)
            st.session_state["batch_uploaded_data"] = sample_df
            st.session_state["batch_source_name"] = "Sample Batch (30 Candidates from Benchmark Dataset)"
            st.session_state["batch_results_data"] = None
            st.rerun()

    uploaded_csv = st.file_uploader(
        "Upload Your Candidates CSV File",
        type=["csv"],
        help="Upload a CSV containing: age, gender, country, years_experience, expected_salary_usd, offered_salary_usd, remote_preference, relocation_willingness, company_size, company_rating.",
    )

    if uploaded_csv is not None:
        try:
            user_batch_df = pd.read_csv(uploaded_csv)
            st.session_state["batch_uploaded_data"] = user_batch_df
            st.session_state["batch_source_name"] = f"Uploaded CSV ({uploaded_csv.name})"
            st.session_state["batch_results_data"] = None
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

    active_batch = st.session_state.get("batch_uploaded_data")

    if active_batch is not None:
        st.success(f"Loaded: **{st.session_state.get('batch_source_name', 'Candidates Batch')}** ({len(active_batch):,} candidate records)")

        required_columns = [
            "age",
            "gender",
            "country",
            "years_experience",
            "expected_salary_usd",
            "offered_salary_usd",
            "remote_preference",
            "relocation_willingness",
            "company_size",
            "company_rating",
        ]
        missing_fields = [c for c in required_columns if c not in active_batch.columns]

        if missing_fields:
            st.error(f"Missing required columns in your CSV: {missing_fields}")
            st.info("Please ensure the CSV includes all required fields. You can download the sample template above as a reference.")
        else:
            with st.expander("Preview Raw Input Data", expanded=False):
                st.dataframe(active_batch.head(10), use_container_width=True)

            run_bulk = st.button(
                f"Run Bulk Scoring with {selected_model_label}",
                type="primary",
                use_container_width=True,
            )

            # run bulk prediction
            if run_bulk:
                with st.spinner(f"Scoring {len(active_batch):,} candidates with {selected_model_label}..."):
                    proc_df = active_batch.copy()

                    # compute engineered features if not already present
                    if "salary_offer_gap_usd" not in proc_df.columns:
                        proc_df["salary_offer_gap_usd"] = proc_df["offered_salary_usd"] - proc_df["expected_salary_usd"]
                    if "offer_to_expected_salary_ratio" not in proc_df.columns:
                        proc_df["offer_to_expected_salary_ratio"] = proc_df["offered_salary_usd"] / proc_df["expected_salary_usd"].replace(0, 1)
                    if "salary_hike_pct" not in proc_df.columns:
                        proc_df["salary_hike_pct"] = (proc_df["salary_offer_gap_usd"] / proc_df["expected_salary_usd"].replace(0, 1)) * 100
                    if "exp_to_age_ratio" not in proc_df.columns:
                        proc_df["exp_to_age_ratio"] = proc_df["years_experience"] / proc_df["age"].replace(0, 1)

                    model_kind, model_obj = models_dict[selected_model_label]
                    fix_sklearn_compatibility(model_obj)

                    if model_kind == "pipeline":
                        b_preds = model_obj.predict(proc_df)
                        b_probs = model_obj.predict_proba(proc_df)
                    else:
                        preprocessor = pipeline.named_steps["preprocessor"]
                        fix_sklearn_compatibility(preprocessor)
                        X_proc_bulk = preprocessor.transform(proc_df)
                        b_preds = model_obj.predict(X_proc_bulk)
                        b_probs = model_obj.predict_proba(X_proc_bulk)

                    proc_df["Predicted_Verdict"] = ["Accept" if p == 1 else "Reject" for p in b_preds]
                    proc_df["Acceptance_Probability_%"] = (b_probs[:, 1] * 100).round(2)
                    proc_df["Rejection_Probability_%"] = (b_probs[:, 0] * 100).round(2)
                    proc_df["Offer_Risk"] = [
                        "Low Risk" if prob >= 70 else ("Moderate Risk" if prob >= 50 else "High Risk")
                        for prob in proc_df["Acceptance_Probability_%"]
                    ]
                    proc_df["HR_Recommendation"] = [
                        "Safe offer. Proceed to onboarding." if p == 1 else "Offer at risk. Re-evaluate compensation or flexibility."
                        for p in b_preds
                    ]

                    st.session_state["batch_results_data"] = proc_df

            res_df = st.session_state.get("batch_results_data")

            if res_df is not None:
                st.markdown("---")
                st.subheader("Executive Batch Overview")

                tot = len(res_df)
                acc = int((res_df["Predicted_Verdict"] == "Accept").sum())
                rej = int((res_df["Predicted_Verdict"] == "Reject").sum())
                acc_rate = (acc / tot * 100) if tot > 0 else 0
                avg_prob = float(res_df["Acceptance_Probability_%"].mean())

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Candidates Scored", f"{tot:,}")
                m2.metric("Predicted Accepted Offers", f"{acc:,}", f"{acc_rate:.1f}% Acceptance Rate")
                m3.metric("Offers At Risk / Rejection", f"{rej:,}", f"-{((rej/tot)*100):.1f}% Rejection Rate", delta_color="inverse")
                m4.metric("Avg Acceptance Probability", f"{avg_prob:.1f}%")

                # visualizations
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    fig_pie = px.pie(
                        names=["Accept", "Reject"],
                        values=[acc, rej],
                        title="Predicted Acceptance Split",
                        color=["Accept", "Reject"],
                        color_discrete_map={"Accept": "#2ECC71", "Reject": "#E74C3C"},
                        hole=0.45,
                    )
                    fig_pie.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=320)
                    st.plotly_chart(fig_pie, use_container_width=True)

                with v_col2:
                    fig_hist = px.histogram(
                        res_df,
                        x="Acceptance_Probability_%",
                        nbins=20,
                        title="Acceptance Probability Distribution",
                        color="Predicted_Verdict",
                        color_discrete_map={"Accept": "#2ECC71", "Reject": "#E74C3C"},
                    )
                    fig_hist.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=320)
                    st.plotly_chart(fig_hist, use_container_width=True)

                # segment by work arrangement
                if "remote_preference" in res_df.columns:
                    grouped_work = res_df.groupby(["remote_preference", "Predicted_Verdict"]).size().reset_index(name="Count")
                    fig_bar = px.bar(
                        grouped_work,
                        x="remote_preference",
                        y="Count",
                        color="Predicted_Verdict",
                        barmode="group",
                        title="Offer Outcomes by Work Arrangement",
                        color_discrete_map={"Accept": "#2ECC71", "Reject": "#E74C3C"},
                    )
                    fig_bar.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=320)
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.subheader("Filter & Inspect Scored Candidates")
                flt1, flt2 = st.columns(2)
                with flt1:
                    verdict_filter = st.selectbox("Filter by Verdict", ["All", "Accept", "Reject"], key="b_v_filter")
                with flt2:
                    risk_filter = st.selectbox("Filter by Risk Tier", ["All", "Low Risk", "Moderate Risk", "High Risk"], key="b_r_filter")

                view_df = res_df.copy()
                if verdict_filter != "All":
                    view_df = view_df[view_df["Predicted_Verdict"] == verdict_filter]
                if risk_filter != "All":
                    view_df = view_df[view_df["Offer_Risk"] == risk_filter]

                # organize columns for clear executive review
                priority_cols = [
                    "candidate_id",
                    "Predicted_Verdict",
                    "Acceptance_Probability_%",
                    "Offer_Risk",
                    "offered_salary_usd",
                    "expected_salary_usd",
                    "salary_offer_gap_usd",
                    "salary_hike_pct",
                    "remote_preference",
                    "company_size",
                    "HR_Recommendation",
                ]
                existing_prio = [c for c in priority_cols if c in view_df.columns]
                remaining_cols = [c for c in view_df.columns if c not in existing_prio]
                view_df = view_df[existing_prio + remaining_cols]

                st.dataframe(view_df, use_container_width=True, hide_index=True)

                # download results button
                csv_export = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Download Full Scored Batch Report ({len(res_df):,} Candidates - CSV)",
                    data=csv_export,
                    file_name="job_offer_bulk_predictions_report.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                )

st.markdown("---")
st.markdown(
    """
    <div class='footer'>
        <h4>OfferSense AI</h4>
        <p>Machine Learning Based Job Offer Acceptance Prediction System</p>
    </div>
    """,
    unsafe_allow_html=True,
)
