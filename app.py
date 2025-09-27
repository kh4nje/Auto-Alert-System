import pandas as pd
import streamlit as st
import numpy as np
from io import BytesIO

# Streamlit app configuration
st.set_page_config(page_title="IDSRS Outbreak Detection", page_icon="🩺", layout="wide")

# Title and description
st.title("🩺 IDSRS Outbreak Detection App")
st.markdown("""
Welcome to the Integrated Disease Surveillance and Response System (IDSRS) Outbreak Detection App!  
This tool helps you identify disease outbreaks by analyzing weekly surveillance data.  
**Steps**:  
1. Upload your Excel (.xlsx) or CSV (.csv) file.  
2. Select the latest week to analyze.  
3. Download the Excel report with outbreak alerts (cases exceeding mean + 3 * standard deviation of the previous three weeks).  
The report includes organizational unit details, facility, disease, weekly case counts, and statistical metrics.
""")

# File uploader with guidance
st.subheader("Step 1: Upload Your Data")
st.markdown("Upload your disease surveillance data file. Ensure it includes columns like `periodname`, `orgunitlevel1`–`orgunitlevel6`, `organisationunitname`, and disease names.")
uploaded_file = st.file_uploader(
    "Choose a file",
    type=['xlsx', 'csv'],
    help="Supported formats: Excel (.xlsx) or CSV (.csv). The file should contain weekly disease case data."
)

if uploaded_file is not None:
    with st.spinner("Processing your file..."):
        # Step 1: Load the file
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
                df.columns = df.columns.str.strip()
                df['Epi Week Number'] = df['periodname'].str.extract(r'Week (\d+)', expand=False).astype(int)
                id_cols = ['periodname', 'orgunitlevel1', 'orgunitlevel2', 'orgunitlevel3', 'orgunitlevel4', 'orgunitlevel5', 'orgunitlevel6', 'organisationunitname', 'Epi Week Number']
                disease_cols = [col for col in df.columns if col not in id_cols]
                long_df = pd.melt(
                    df,
                    id_vars=['organisationunitname', 'orgunitlevel1', 'orgunitlevel2', 'orgunitlevel3', 'orgunitlevel4', 'orgunitlevel5', 'orgunitlevel6', 'Epi Week Number'],
                    value_vars=disease_cols,
                    var_name='Disease_Name',
                    value_name='Number_Cases'
                )
                long_df = long_df.rename(columns={'organisationunitname': 'Facility_Name'})
                long_df = long_df[['Facility_Name', 'orgunitlevel1', 'orgunitlevel2', 'orgunitlevel3', 'orgunitlevel4', 'orgunitlevel5', 'orgunitlevel6', 'Disease_Name', 'Epi Week Number', 'Number_Cases']]
                long_df['Number_Cases'] = long_df['Number_Cases'].fillna(0).astype(int)
                long_df = long_df.sort_values(by=['Facility_Name', 'Disease_Name', 'Epi Week Number'])
                st.success("✅ Excel file processed successfully!")
            else:
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                long_df = None
                for encoding in encodings:
                    try:
                        long_df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"✅ Successfully read CSV with {encoding} encoding.")
                        break
                    except UnicodeDecodeError:
                        st.warning(f"Failed to read CSV with {encoding} encoding. Trying next encoding...")
                if long_df is None:
                    st.error("❌ Unable to read CSV. Please ensure the file is not corrupted and uses a supported encoding (e.g., UTF-8).")
                    st.stop()
                uploaded_file.seek(0)  # Reset file pointer
                expected_cols = ['Facility_Name', 'Disease_Name', 'Epi Week Number', 'Number_Cases']
                if not all(col in long_df.columns for col in expected_cols):
                    st.error("❌ CSV must contain columns: Facility_Name, Disease_Name, Epi Week Number, Number_Cases")
                    st.stop()
                orgunit_cols = ['orgunitlevel1', 'orgunitlevel2', 'orgunitlevel3', 'orgunitlevel4', 'orgunitlevel5', 'orgunitlevel6']
                for col in orgunit_cols:
                    if col not in long_df.columns:
                        long_df[col] = None
                long_df = long_df[['Facility_Name', 'orgunitlevel1', 'orgunitlevel2', 'orgunitlevel3', 'orgunitlevel4', 'orgunitlevel5', 'orgunitlevel6', 'Disease_Name', 'Epi Week Number', 'Number_Cases']]
                long_df['Number_Cases'] = long_df['Number_Cases'].fillna(0).astype(int)
                long_df = long_df.sort_values(by=['Facility_Name', 'Disease_Name', 'Epi Week Number'])
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}. Please check the file format and content.")
            st.stop()

        # Display data preview
        st.subheader("Data Preview")
        st.markdown("Here’s a preview of your uploaded data (first 10 rows):")
        st.dataframe(long_df.head(10), use_container_width=True)

        # Step 2: Week selection
        weeks = sorted(long_df['Epi Week Number'].unique())
        if len(weeks) < 4:
            st.error("❌ Your data must contain at least 4 weeks for analysis. Please upload a file with more weeks.")
            st.stop()

        st.subheader("Step 2: Select the Latest Week")
        st.markdown("Choose the latest week to analyze for outbreaks (compares against the three previous weeks):")
        current_week = st.selectbox(
            "Latest Week",
            weeks,
            index=len(weeks)-1,
            help="Select the most recent week in your data. The app will compare it to the three previous weeks."
        )

        # Find the three preceding weeks, handling year-spanning cases
        all_weeks = list(range(1, 53))  # Assume weeks 1-52 in a cycle
        week_indices = {w: i for i, w in enumerate(all_weeks)}
        current_idx = week_indices.get(current_week, -1)
        if current_idx == -1:
            st.error(f"❌ Week {current_week} is not in the expected range (1–52). Please check your data.")
            st.stop()

        prev_indices = [(current_idx - i - 1) % 52 for i in range(3)]
        prev_weeks = [all_weeks[idx] for idx in prev_indices[::-1]]  # Reverse for ascending order
        available_prev_weeks = [w for w in prev_weeks if w in weeks]
        
        if len(available_prev_weeks) != 3:
            st.error(f"❌ Not enough previous weeks available before Week {current_week}. Available: {available_prev_weeks}. Please ensure the three prior weeks are in the data.")
            st.stop()

        st.info(f"Comparing Week {current_week} against Weeks {available_prev_weeks}")

        # Step 3: Anomaly detection
        with st.spinner("Analyzing data for outbreaks..."):
            grouped = long_df.groupby(['Facility_Name', 'Disease_Name'])

            def detect_alert(group):
                prev_cases = group[group['Epi Week Number'].isin(available_prev_weeks)]['Number_Cases'].values
                if len(prev_cases) != 3:
                    return None
                current_cases = group[group['Epi Week Number'] == current_week]['Number_Cases'].values
                if len(current_cases) != 1:
                    return None
                current_cases = current_cases[0]
                mean = np.mean(prev_cases)
                std = np.std(prev_cases, ddof=1)
                threshold = mean + 3 * std if std > 0 else mean
                if current_cases > threshold:
                    orgunit_values = group[group['Epi Week Number'] == current_week][['orgunitlevel1', 'orgunitlevel2', 'orgunitlevel3', 'orgunitlevel4', 'orgunitlevel5', 'orgunitlevel6']].iloc[0].to_dict()
                    return {
                        'Facility': group.name[0],
                        'orgunitlevel1': orgunit_values['orgunitlevel1'],
                        'orgunitlevel2': orgunit_values['orgunitlevel2'],
                        'orgunitlevel3': orgunit_values['orgunitlevel3'],
                        'orgunitlevel4': orgunit_values['orgunitlevel4'],
                        'orgunitlevel5': orgunit_values['orgunitlevel5'],
                        'orgunitlevel6': orgunit_values['orgunitlevel6'],
                        'Disease': group.name[1],
                        f'Week_{available_prev_weeks[0]}': prev_cases[0],
                        f'Week_{available_prev_weeks[1]}': prev_cases[1],
                        f'Week_{available_prev_weeks[2]}': prev_cases[2],
                        f'Week_{current_week}': current_cases,
                        'Mean': round(mean, 2),
                        'Std': round(std, 2),
                        'Threshold': round(threshold, 2),
                        'Deviation': round(current_cases - mean, 2)
                    }
                return None

            # Step 4: Generate and download alerts as Excel
            alerts = grouped.apply(detect_alert, include_groups=False).dropna().tolist()
            if alerts:
                alerts_df = pd.DataFrame(alerts)
                alerts_df = alerts_df[[
                    'Facility', 'orgunitlevel1', 'orgunitlevel2', 'orgunitlevel3', 'orgunitlevel4', 'orgunitlevel5', 'orgunitlevel6', 'Disease',
                    f'Week_{available_prev_weeks[0]}', f'Week_{available_prev_weeks[1]}', f'Week_{available_prev_weeks[2]}', f'Week_{current_week}',
                    'Mean', 'Std', 'Threshold', 'Deviation'
                ]]
                st.subheader("Step 3: Review and Download Results")
                st.markdown("Below is a preview of the outbreak alerts found in your data:")
                st.dataframe(alerts_df, use_container_width=True)
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    alerts_df.to_excel(writer, index=False, sheet_name='Alerts')
                output.seek(0)
                st.download_button(
                    label=f"📥 Download Alerts for Week {current_week}",
                    data=output,
                    file_name=f'alerts_week_{current_week}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='download-button',
                    help=f"Download the outbreak alerts for Week {current_week} as an Excel file."
                )
                st.success(f"✅ Outbreak alerts for Week {current_week} are ready! Click the button above to download alerts_week_{current_week}.xlsx.")
            else:
                st.warning("⚠️ No outbreak alerts found for the selected week. Try a different week or check your data.")