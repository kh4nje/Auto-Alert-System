import pandas as pd
from google.colab import files
import numpy as np
from io import BytesIO

# Step 1: Upload the file
print("Upload your disease surveillance data file (Excel or CSV).")
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Step 2: Load and process the file
print("Processing your file...")
try:
    if filename.endswith('.xlsx'):
        df = pd.read_excel(filename)
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
        print("Excel file processed successfully. Now performing detection.")
    else:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        long_df = None
        for encoding in encodings:
            try:
                long_df = pd.read_csv(filename, encoding=encoding)
                print(f"Successfully read CSV with {encoding} encoding.")
                break
            except UnicodeDecodeError:
                print(f"Failed to read CSV with {encoding} encoding. Trying next encoding...")
        if long_df is None:
            raise ValueError("Unable to read CSV with any tested encoding. Please check the file for corruption or specify the correct encoding.")
        expected_cols = ['Facility_Name', 'Disease_Name', 'Epi Week Number', 'Number_Cases']
        if not all(col in long_df.columns for col in expected_cols):
            raise ValueError("CSV must contain columns: Facility_Name, Disease_Name, Epi Week Number, Number_Cases")
        orgunit_cols = ['orgunitlevel1', 'orgunitlevel2', 'orgunitlevel3', 'orgunitlevel4', 'orgunitlevel5', 'orgunitlevel6']
        for col in orgunit_cols:
            if col not in long_df.columns:
                long_df[col] = None
        long_df = long_df[['Facility_Name', 'orgunitlevel1', 'orgunitlevel2', 'orgunitlevel3', 'orgunitlevel4', 'orgunitlevel5', 'orgunitlevel6', 'Disease_Name', 'Epi Week Number', 'Number_Cases']]
        long_df['Number_Cases'] = long_df['Number_Cases'].fillna(0).astype(int)
        long_df = long_df.sort_values(by=['Facility_Name', 'Disease_Name', 'Epi Week Number'])
        print("Using uploaded CSV for detection.")
except Exception as e:
    print(f"Error processing file: {str(e)}. Please check the file format and content.")
    raise

# Step 3: Week selection
weeks = sorted(long_df['Epi Week Number'].unique())
if len(weeks) < 4:
    raise ValueError("Data must have at least 4 weeks for analysis.")
print(f"Available weeks: {weeks}")
current_week = int(input(f"Enter the latest week number (e.g., {max(weeks)}): "))
if current_week not in weeks:
    raise ValueError(f"Week {current_week} not found in the dataset.")

# Find the three preceding weeks, handling year-spanning cases
all_weeks = list(range(1, 53))  # Assume weeks 1-52 in a cycle
week_indices = {w: i for i, w in enumerate(all_weeks)}
current_idx = week_indices.get(current_week, -1)
if current_idx == -1:
    raise ValueError(f"Week {current_week} not in expected range (1-52).")
prev_indices = [(current_idx - i - 1) % 52 for i in range(3)]
prev_weeks = [all_weeks[idx] for idx in prev_indices[::-1]]  # Reverse for ascending order
available_prev_weeks = [w for w in prev_weeks if w in weeks]
if len(available_prev_weeks) != 3:
    raise ValueError(f"Not enough previous weeks available before Week {current_week}. Available: {available_prev_weeks}")
print(f"Comparing Week {current_week} against Weeks {available_prev_weeks}")

# Step 4: Anomaly detection
print("Analyzing data for outbreaks...")
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

# Step 5: Save and download alerts as Excel
alerts = grouped.apply(detect_alert, include_groups=False).dropna().tolist()
if alerts:
    alerts_df = pd.DataFrame(alerts)
    alerts_df = alerts_df[[
        'Facility', 'orgunitlevel1', 'orgunitlevel2', 'orgunitlevel3', 'orgunitlevel4', 'orgunitlevel5', 'orgunitlevel6', 'Disease',
        f'Week_{available_prev_weeks[0]}', f'Week_{available_prev_weeks[1]}', f'Week_{available_prev_weeks[2]}', f'Week_{current_week}',
        'Mean', 'Std', 'Threshold', 'Deviation'
    ]]
    print("Outbreak alerts found. Generating Excel file...")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        alerts_df.to_excel(writer, index=False, sheet_name='Alerts')
    output.seek(0)
    with open('alerts.xlsx', 'wb') as f:
        f.write(output.getvalue())
    files.download('alerts.xlsx')
    print(f"Alerts for Week {current_week} saved and downloaded as alerts.xlsx")
else:
    print("No outbreak alerts found for the selected week.")