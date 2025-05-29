import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from io import StringIO

import pandas as pd


def clean_college_name(name):
    name = str(name).upper().strip()
    cleaned = ' '.join(name.split())
    return cleaned.strip()


# Load the CSV file
df = pd.read_csv('modified_tl.csv')

# Rename the columns if necessary to match your output format
# df = df.rename(columns={'Affiliation': 'CollegeName', 'Count': 'TotalRegistrations'})

# Ensure the columns you're interested in are in the correct order
df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)

# Create a new column for indexing starting from 1
df['Index'] = df.index + 1

# Reorder columns to match the output format
df = df[['Index', 'Affiliation', 'Count']]

# Rename columns to match output
df = df.rename(columns={'Index': '', 'Affiliation': 'CollegeName', 'Count': 'TotalRegistrations'})

# Convert the DataFrame to a string
tech_leads_data_csv_str = df.to_csv(index=False)


df2 = pd.read_csv('modified_ai.csv')

# Rename the columns if necessary to match your output format
# df = df.rename(columns={'Affiliation': 'CollegeName', 'Count': 'TotalRegistrations'})

# Ensure the columns you're interested in are in the correct order
df2 = df2.sort_values(by='Count', ascending=False).reset_index(drop=True)

# Create a new column for indexing starting from 1
df2['Index'] = df2.index + 1

# Reorder columns to match the output format
df2 = df2[['Index', 'Affiliation', 'Count']]

# Rename columns to match output
df2 = df2.rename(columns={'Index': '', 'Affiliation': 'CollegeName', 'Count': 'TotalRegistrations'})

# Convert the DataFrame to a string
ai_developers_data_csv_str = df2.to_csv(index=False)



@st.cache_data 
def load_and_process_data(tech_leads_csv, devs_csv):
    df_tech_leads = pd.read_csv(StringIO(tech_leads_csv), usecols=['CollegeName', 'TotalRegistrations'])
    df_devs = pd.read_csv(StringIO(devs_csv), usecols=['CollegeName', 'TotalRegistrations'])

    df_tech_leads.rename(columns={'CollegeName': 'College_Name', 'TotalRegistrations': 'Tech_Leads'}, inplace=True)
    df_devs.rename(columns={'CollegeName': 'College_Name', 'TotalRegistrations': 'Developers'}, inplace=True)

    df_tech_leads['Cleaned_Name'] = df_tech_leads['College_Name'].apply(clean_college_name)
    df_devs['Cleaned_Name'] = df_devs['College_Name'].apply(clean_college_name)

    df_tech_leads_agg = df_tech_leads.groupby('Cleaned_Name')['Tech_Leads'].sum().reset_index()
    df_devs_agg = df_devs.groupby('Cleaned_Name')['Developers'].sum().reset_index()

    df_merged = pd.merge(df_tech_leads_agg, df_devs_agg, on='Cleaned_Name', how='outer')
    df_merged.fillna(0, inplace=True)
    df_merged['Developers'] = df_merged['Developers'].astype(int)
    df_merged['Tech_Leads'] = df_merged['Tech_Leads'].astype(int)

    filter_out_list = [
        "NEXTWAVE", "NIAT", "VIT", "KLH", "KUWL", "IIITH", 
        "GOVERNMENT INSTITUTE OF ELECTRONICS", 
        "JNTUH-5 YEAR INTEGRATED MTECH SELF FINANCE", "COLLEGE",
        "ICFAI", "GD GOENKA UNIVERSITY", "AMRITA VISHWA VIDHYAPEETHAM"
    ]
    df_merged = df_merged[~df_merged['Cleaned_Name'].isin(filter_out_list)]
    df_merged = df_merged[(df_merged['Developers'] > 0) | (df_merged['Tech_Leads'] > 0)].copy().reset_index(drop=True)
    return df_merged

# MODIFIED: Function now returns the processed DataFrame along with the figure
def create_sankey_figure_and_get_df(df_merged_input):
    # Make a copy to avoid modifying the input DataFrame directly if it's cached
    df_merged_processed = df_merged_input.copy()

    IDEAL_DEVS_PER_TL = 20
    df_merged_processed['Link_Color'] = 'rgba(200, 200, 200, 0.7)'
    df_merged_processed['Imbalance_Status'] = 'Balanced or Other'

    for index, row in df_merged_processed.iterrows():
        devs, tls = row['Developers'], row['Tech_Leads']
        link_color_to_set, status_to_set = df_merged_processed.loc[index, 'Link_Color'], df_merged_processed.loc[index, 'Imbalance_Status']

        if devs > 0 and (tls == 0 or tls < np.ceil(devs / IDEAL_DEVS_PER_TL)):
            link_color_to_set = 'rgba(255,0,0,0.7)'
            status_to_set = 'Critically Needs Tech Leads (0 TLs)' if tls == 0 else 'Needs Tech Leads'
        elif tls > 0 and (devs == 0 or devs < tls * IDEAL_DEVS_PER_TL):
            link_color_to_set = 'rgba(0,0,255,0.7)'
            status_to_set = 'Critically Needs Dev Interns (0 Devs)' if devs == 0 else 'Needs Dev Interns'
        
        df_merged_processed.loc[index, ['Link_Color', 'Imbalance_Status']] = link_color_to_set, status_to_set

    dev_sorted_df = df_merged_processed.sort_values(by='Developers', ascending=False).reset_index(drop=True)
    tl_sorted_df = df_merged_processed.sort_values(by='Tech_Leads', ascending=False).reset_index(drop=True)

    sankey_node_display_labels, sankey_node_id_labels, sankey_node_colors_list = [], [], []
    sankey_node_x_coords, sankey_node_y_coords = [], []
    node_color_dev_side, node_color_tl_side = 'rgba(173,216,230,0.8)', 'rgba(255,223,186,0.8)'

    # Developer nodes
    for i, row_dev in dev_sorted_df.iterrows():
        name, dev_count = row_dev['Cleaned_Name'], row_dev['Developers']
        sankey_node_id_labels.append(f"{name}_Dev")
        sankey_node_display_labels.append(f"{name} ({dev_count} Dev Interns)")
        sankey_node_colors_list.append(node_color_dev_side)
        sankey_node_x_coords.append(0.01)
        sankey_node_y_coords.append(i / max(1, len(dev_sorted_df) - 1) if len(dev_sorted_df) > 1 else 0.5)

    # Tech lead nodes
    for i, row_tl in tl_sorted_df.iterrows():
        name, tl_count = row_tl['Cleaned_Name'], row_tl['Tech_Leads']
        sankey_node_id_labels.append(f"{name}_TL")
        sankey_node_display_labels.append(f"{name} ({tl_count} Tech Leads)")
        sankey_node_colors_list.append(node_color_tl_side)
        sankey_node_x_coords.append(0.99)
        sankey_node_y_coords.append(tl_sorted_df[tl_sorted_df['Cleaned_Name'] == name].index[0] / max(1, len(tl_sorted_df) - 1) if len(tl_sorted_df) > 1 else 0.5)

    sankey_node_dict = {label: i for i, label in enumerate(sankey_node_id_labels)}
    source_indices, target_indices, values, link_colors_for_sankey, link_hover_texts = [], [], [], [], []

    for _, row in df_merged_processed.iterrows():
        college_name, dev_count, tl_count = row['Cleaned_Name'], row['Developers'], row['Tech_Leads']
        link_color_from_df, status_text = row['Link_Color'], row['Imbalance_Status']
        source_node_id, target_node_id = f"{college_name}_Dev", f"{college_name}_TL"
        
        if source_node_id in sankey_node_dict and target_node_id in sankey_node_dict:
            source_indices.append(sankey_node_dict[source_node_id])
            target_indices.append(sankey_node_dict[target_node_id])
            values.append(max(1, dev_count + tl_count))
            link_colors_for_sankey.append(link_color_from_df)
            link_hover_texts.append(f"{college_name}<br>Dev Interns: {dev_count}<br>Tech Leads: {tl_count}<br>Status: {status_text}")

    if not source_indices: return None, df_merged_processed # Return None for fig, but still return df

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=15, thickness=15, line=dict(color="black", width=0.5),
                  label=sankey_node_display_labels, color=sankey_node_colors_list,
                  x=sankey_node_x_coords, y=sankey_node_y_coords,
                  customdata=sankey_node_display_labels, hovertemplate='<b>%{customdata}</b><extra></extra>',
                  align="justify" 
                  ),
        link=dict(source=source_indices, target=target_indices, value=values,
                  color=link_colors_for_sankey, label=link_hover_texts,
                  hovertemplate='<br>%{label}<extra></extra>'))])
    
    fig.update_layout(
        font_size=10,
        height=max(800, len(df_merged_processed) * 2 + 200), 
        margin=dict(l=480, r=480, t=50, b=50) 
    )
    return fig, df_merged_processed # Return both

# --- Streamlit App ---
st.set_page_config(layout="wide") 

st.title("College Intern Imbalance: AI Dev Interns vs. Tech Leads")

st.markdown("""
**Ideal Ratio: 1 Tech Lead : 20 AI Dev Interns**

**Link Colors:**
- <span style='color:red; font-weight:bold;'>Red Link:</span> College primarily needs Tech Leads.
- <span style='color:blue; font-weight:bold;'>Blue Link:</span> College primarily needs AI Dev Interns.

Numbers on nodes: (Dev Interns) = AI Dev Intern count, (Tech Leads) = Tech Lead count.
""", unsafe_allow_html=True)


df_colleges_initial = load_and_process_data(tech_leads_data_csv_str, ai_developers_data_csv_str)

if df_colleges_initial.empty:
    st.warning("No data available after processing and filtering. Sankey diagram cannot be generated.")
else:
    # Generate the Sankey figure and get the DataFrame with imbalance status
    sankey_fig, df_colleges_with_status = create_sankey_figure_and_get_df(df_colleges_initial) 

    if sankey_fig:
        st.plotly_chart(sankey_fig, use_container_width=True)
        
        with st.expander("View Processed Data Table"):
            # Use df_colleges_with_status here as it contains the Link_Color and Imbalance_Status
            st.dataframe(df_colleges_with_status[['Cleaned_Name', 'Developers', 'Tech_Leads', 'Imbalance_Status', 'Link_Color']].sort_values(by='Developers', ascending=False))
    else:
        st.error("Could not generate the Sankey diagram.")