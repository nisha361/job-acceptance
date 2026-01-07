import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("HR_Job_Placement_Dataset.csv")
df["status"] = df["status"].map({"Placed":1,"Not Placed":0})

st.set_page_config(page_title="Job Acceptance Prediction", layout="wide")

st.title("🎯 Job Acceptance Prediction Dashboard")

# KPIs
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Candidates", len(df))
col2.metric("Placement Rate (%)", round(df["status"].mean()*100,2))
col3.metric("Avg Skills Match", round(df["skills_match_percentage"].mean(),2))
col4.metric("Offer Dropout Rate (%)", round((1-df["status"].mean())*100,2))

# Placement Chart
st.subheader("Placement Distribution")
fig, ax = plt.subplots()
df["status"].value_counts().plot(kind="bar", ax=ax)
st.pyplot(fig)

# Skills Match
st.subheader("Skills Match vs Acceptance")
fig2, ax2 = plt.subplots()
df.boxplot(column="skills_match_percentage", by="status", ax=ax2)
st.pyplot(fig2)

# Company Tier
st.subheader("Company Tier vs Acceptance")
fig3, ax3 = plt.subplots()
df.groupby("company_tier")["status"].mean().plot(kind="bar", ax=ax3)
st.pyplot(fig3)
