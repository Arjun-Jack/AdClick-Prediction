import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


# ----------------- Load all pickle files -----------------
with open(r"d:\Online_Ad_Click_Predicition\model\label_dy.pkl", "rb") as f:
    label_dy = pickle.load(f)

with open(r"d:\Online_Ad_Click_Predicition\model\label_device.pkl", "rb") as f:
    label_device = pickle.load(f)

with open(r"d:\Online_Ad_Click_Predicition\model\label_gender.pkl", "rb") as f:
    label_gender = pickle.load(f)

with open(r"d:\Online_Ad_Click_Predicition\model\label_interest.pkl", "rb") as f:
    label_interest = pickle.load(f)

with open(r"d:\Online_Ad_Click_Predicition\model\label_location.pkl", "rb") as f:
    label_location = pickle.load(f)

with open(r"d:\Online_Ad_Click_Predicition\model\label_time.pkl", "rb") as f:
    label_time = pickle.load(f)

with open(r"d:\Online_Ad_Click_Predicition\model\final_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
df = pd.read_csv("adsclicking.csv")

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="Ad Click Prediction", page_icon=":ad:", layout="wide")
st.title("📊 Ad Click Prediction")

# ----------------- User Inputs (Readable for User) -----------------
Age = st.number_input("Enter Age:", min_value=10, max_value=100, step=1)
Gender = st.selectbox("Select Gender:", label_gender.classes_)
Income = st.number_input("Enter Income:", min_value=0)
Location = st.selectbox("Select Location:", label_location.classes_)
Device = st.selectbox("Select Device:", label_device.classes_)
Interest_Category = st.selectbox("Select Interest Category:", label_interest.classes_)
Time_Spent_on_Site = st.number_input("Time Spent on Site (minutes):", min_value=0)
Number_of_Pages_Viewed = st.number_input("Pages Viewed:", min_value=1)
Click = st.selectbox("Clicked Before?", ["No", "Yes"])

# ----------------- Backend Encoding (Hidden from User) -----------------
if st.button("Predict my ad click : "):
    input_data = pd.DataFrame([[
        Age,
        label_gender.transform([Gender])[0],
        Income,
        label_location.transform([Location])[0],
        label_device.transform([Device])[0],
        label_interest.transform([Interest_Category])[0],
        Time_Spent_on_Site,
        Number_of_Pages_Viewed
    ]], columns=[
        'Age', 'Gender', 'Income', 'Location', 'Device',
        'Interest_Category', 'Time_Spent_on_Site', 'Number_of_Pages_Viewed'
    ])

    # Show input data (Readable form)
    st.subheader("📋 Your Input Data")
    user_friendly_data = pd.DataFrame([[
        Age, Gender, Income, Location, Device, Interest_Category,
        Time_Spent_on_Site, Number_of_Pages_Viewed
    ]], columns=[
        'Age', 'Gender', 'Income', 'Location', 'Device',
        'Interest_Category', 'Time_Spent_on_Site', 'Number_of_Pages_Viewed'
    ])
    st.dataframe(user_friendly_data)

    # Make prediction
    result = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    # Show prediction result
    if result == 1:
        st.success(f"✅ Will Click the Ad! (Confidence: {prob[1]*100:.2f}%)")
    else:
        st.error(f"❌ Will NOT Click the Ad. (Confidence: {prob[0]*100:.2f}%)")
    
    st.progress(int(prob[result]*100))

    # Show probability chart
    st.subheader("📊 Prediction Probability")
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Not Click", "Click"], prob, color=["red", "green"])
    for bar, p in zip(bars, prob):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{p*100:.2f}%", ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

prob = model.predict_proba(input_data)[0]

import seaborn as sns

# Bar Graph
st.subheader("📊 Probability Distribution")
fig, ax = plt.subplots()
sns.barplot(x=["Not Click", "Click"], y=prob, palette="pastel", ax=ax)
st.pyplot(fig)

# Line Graph
st.subheader("📈 Line Graph View")
fig, ax = plt.subplots()
ax.plot(["Not Click", "Click"], prob, marker="o", linestyle="--")
st.pyplot(fig)

# Scatter/Dot Graph
st.subheader("⚪ Dot Graph View")
fig, ax = plt.subplots()
ax.scatter(["Not Click", "Click"], prob, color="purple", s=200)
st.pyplot(fig)



# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Data Overview", "Visualizations", "Prediction"])

# ---- Data Overview ----
if page == "Data Overview":
    st.title("📊 Ad Click Data Overview")
    st.write(df.head())
    st.write("Basic Statistics:")
    st.write(df.describe())

# ---- Visualizations ----
elif page == "Visualizations":
    st.title("📈 Data Visualizations")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Age'], bins=20, kde=True, color='skyblue', ax=ax1)
    ax1.set_title("Age Distribution")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(y=df['Income'], color='lightgreen', ax=ax2)
    ax2.set_title("Income Distribution")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.barplot(x='Gender', y='Click', data=df, estimator=lambda x: sum(x)/len(x), ax=ax3)
    ax3.set_title("Click Rate by Gender")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    df['Device'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'), ax=ax4)
    ax4.set_ylabel("")
    ax4.set_title("Device Usage Share")
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax5)
    ax5.set_title("Correlation Heatmap")
    st.pyplot(fig5)

# ---- Prediction ----
elif page == "Prediction":
    st.title("🤖 Ad Click Prediction")
    # Load trained model (replace path with your pickle file)
    model = pickle.load(open("model/final_model.pkl", "rb"))
    label_gender = pickle.load(open("model/label_gender.pkl", "rb"))
    label_location = pickle.load(open("model/label_location.pkl", "rb"))
    label_device = pickle.load(open("model/label_device.pkl", "rb"))
    label_interest = pickle.load(open("model/label_interest.pkl", "rb"))

    Age = st.number_input("Age", 18, 100, 25)
    Gender = st.selectbox("Gender", label_gender.classes_)
    Income = st.number_input("Income", 20000, 100000, 50000)
    Location = st.selectbox("Location", label_location.classes_)
    Device = st.selectbox("Device", label_device.classes_)
    Interest = st.selectbox("Interest", label_interest.classes_)
    Time = st.slider("Time Spent on Site (min)", 0, 120, 60)
    Pages = st.slider("Number of Pages Viewed", 1, 20, 10)

    if st.button("Predict"):
        input_data = pd.DataFrame([[
            Age,
            label_gender.transform([Gender])[0],
            Income,
            label_location.transform([Location])[0],
            label_device.transform([Device])[0],
            label_interest.transform([Interest])[0],
            Time,
            Pages
        ]], columns=['Age','Gender','Income','Location','Device','Interest_Category','Time_Spent_on_Site','Number_of_Pages_Viewed'])

        result = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        st.subheader("Prediction Probability")
        st.bar_chart({"Not Click": prob[0], "Click": prob[1]})

        if result == 1:
            st.success(f"Will Click the Ad! ({prob[1]*100:.2f}% confident)")
        else:
            st.error(f"Will NOT Click the Ad. ({prob[0]*100:.2f}% confident)")
