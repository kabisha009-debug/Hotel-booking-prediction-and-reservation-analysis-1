
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Hotel Booking Analysis",
    page_icon="🏨",
    layout="wide"
)


# ==================================================
# DARK DASHBOARD THEME
# ==================================================

BG_COLOR = "#101827"
CARD_COLOR = "#1B263B"
ACCENT = "#18B6A4"
BLUE = "#60A5FA"
TEXT_COLOR = "#F8FAFC"
MUTED_COLOR = "#B8C7DA"


# ==================================================
# GLOBAL FONT AND THEME STYLING
# ==================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #101827;
    color: #F8FAFC;
}

/* Main title */
h1 {
    color: #F8FAFC !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* Page headings */
h2 {
    color: #F8FAFC !important;
    font-size: 21px !important;
    font-weight: 650 !important;
}

/* Section headings */
h3 {
    color: #F8FAFC !important;
    font-size: 17px !important;
    font-weight: 600 !important;
}

/* Normal text */
.stMarkdown p {
    color: #D1DCEB;
    font-size: 13px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1B263B;
}

[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
    font-size: 13px !important;
}

/* Input labels */
[data-testid="stWidgetLabel"] p {
    color: #D1DCEB !important;
    font-size: 12px !important;
}

/* Input fields */
input, textarea {
    background-color: #24344D !important;
    color: #F8FAFC !important;
    font-size: 13px !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #24344D;
    border: 1px solid #354966;
    padding: 10px;
    border-radius: 10px;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #B8C7DA !important;
    font-size: 12px !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #18B6A4 !important;
    font-size: 23px !important;
    font-weight: bold;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #18B6A4;
    color: #101827;
    border-radius: 8px;
    border: none;
    font-size: 13px !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# COMMON CHART STYLING
# ==================================================

def style_chart(fig, ax, title):

    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    ax.set_title(
        title,
        color=TEXT_COLOR,
        fontsize=11,
        fontweight="bold",
        pad=10
    )

    ax.tick_params(
        colors=MUTED_COLOR,
        labelsize=8
    )

    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)

    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(
        axis="y",
        color="#52627A",
        alpha=0.20
    )

    ax.set_axisbelow(True)

    fig.tight_layout()


# ==================================================
# REUSABLE DONUT CHART
# ==================================================

def create_donut_chart(
    values,
    labels,
    colors,
    title,
    center_text=None
):

    fig, ax = plt.subplots(figsize=(5.5, 4))

    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    wedges, _, _ = ax.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        pctdistance=0.78,
        radius=0.82,
        wedgeprops={
            "width": 0.35,
            "edgecolor": BG_COLOR,
            "linewidth": 2
        },
        textprops={
            "color": TEXT_COLOR,
            "fontsize": 8
        }
    )

    ax.set_title(
        title,
        color=TEXT_COLOR,
        fontsize=11,
        fontweight="bold",
        pad=8
    )

    legend = ax.legend(
        wedges,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=8,
        frameon=False
    )

    plt.setp(
        legend.get_texts(),
        color=TEXT_COLOR
    )

    if center_text:

        ax.text(
            0,
            0,
            center_text,
            ha="center",
            va="center",
            color=TEXT_COLOR,
            fontsize=9,
            fontweight="bold"
        )

    fig.tight_layout()

    return fig


# ==================================================
# DATASET LOADING
# ==================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv("hotel_bookings.csv")

    return df


# ==================================================
# MODEL LOADING
# ==================================================

@st.cache_resource
def load_model_files():

    model = joblib.load("hotel_cancellation_model.pkl")

    encoder = joblib.load("hotel_encoder.pkl")

    feature_columns = joblib.load("feature_columns.pkl")

    categorical_cols = joblib.load("categorical_cols.pkl")

    return model, encoder, feature_columns, categorical_cols


model, encoder, feature_columns, categorical_cols = load_model_files()


# ==================================================
# MAIN TITLE
# ==================================================

st.title("🏨 Hotel Booking Cancellation Prediction")

st.write(
    "Welcome to the Hotel Booking Prediction and Reservation Analysis System"
)


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Cancellation Prediction",
        "Reservation Analysis"
    ]
)


# ==================================================
# PAGE 1: CANCELLATION PREDICTION
# ==================================================

if page == "Cancellation Prediction":

    st.header("Booking Cancellation Prediction")

    st.write("Enter the booking details to predict cancellation.")

    with st.form("booking_form"):

        hotel = st.selectbox(
            "Hotel Type",
            ["City Hotel", "Resort Hotel"]
        )

        lead_time = st.number_input(
            "Lead Time",
            min_value=0,
            value=100
        )

        arrival_date_month = st.selectbox(
            "Arrival Month",
            [
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December"
            ]
        )

        stays_in_weekend_nights = st.number_input(
            "Weekend Nights",
            min_value=0,
            value=2
        )

        stays_in_week_nights = st.number_input(
            "Week Nights",
            min_value=0,
            value=3
        )

        adults = st.number_input(
            "Adults",
            min_value=0,
            value=2
        )

        children = st.number_input(
            "Children",
            min_value=0,
            value=0
        )

        meal = st.selectbox(
            "Meal",
            ["BB", "HB", "FB", "SC", "Undefined"]
        )

        country = st.text_input(
            "Country Code",
            value="PRT"
        )

        market_segment = st.selectbox(
            "Market Segment",
            [
                "Online TA",
                "Offline TA/TO",
                "Direct",
                "Groups",
                "Corporate",
                "Complementary",
                "Aviation",
                "Undefined"
            ]
        )

        distribution_channel = st.selectbox(
            "Distribution Channel",
            ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        )

        is_repeated_guest = st.selectbox(
            "Repeated Guest",
            [0, 1]
        )

        previous_cancellations = st.number_input(
            "Previous Cancellations",
            min_value=0,
            value=0
        )

        previous_bookings_not_canceled = st.number_input(
            "Previous Bookings Not Canceled",
            min_value=0,
            value=0
        )

        reserved_room_type = st.text_input(
            "Reserved Room Type",
            value="A"
        )

        booking_changes = st.number_input(
            "Booking Changes",
            min_value=0,
            value=0
        )

        deposit_type = st.selectbox(
            "Deposit Type",
            ["No Deposit", "Non Refund", "Refundable"]
        )

        days_in_waiting_list = st.number_input(
            "Days in Waiting List",
            min_value=0,
            value=0
        )

        customer_type = st.selectbox(
            "Customer Type",
            ["Transient", "Transient-Party", "Contract", "Group"]
        )

        adr = st.number_input(
            "ADR",
            min_value=0.0,
            value=100.0
        )

        required_car_parking_spaces = st.number_input(
            "Required Parking Spaces",
            min_value=0,
            value=0
        )

        total_of_special_requests = st.number_input(
            "Total Special Requests",
            min_value=0,
            value=0
        )

        submit = st.form_submit_button("Predict Cancellation")


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if submit:

        input_data = pd.DataFrame([{

            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date_month": arrival_date_month,
            "stays_in_weekend_nights": stays_in_weekend_nights,
            "stays_in_week_nights": stays_in_week_nights,
            "adults": adults,
            "children": children,
            "meal": meal,
            "country": country.strip().upper(),
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "is_repeated_guest": is_repeated_guest,
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "reserved_room_type": reserved_room_type.strip().upper(),
            "booking_changes": booking_changes,
            "deposit_type": deposit_type,
            "days_in_waiting_list": days_in_waiting_list,
            "customer_type": customer_type,
            "adr": adr,
            "required_car_parking_spaces": required_car_parking_spaces,
            "total_of_special_requests": total_of_special_requests

        }])

        input_data = input_data[feature_columns]

        input_data[categorical_cols] = encoder.transform(
            input_data[categorical_cols]
        )

        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        col1.metric(
            "Not Cancelled Probability",
            f"{probability[0][0] * 100:.2f}%"
        )

        col2.metric(
            "Cancelled Probability",
            f"{probability[0][1] * 100:.2f}%"
        )

        if prediction[0] == 1:

            st.error(
                "This booking is likely to be CANCELLED."
            )

        else:

            st.success(
                "This booking is likely to be NOT CANCELLED."
            )


# ==================================================
# PAGE 2: RESERVATION ANALYSIS
# ==================================================

elif page == "Reservation Analysis":

    st.header("📊 Reservation Analysis Dashboard")

    df = load_dataset()


    # --------------------------------------------------
    # BOOKING OVERVIEW
    # --------------------------------------------------

    st.subheader("Booking Overview")

    total_bookings = len(df)

    cancelled_bookings = (df["is_canceled"] == 1).sum()

    not_cancelled = (df["is_canceled"] == 0).sum()

    cancellation_rate = (
        cancelled_bookings / total_bookings
    ) * 100

    hotel_types = df["hotel"].nunique()

    average_adr = df["adr"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Bookings",
        f"{total_bookings:,}"
    )

    col2.metric(
        "Cancelled Bookings",
        f"{cancelled_bookings:,}"
    )

    col3.metric(
        "Not Cancelled",
        f"{not_cancelled:,}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Cancellation Rate",
        f"{cancellation_rate:.2f}%"
    )

    col5.metric(
        "Hotel Types",
        hotel_types
    )

    col6.metric(
        "Average ADR",
        f"{average_adr:.2f}"
    )


    # --------------------------------------------------
    # MONTHLY BOOKING TRENDS
    # --------------------------------------------------

    st.subheader("Monthly Booking Trends")

    month_order = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    monthly_bookings = (
        df.groupby("arrival_date_month")
        .size()
        .reindex(month_order)
    )

    fig, ax = plt.subplots(figsize=(7, 3.5))

    sns.lineplot(
        x=range(len(month_order)),
        y=monthly_bookings.values,
        marker="o",
        markersize=5,
        linewidth=2,
        color=ACCENT,
        ax=ax
    )

    ax.fill_between(
        range(len(month_order)),
        monthly_bookings.values,
        alpha=0.15,
        color=ACCENT
    )

    ax.set_xticks(range(len(month_order)))

    ax.set_xticklabels(
        month_order,
        rotation=30,
        fontsize=8
    )

    ax.set_xlabel("Arrival Month")
    ax.set_ylabel("Number of Bookings")

    style_chart(
        fig,
        ax,
        "Monthly Booking Trends"
    )

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)


    # --------------------------------------------------
    # HOTEL TYPE ANALYSIS
    # --------------------------------------------------

    st.subheader("Hotel Type Analysis")

    hotel_bookings = df["hotel"].value_counts()

    fig = create_donut_chart(
        hotel_bookings.values,
        hotel_bookings.index,
        [ACCENT, BLUE],
        "Booking Share by Hotel Type",
        f"{hotel_bookings.sum():,}\nBookings"
    )

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)


    # --------------------------------------------------
    # MARKET SEGMENT ANALYSIS
    # --------------------------------------------------

    st.subheader("Market Segment Analysis")

    market_bookings = df["market_segment"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        x=market_bookings.values,
        y=market_bookings.index,
        hue=market_bookings.index,
        palette="crest",
        legend=False,
        ax=ax
    )

    ax.set_xlabel("Number of Bookings")
    ax.set_ylabel("Market Segment")

    style_chart(
        fig,
        ax,
        "Bookings by Market Segment"
    )

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)


    # --------------------------------------------------
    # CUSTOMER TYPE ANALYSIS
    # --------------------------------------------------

    st.subheader("Customer Type Analysis")

    customer_bookings = df["customer_type"].value_counts()

    fig = create_donut_chart(
        customer_bookings.values,
        customer_bookings.index,
        [ACCENT, BLUE, "#A78BFA", "#F59E0B"],
        "Customer Type Distribution"
    )

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)


    # --------------------------------------------------
    # CANCELLATION DISTRIBUTION
    # --------------------------------------------------

    st.subheader("Cancellation Analysis")

    cancellation_counts = (
        df["is_canceled"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    fig = create_donut_chart(
        cancellation_counts.values,
        ["Not Cancelled", "Cancelled"],
        [ACCENT, "#F97373"],
        "Booking Cancellation Distribution"
    )

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)


    # --------------------------------------------------
    # CANCELLATION RATE BY HOTEL TYPE
    # --------------------------------------------------

    st.subheader("Cancellation Rate by Hotel Type")

    hotel_cancellation = (
        df.groupby("hotel")["is_canceled"].mean() * 100
    ).reset_index()

    hotel_cancellation.columns = [
        "Hotel Type",
        "Cancellation Rate"
    ]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    sns.barplot(
        data=hotel_cancellation,
        x="Hotel Type",
        y="Cancellation Rate",
        hue="Hotel Type",
        palette=[ACCENT, BLUE],
        legend=False,
        ax=ax
    )

    ax.set_xlabel("Hotel Type")
    ax.set_ylabel("Cancellation Rate (%)")

    style_chart(
        fig,
        ax,
        "Cancellation Rate by Hotel Type"
    )

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)
