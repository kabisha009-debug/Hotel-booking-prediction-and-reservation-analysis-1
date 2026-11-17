
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
    page_icon="🧳",
    layout="wide"
)


# ==================================================
# CHAMPAGNE & MUSTARD BRONZE THEME
# ==================================================

BG_COLOR = "#F4EBD9"
CARD_COLOR = "#FFF9EF"
ACCENT = "#8C6D23"
SECONDARY_ACCENT = "#C6A96B"
DARK_BRONZE = "#5A4212"
TEXT_COLOR = "#302719"
MUTED_COLOR = "#806F52"
BORDER_COLOR = "#E8D9BD"
LIGHT_GOLD = "#D9BD87"

CHART_COLORS = [
    "#8C6D23",
    "#C6A96B",
    "#5A4212",
    "#D9BD87",
    "#A77B35",
    "#E8D9BD"
]


# ==================================================
# GLOBAL FONT AND THEME STYLING
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #F4EBD9;
    color: #302719;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1 {
    color: #5A4212 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

h2 {
    color: #5A4212 !important;
    font-size: 21px !important;
    font-weight: 650 !important;
}

h3 {
    color: #8C6D23 !important;
    font-size: 17px !important;
    font-weight: 600 !important;
}

.stMarkdown p {
    color: #806F52;
    font-size: 13px;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background-color: #5A4212;
    border-right: 1px solid #8C6D23;
}

[data-testid="stSidebar"] * {
    color: #FFF9EF !important;
    font-size: 13px !important;
}

[data-testid="stSidebar"] h1 {
    font-size: 20px !important;
    font-weight: 700 !important;
    margin-bottom: 18px;
}

[data-testid="stSidebar"] h2 {
    color: #FFF9EF !important;
    font-size: 20px !important;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #E8D9BD !important;
    font-size: 12px !important;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 10px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 9px 11px;
    border-radius: 8px;
    transition: background-color 0.2s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background-color: #76551C;
}

/* GLOBAL INPUTS */

[data-testid="stWidgetLabel"] p {
    color: #5A4212 !important;
    font-size: 12px !important;
}

.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stTextArea textarea {
    background-color: #FFF9EF !important;
    color: #302719 !important;
    border: 1px solid #D9BD87 !important;
    border-radius: 7px !important;
    font-size: 13px !important;
}

[data-baseweb="select"] > div {
    background-color: #FFF9EF !important;
    border-color: #D9BD87 !important;
    color: #302719 !important;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background-color: #FFF9EF;
    border: 1px solid #E8D9BD;
    padding: 12px;
    border-radius: 10px;
}

[data-testid="stMetricLabel"] {
    color: #806F52 !important;
    font-size: 12px !important;
}

[data-testid="stMetricValue"] {
    color: #8C6D23 !important;
    font-size: 23px !important;
    font-weight: bold;
}

/* BUTTONS */

.stButton > button,
.stFormSubmitButton > button {
    background-color: #8C6D23 !important;
    color: #FFFFFF !important;
    border: 2px solid #5A4212 !important;
    border-radius: 9px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    min-height: 45px !important;
    opacity: 1 !important;
    visibility: visible !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #5A4212 !important;
    color: #FFFFFF !important;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {
    border: 1px solid #E8D9BD;
    border-radius: 8px;
}

hr {
    border-color: #E8D9BD;
}

[data-testid="stAlert"] {
    border-radius: 8px;
}


/* CANCELLATION PREDICTION PAGE FONTS */

.st-key-prediction_page h1 {
    font-size: 34px !important;
    color: #5A4212 !important;
}

.st-key-prediction_page h2 {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #5A4212 !important;
}

.st-key-prediction_page h3 {
    font-size: 26px !important;
    font-weight: 700 !important;
}

.st-key-prediction_page p {
    font-size: 19px !important;
    color: #806F52 !important;
}

.st-key-prediction_page [data-testid="stWidgetLabel"] p {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #5A4212 !important;
}

.st-key-prediction_page input {
    font-size: 18px !important;
}

.st-key-prediction_page [data-baseweb="select"] {
    font-size: 18px !important;
}

.st-key-prediction_page .stFormSubmitButton button {
    font-size: 19px !important;
    font-weight: 700 !important;
    padding: 14px 26px !important;
}

.st-key-prediction_page [data-testid="stMetricLabel"] {
    font-size: 17px !important;
}

.st-key-prediction_page [data-testid="stMetricValue"] {
    font-size: 30px !important;
}

.st-key-prediction_page [data-testid="stFormSubmitButton"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}


/* WELCOME PAGE */

.welcome-card {
    background-color: #FFF9EF;
    border: 1px solid #E8D9BD;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 15px;
}

.welcome-card h3 {
    color: #5A4212 !important;
    font-size: 19px !important;
}

.welcome-card p {
    color: #806F52 !important;
    font-size: 14px !important;
    line-height: 1.7;
}

/* PREDICTION RESULT */

.result-card {
    background-color: #FFF9EF;
    border: 2px solid #8C6D23;
    border-radius: 10px;
    padding: 18px;
    margin-top: 15px;
}

.result-card h3 {
    color: #5A4212 !important;
    font-size: 20px !important;
}

.result-card p {
    color: #806F52 !important;
    font-size: 14px !important;
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
        color=BORDER_COLOR,
        alpha=0.5,
        linestyle="--",
        linewidth=0.6
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

    fig, ax = plt.subplots(figsize=(4, 3))

    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    def percentage_label(pct):
        return f"{pct:.1f}%" if pct >= 2 else ""

    wedges, _, _ = ax.pie(
        values,
        labels=None,
        autopct=percentage_label,
        startangle=90,
        colors=colors,
        pctdistance=0.78,
        radius=0.72,
        wedgeprops={
            "width": 0.28,
            "edgecolor": BG_COLOR,
            "linewidth": 1.5
        },
        textprops={
            "color": TEXT_COLOR,
            "fontsize": 8
        }
    )

    ax.set_title(
        title,
        color=TEXT_COLOR,
        fontsize=10,
        fontweight="bold",
        pad=5
    )

    legend = ax.legend(
        wedges,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=7,
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
            fontsize=8,
            fontweight="bold"
        )

    ax.set_aspect("equal")

    fig.subplots_adjust(
        top=0.88,
        bottom=0.20,
        left=0.12,
        right=0.88
    )

    return fig


# ==================================================
# DATASET LOADING
# ==================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv("hotel_bookings.csv")

    return df


df = load_dataset()


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
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.markdown(
    "<h4>🧳 Hotel Booking System</h4>",
    unsafe_allow_html=True
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Welcome",
        "Cancellation Prediction",
        "Reservation Analysis"
    ]
)


# ==================================================
# WELCOME PAGE
# ==================================================

if page == "Welcome":

    st.markdown("""
    <style>
    .welcome-title {
        font-size: 36px;
        font-weight: 700;
        color: #5A4212;
        margin-bottom: 8px;
    }

    .welcome-subtitle {
        font-size: 22px;
        font-weight: 600;
        color: #8C6D23;
        margin-bottom: 20px;
    }

    .welcome-heading {
        font-size: 25px;
        font-weight: 700;
        color: #5A4212;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .welcome-text {
        font-size: 18px;
        line-height: 1.8;
        color: #302719;
    }

    .welcome-section-title {
        font-size: 21px;
        font-weight: 700;
        color: #8C6D23;
        margin-top: 12px;
    }

    .welcome-workflow {
        font-size: 17px;
        font-weight: 600;
        line-height: 1.8;
        color: #5A4212;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="welcome-title">🧳 Welcome to Hotel Booking System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="welcome-subtitle">'
        'Hotel Booking Cancellation Prediction and Reservation Analysis System'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="welcome-heading">About This Project</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="welcome-text">
        Enter the hotel reservation details, such as hotel type,
        lead time, arrival month and booking information.
        <br><br>
        The trained Machine Learning model analyses the details
        and predicts whether the booking is likely to be cancelled
        or not cancelled.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div class="welcome-heading">Explore Our System</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="welcome-section-title">
        📋 Cancellation Prediction
    </div>
    <div class="welcome-text">
        Enter booking information and get a prediction about
        whether the hotel reservation is likely to be cancelled.
    </div>

    <div class="welcome-section-title">
        📊 Reservation Analysis
    </div>
    <div class="welcome-text">
        Explore booking statistics, monthly booking trends,
        hotel types, market segments and cancellation patterns
        through visualizations.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div class="welcome-heading">Project Workflow</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="welcome-workflow">
        Dataset Collection → Data Preprocessing → Model Training
        → Cancellation Prediction → Reservation Analysis
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="welcome-text" style="margin-top:15px;">'
        'Select Cancellation Prediction or Reservation Analysis '
        'from the sidebar to get started.'
        '</div>',
        unsafe_allow_html=True
    )


# ==================================================
# PAGE 2: CANCELLATION PREDICTION
# ==================================================

elif page == "Cancellation Prediction":

    with st.container(key="prediction_page"):

        st.title("Hotel Booking Cancellation Prediction")

        st.write(
            "Enter the booking details to predict cancellation."
        )

        # BOOKING FORM

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

            submit = st.form_submit_button(
                "Predict Cancellation"
            )

        # PREDICTION

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

            # CHECK MODEL INPUT COLUMNS

            if set(input_data.columns) != set(feature_columns):

                st.error(
                    "The prediction form columns do not match "
                    "the saved model feature columns. "
                    "Please check feature_columns.pkl."
                )

            else:

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

                # CANCELLED RESULT

                if prediction[0] == 1:

                    st.markdown(
                        f"""
                        <div class="result-card">

                            <h3>
                                ⚠️ Booking is Predicted to be
                                <span style="color: #5A4212; font-weight: 900; font-size: 28px;">
                                    CANCELLED
                                </span>
                            </h3>

                            <p style="font-size: 19px !important; color: #302719 !important;">
                                The model predicts that this reservation
                                may be cancelled.
                            </p>

                            <h3 style="font-size: 42px !important; color: #5A4212 !important; font-weight: 900 !important;">
                                {probability[0][1] * 100:.2f}%
                            </h3>

                            <p>Cancellation Probability</p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # NOT CANCELLED RESULT

                else:

                    st.markdown(
                        f"""
                        <div class="result-card">

                            <h3>✓ Booking is Predicted Not to be Cancelled</h3>

                            <p>
                            The model predicts that this reservation
                            may not be cancelled.
                            </p>

                            <h3>{probability[0][0] * 100:.2f}%</h3>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ==================================================
# PAGE 3: RESERVATION ANALYSIS
# ==================================================

elif page == "Reservation Analysis":

    st.header("📊 Reservation Analysis Dashboard")

    # BOOKING OVERVIEW

    st.subheader("Booking Overview")

    total_bookings = len(df)

    cancelled_bookings = (df["is_canceled"] == 1).sum()

    not_cancelled = (df["is_canceled"] == 0).sum()

    cancellation_rate = (
        cancelled_bookings / total_bookings
    ) * 100 if total_bookings else 0

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

    # ==================================================
    # MONTHLY BOOKING TRENDS - SMALL BAR CHART
    # ==================================================

    st.subheader("Monthly Booking Trends")

    month_order = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    monthly_bookings = (
        df.groupby("arrival_date_month")
        .size()
        .reindex(month_order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(5, 2.4))

    sns.barplot(
        x=month_order,
        y=monthly_bookings.values,
        color=ACCENT,
        width=0.55,
        ax=ax
    )

    ax.set_xlabel("Arrival Month", fontsize=8)

    ax.set_ylabel("Number of Bookings", fontsize=8)

    ax.set_xticklabels(
        month_order,
        rotation=35,
        ha="right",
        fontsize=7
    )

    ax.tick_params(
        axis="y",
        labelsize=7
    )

    style_chart(
        fig,
        ax,
        "Monthly Booking Trends"
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    # ==================================================
    # HOTEL TYPE ANALYSIS
    # ==================================================

    st.subheader("Hotel Type Analysis")

    hotel_bookings = df["hotel"].value_counts()

    fig = create_donut_chart(
        hotel_bookings.values,
        hotel_bookings.index,
        [ACCENT, SECONDARY_ACCENT],
        "Booking Share by Hotel Type",
        f"{hotel_bookings.sum():,}\nBookings"
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    # ==================================================
    # MARKET SEGMENT ANALYSIS
    # ==================================================

    st.subheader("Market Segment Analysis")

    market_bookings = df["market_segment"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 3))

    sns.barplot(
        x=market_bookings.values,
        y=market_bookings.index,
        hue=market_bookings.index,
        palette=sns.color_palette(
            CHART_COLORS,
            n_colors=len(market_bookings)
        ),
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

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    # ==================================================
    # CUSTOMER TYPE ANALYSIS
    # ==================================================

    st.subheader("Customer Type Analysis")

    customer_bookings = df["customer_type"].value_counts()

    customer_colors = [
        CHART_COLORS[i % len(CHART_COLORS)]
        for i in range(len(customer_bookings))
    ]

    fig = create_donut_chart(
        customer_bookings.values,
        customer_bookings.index,
        customer_colors,
        "Customer Type Distribution"
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    # ==================================================
    # CANCELLATION DISTRIBUTION
    # ==================================================

    st.subheader("Cancellation Analysis")

    cancellation_counts = (
        df["is_canceled"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    fig = create_donut_chart(
        cancellation_counts.values,
        ["Not Cancelled", "Cancelled"],
        [ACCENT, "#A65A42"],
        "Booking Cancellation Distribution"
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    # ==================================================
    # CANCELLATION RATE BY HOTEL TYPE
    # ==================================================

    st.subheader("Cancellation Rate by Hotel Type")

    hotel_cancellation = (
        df.groupby("hotel")["is_canceled"].mean() * 100
    ).reset_index()

    hotel_cancellation.columns = [
        "Hotel Type",
        "Cancellation Rate"
    ]

    fig, ax = plt.subplots(figsize=(4.5, 2.5))

    sns.barplot(
        data=hotel_cancellation,
        x="Hotel Type",
        y="Cancellation Rate",
        hue="Hotel Type",
        palette=[ACCENT, SECONDARY_ACCENT],
        legend=False,
        ax=ax
    )

    ax.set_xlabel("Hotel Type", fontsize=8)

    ax.set_ylabel("Cancellation Rate (%)", fontsize=8)

    style_chart(
        fig,
        ax,
        "Cancellation Rate by Hotel Type"
    )

    ax.tick_params(
        axis="both",
        labelsize=7
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    st.divider()

    st.caption(
        "Hotel Booking Cancellation Prediction and Reservation Analysis System"
    )
