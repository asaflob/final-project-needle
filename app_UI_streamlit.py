import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configure the Streamlit page layout and settings
st.set_page_config(
    page_title="NBA Draft & Analytics Hub",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================================
# DATA INTEGRATION API
# =====================================================================
class DataManager:
    """
    This class acts as the API between the Streamlit UI and the Data Backend.
    Currently, it generates mock data.
    INSTRUCTION FOR DATA PARTNER: Replace the logic inside these methods
    to return your actual Pandas DataFrames. Keep the return types consistent.
    """

    def __init__(self):
        # We use st.cache_data to ensure mock data doesn't regenerate on every click
        pass

    @staticmethod
    @st.cache_data
    def get_player_list():
        """Returns a list of all NBA players available in the dataset."""
        return ["LeBron James", "Nikola Jokic", "Stephen Curry", "Giannis Antetokounmpo", "Luka Doncic", "Jayson Tatum"]

    @staticmethod
    @st.cache_data
    def get_player_production(player_name):
        """
        Returns production stats for a specific player across multiple seasons.
        Expected format: Pandas DataFrame with columns ['Season', 'PTS', 'AST', 'REB', 'PER', 'WS']
        """
        # MOCK DATA GENERATION
        seasons = [f"20{18 + i}-{19 + i}" for i in range(6)]
        base_pts = np.random.uniform(20, 30)
        df = pd.DataFrame({
            "Season": seasons,
            "PTS": [base_pts + np.random.normal(0, 2) for _ in range(6)],
            "AST": [np.random.uniform(5, 10) for _ in range(6)],
            "REB": [np.random.uniform(4, 12) for _ in range(6)],
            "PER": [np.random.uniform(18, 30) for _ in range(6)],
            "Win Shares": [np.random.uniform(5, 15) for _ in range(6)]
        })
        return df

    @staticmethod
    @st.cache_data
    def get_draft_prospects():
        """
        Returns a dataset of current draft prospects.
        Expected format: Pandas DataFrame with prospect details and predictions.
        """
        # MOCK DATA GENERATION
        names = ["Prospect A", "Prospect B", "Prospect C", "Prospect D", "Prospect E", "Prospect F"]
        positions = ["PG", "SG", "SF", "PF", "C", "PG"]
        colleges = ["Duke", "Kentucky", "UCLA", "Kansas", "Gonzaga", "Ignite"]
        proj_pick = [1, 5, 12, 18, 25, 35]
        bust_prob = [np.random.uniform(0.05, 0.4) for _ in range(6)]
        star_prob = [np.random.uniform(0.1, 0.8) for _ in range(6)]

        df = pd.DataFrame({
            "Name": names,
            "Position": positions,
            "College/Team": colleges,
            "Projected Pick": proj_pick,
            "Star Probability": star_prob,
            "Bust Probability": bust_prob
        })
        return df

    @staticmethod
    @st.cache_data
    def get_hidden_gems(min_win_shares, max_usage, max_salary):
        """
        Filters the dataset to find 'Hidden Gems' based on user UI criteria.
        Expected format: Pandas DataFrame of undervalued players.
        """
        # MOCK DATA GENERATION
        players = [f"Undervalued Player {i}" for i in range(1, 21)]
        df = pd.DataFrame({
            "Player": players,
            "Win Shares": np.random.uniform(min_win_shares, 12, 20),
            "Usage Rate (%)": np.random.uniform(10, max_usage, 20),
            "Salary ($M)": np.random.uniform(1.0, max_salary, 20),
            "Age": np.random.randint(20, 32, 20)
        })
        # Mock filtering based on UI sliders
        df = df[(df["Win Shares"] >= min_win_shares) &
                (df["Usage Rate (%)"] <= max_usage) &
                (df["Salary ($M)"] <= max_salary)]
        return df.sort_values(by="Win Shares", ascending=False)


# =====================================================================
# UI RENDERING FUNCTIONS
# =====================================================================

def render_sidebar():
    """Renders the sidebar navigation."""
    st.sidebar.title("🏀 NBA Analytics")
    st.sidebar.markdown("---")

    # Navigation menu
    selected_page = st.sidebar.radio(
        "Navigation",
        ["Home / Overview", "Player Production", "Draft Predictor", "Hidden Gems Finder"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Project Info:**\n\n"
        "Data Science Final Project.\n\n"
        "Focus: Production, Draft Predictions, & Undervalued Assets."
    )
    return selected_page


def page_home():
    """Renders the landing page."""
    st.title("NBA Player Production & Draft Analytics")
    st.markdown("""
    Welcome to our Data Science Final Project! This dashboard is designed to analyze NBA player 
    production over time, evaluate upcoming draft prospects using machine learning models, 
    and identify "hidden gems" (undervalued players) in the league.
    """)

    # Example of high-level metric cards
    st.subheader("Database Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Players Analyzed", value="4,502")
    with col2:
        st.metric(label="Seasons Covered", value="20")
    with col3:
        st.metric(label="Draft Models Trained", value="3")
    with col4:
        st.metric(label="Data Last Updated", value="Today")

    st.divider()

    st.markdown("### How to use this app:")
    st.markdown("""
    * **Player Production:** Search for current/historical players to view their career trajectories and advanced stats.
    * **Draft Predictor:** Analyze the incoming draft class. View our model's probability of a player becoming an All-Star vs. a Bust.
    * **Hidden Gems Finder:** Use interactive filters to find highly productive players on low salaries or low usage rates.
    """)


def page_player_production(data_api):
    """Renders the historical player production analysis page."""
    st.title("📈 Player Production Analysis")

    players = data_api.get_player_list()
    selected_player = st.selectbox("Search for a Player", players)

    if selected_player:
        df = data_api.get_player_production(selected_player)

        # Display top-level stats for the most recent season
        st.subheader(f"{selected_player} - Recent Performance")
        recent_stats = df.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Points (PTS)", f"{recent_stats['PTS']:.1f}")
        c2.metric("Assists (AST)", f"{recent_stats['AST']:.1f}")
        c3.metric("Rebounds (REB)", f"{recent_stats['REB']:.1f}")
        c4.metric("Win Shares", f"{recent_stats['Win Shares']:.1f}")

        st.markdown("### Career Trajectory")

        # Using tabs to organize charts cleanly
        tab1, tab2, tab3 = st.tabs(["Scoring & Playmaking", "Advanced Metrics", "Raw Data"])

        with tab1:
            fig_scoring = px.line(df, x="Season", y=["PTS", "AST", "REB"],
                                  title=f"{selected_player} Traditional Stats over Time",
                                  markers=True)
            st.plotly_chart(fig_scoring, use_container_width=True)

        with tab2:
            fig_adv = px.bar(df, x="Season", y=["PER", "Win Shares"],
                             barmode="group",
                             title=f"{selected_player} Advanced Stats")
            st.plotly_chart(fig_adv, use_container_width=True)

        with tab3:
            st.dataframe(df, use_container_width=True)


def page_draft_predictor(data_api):
    """Renders the draft prediction page."""
    st.title("🎓 Draft Prospect Predictor")
    st.markdown("Evaluate incoming college and international prospects based on our predictive models.")

    df = data_api.get_draft_prospects()

    # Sidebar filters specific to this page
    st.sidebar.subheader("Draft Filters")
    pos_filter = st.sidebar.multiselect("Filter by Position", ["PG", "SG", "SF", "PF", "C"],
                                        default=["PG", "SG", "SF", "PF", "C"])

    # Apply filters
    filtered_df = df[df["Position"].isin(pos_filter)]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Prospect Rankings & Probabilities")
        st.dataframe(
            filtered_df.style.background_gradient(subset=['Star Probability'], cmap='Greens')
            .background_gradient(subset=['Bust Probability'], cmap='Reds')
            .format({'Star Probability': '{:.1%}', 'Bust Probability': '{:.1%}'}),
            use_container_width=True,
            height=400
        )

    with col2:
        st.subheader("Risk vs Reward")
        fig = px.scatter(filtered_df, x="Bust Probability", y="Star Probability",
                         text="Name", color="Position", size="Projected Pick",
                         size_max=20,
                         title="Prospect Landscape")
        fig.update_traces(textposition='top center')
        fig.update_layout(xaxis_tickformat='.0%', yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)


def page_hidden_gems(data_api):
    """Renders the hidden gems (undervalued players) tool."""
    st.title("💎 Hidden Gems Finder")
    st.markdown("Use the sliders below to find players who overproduce relative to their cost and usage.")

    # Controls layout
    st.markdown("### Filtering Criteria")
    c1, c2, c3 = st.columns(3)
    with c1:
        min_ws = st.slider("Minimum Win Shares", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
    with c2:
        max_usage = st.slider("Maximum Usage Rate (%)", min_value=10.0, max_value=40.0, value=20.0, step=1.0)
    with c3:
        max_salary = st.slider("Maximum Salary ($M)", min_value=1.0, max_value=50.0, value=15.0, step=1.0)

    st.divider()

    # Fetch data based on UI parameters
    df = data_api.get_hidden_gems(min_ws, max_usage, max_salary)

    if df.empty:
        st.warning("No players found matching these criteria. Try loosening your constraints.")
    else:
        st.success(f"Found {len(df)} hidden gems!")

        # Visualization
        fig = px.scatter(df, x="Salary ($M)", y="Win Shares",
                         size="Usage Rate (%)", color="Age",
                         hover_name="Player",
                         title="Value Matrix: Win Shares vs. Salary",
                         color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)

        # Data Table
        st.markdown("### Player List")
        st.dataframe(df, use_container_width=True)


def main():
    """Main function that controls the flow of the Streamlit app."""
    # Instantiate the API connector
    data_api = DataManager()

    # Get navigation choice
    selected_page = render_sidebar()

    # Route to the correct page function
    if selected_page == "Home / Overview":
        page_home()
    elif selected_page == "Player Production":
        page_player_production(data_api)
    elif selected_page == "Draft Predictor":
        page_draft_predictor(data_api)
    elif selected_page == "Hidden Gems Finder":
        page_hidden_gems(data_api)


if __name__ == "__main__":
    main()