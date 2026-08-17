# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
#
# # Configure the Streamlit page layout and settings
# st.set_page_config(
#     page_title="NBA Draft & Analytics Hub",
#     page_icon="🏀",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
#
#
# # =====================================================================
# # DATA INTEGRATION API - YOUR PARTNER WILL EDIT THIS SECTION
# # =====================================================================
# class DataManager:
#     """
#     This class acts as the API between the Streamlit UI and the Data Backend.
#     Currently, it generates mock data.
#     INSTRUCTION FOR DATA PARTNER: Replace the logic inside these methods
#     to return your actual Pandas DataFrames. Keep the return types consistent.
#     """
#
#     def __init__(self):
#         # We use st.cache_data to ensure mock data doesn't regenerate on every click
#         pass
#
#     @staticmethod
#     @st.cache_data
#     def get_player_list():
#         """Returns a list of all NBA players available in the dataset."""
#         return ["LeBron James", "Nikola Jokic", "Stephen Curry", "Giannis Antetokounmpo", "Luka Doncic", "Jayson Tatum"]
#
#     @staticmethod
#     @st.cache_data
#     def get_player_production(player_name):
#         """
#         Returns production stats for a specific player across multiple seasons.
#         Expected format: Pandas DataFrame with columns ['Season', 'PTS', 'AST', 'REB', 'PER', 'WS']
#         """
#         # MOCK DATA GENERATION
#         seasons = [f"20{18 + i}-{19 + i}" for i in range(6)]
#         base_pts = np.random.uniform(20, 30)
#         df = pd.DataFrame({
#             "Season": seasons,
#             "PTS": [base_pts + np.random.normal(0, 2) for _ in range(6)],
#             "AST": [np.random.uniform(5, 10) for _ in range(6)],
#             "REB": [np.random.uniform(4, 12) for _ in range(6)],
#             "PER": [np.random.uniform(18, 30) for _ in range(6)],
#             "Win Shares": [np.random.uniform(5, 15) for _ in range(6)]
#         })
#         return df
#
#     @staticmethod
#     @st.cache_data
#     def get_draft_prospects():
#         """
#         Returns a dataset of current draft prospects.
#         Expected format: Pandas DataFrame with prospect details and predictions.
#         """
#         # MOCK DATA GENERATION
#         names = ["Prospect A", "Prospect B", "Prospect C", "Prospect D", "Prospect E", "Prospect F"]
#         positions = ["PG", "SG", "SF", "PF", "C", "PG"]
#         colleges = ["Duke", "Kentucky", "UCLA", "Kansas", "Gonzaga", "Ignite"]
#         proj_pick = [1, 5, 12, 18, 25, 35]
#         bust_prob = [np.random.uniform(0.05, 0.4) for _ in range(6)]
#         star_prob = [np.random.uniform(0.1, 0.8) for _ in range(6)]
#
#         df = pd.DataFrame({
#             "Name": names,
#             "Position": positions,
#             "College/Team": colleges,
#             "Projected Pick": proj_pick,
#             "Star Probability": star_prob,
#             "Bust Probability": bust_prob
#         })
#         return df
#
#     @staticmethod
#     @st.cache_data
#     def get_hidden_gems(min_win_shares, max_usage, max_salary):
#         """
#         Filters the dataset to find 'Hidden Gems' based on user UI criteria.
#         Expected format: Pandas DataFrame of undervalued players.
#         """
#         # MOCK DATA GENERATION
#         players = [f"Undervalued Player {i}" for i in range(1, 21)]
#         df = pd.DataFrame({
#             "Player": players,
#             "Win Shares": np.random.uniform(min_win_shares, 12, 20),
#             "Usage Rate (%)": np.random.uniform(10, max_usage, 20),
#             "Salary ($M)": np.random.uniform(1.0, max_salary, 20),
#             "Age": np.random.randint(20, 32, 20)
#         })
#         # Mock filtering based on UI sliders
#         df = df[(df["Win Shares"] >= min_win_shares) &
#                 (df["Usage Rate (%)"] <= max_usage) &
#                 (df["Salary ($M)"] <= max_salary)]
#         return df.sort_values(by="Win Shares", ascending=False)
#
#
# # =====================================================================
# # UI RENDERING FUNCTIONS
# # =====================================================================
#
# def inject_custom_css():
#     """Injects custom HTML/CSS to style the app with NBA colors and professional spacing."""
#     st.markdown("""
#     <style>
#     /* Typography and Colors */
#     h1 { color: #1d428a !important; font-weight: 800 !important; letter-spacing: -1px; }
#     h2 { color: #c9082a !important; font-weight: 700 !important; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; margin-top: 20px;}
#     h3 { color: #1d428a !important; font-weight: 600 !important; }
#
#     /* Image styling for a polished look */
#     img {
#         border-radius: 12px;
#         box-shadow: 0 6px 12px rgba(0,0,0,0.15);
#         margin-bottom: 15px;
#         transition: transform 0.3s ease;
#     }
#     img:hover {
#         transform: scale(1.02);
#     }
#
#     /* Highlighted Metric Data */
#     [data-testid="stMetricValue"] {
#         color: #c9082a !important;
#         font-weight: 900 !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)
#
#
# def render_sidebar():
#     """Renders the sidebar navigation with the NBA logo."""
#     # Official NBA Logo from Wikimedia Commons
#     st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/0/03/National_Basketball_Association_logo.svg",
#                      width=80)
#     st.sidebar.title("NBA Analytics Pro")
#     st.sidebar.markdown("---")
#
#     # Navigation menu
#     selected_page = st.sidebar.radio(
#         "Navigation Area",
#         ["Home / Overview", "Player Production", "Draft Predictor War Room", "Hidden Gems Finder"]
#     )
#
#     st.sidebar.markdown("---")
#     st.sidebar.info(
#         "**Project Info:**\n\n"
#         "Data Science Final Project.\n\n"
#         "Designed to find the next generation of NBA superstars and undervalued talent."
#     )
#     return selected_page
#
#
# def page_home():
#     """Renders the highly visual landing page."""
#     st.title("NBA Data Science & Analytics Hub")
#     st.markdown("""
#     Welcome to the premier analytics dashboard for NBA player evaluation. This tool leverages advanced machine
#     learning and historical data to dissect player production, forecast draft prospects, and uncover hidden gems.
#     """)
#
#     # Hero Image of a Championship Team (Warriors 2022 Parade via Wikimedia)
#     st.image(
#         "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/2022_Golden_State_Warriors_Championship_Parade_01.jpg/1024px-2022_Golden_State_Warriors_Championship_Parade_01.jpg",
#         caption="The Ultimate Goal: Building a Championship Roster through Data.",
#         use_container_width=True)
#
#     # High-level metric cards
#     st.subheader("Database Overview")
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric(label="Total Players Analyzed", value="4,502")
#     with col2:
#         st.metric(label="Seasons Covered", value="20+")
#     with col3:
#         st.metric(label="Draft Models Trained", value="3")
#     with col4:
#         st.metric(label="Data Status", value="Live")
#
#     st.divider()
#
#     # All-Time Greats Showcase
#     st.subheader("The Standard of Greatness")
#     st.markdown("We track the historical trajectories of Hall of Famers to predict future success.")
#
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         st.image("https://upload.wikimedia.org/wikipedia/commons/a/ae/Michael_Jordan_in_1992.jpg",
#                  caption="Michael Jordan", use_container_width=True)
#     with c2:
#         st.image(
#             "https://upload.wikimedia.org/wikipedia/commons/7/7a/LeBron_James_%2851959977144%29_%28cropped2%29.jpg",
#             caption="LeBron James", use_container_width=True)
#     with c3:
#         st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Kobe_Bryant_2014.jpg", caption="Kobe Bryant",
#                  use_container_width=True)
#
#
# def page_player_production(data_api):
#     """Renders the historical player production analysis page."""
#     st.title("📈 Player Production Analysis")
#
#     col_search, col_img = st.columns([3, 1])
#     with col_search:
#         players = data_api.get_player_list()
#         selected_player = st.selectbox("Search our database for a current or historical player:", players)
#
#     with col_img:
#         # Generic action shot for aesthetics
#         st.image(
#             "https://images.unsplash.com/photo-1519861531473-9200262188bf?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
#             use_container_width=True)
#
#     if selected_player:
#         df = data_api.get_player_production(selected_player)
#
#         st.subheader(f"Recent Performance: {selected_player}")
#         recent_stats = df.iloc[-1]
#
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Points (PTS)", f"{recent_stats['PTS']:.1f}", delta="Top 10%")
#         c2.metric("Assists (AST)", f"{recent_stats['AST']:.1f}")
#         c3.metric("Rebounds (REB)", f"{recent_stats['REB']:.1f}")
#         c4.metric("Win Shares", f"{recent_stats['Win Shares']:.1f}")
#
#         st.markdown("### Career Trajectory")
#
#         tab1, tab2, tab3 = st.tabs(["Scoring & Playmaking", "Advanced Metrics", "Raw Data View"])
#
#         with tab1:
#             fig_scoring = px.line(df, x="Season", y=["PTS", "AST", "REB"],
#                                   title=f"{selected_player} Traditional Stats over Time",
#                                   markers=True, color_discrete_sequence=["#1d428a", "#c9082a", "#868686"])
#             fig_scoring.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_scoring, use_container_width=True)
#
#         with tab2:
#             fig_adv = px.bar(df, x="Season", y=["PER", "Win Shares"],
#                              barmode="group",
#                              title=f"{selected_player} Advanced Stats",
#                              color_discrete_sequence=["#1d428a", "#c9082a"])
#             fig_adv.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_adv, use_container_width=True)
#
#         with tab3:
#             st.dataframe(df, use_container_width=True)
#
#
# def page_draft_predictor(data_api):
#     """Renders the draft prediction page."""
#     st.title("🎓 Draft Predictor War Room")
#
#     st.markdown("Evaluate incoming college and international prospects based on our predictive models.")
#
#     # Adding a visual for the Draft
#     st.image("https://upload.wikimedia.org/wikipedia/commons/d/dd/Adam_Silver_2017.jpg",
#              caption="Who will be the next generational talent?",
#              width=400)
#
#     df = data_api.get_draft_prospects()
#
#     st.sidebar.subheader("Draft Filters")
#     pos_filter = st.sidebar.multiselect("Filter by Position", ["PG", "SG", "SF", "PF", "C"],
#                                         default=["PG", "SG", "SF", "PF", "C"])
#     filtered_df = df[df["Position"].isin(pos_filter)]
#
#     col1, col2 = st.columns([2, 1])
#
#     with col1:
#         st.subheader("Model Projections")
#         st.dataframe(
#             filtered_df.style.background_gradient(subset=['Star Probability'], cmap='Blues')
#             .background_gradient(subset=['Bust Probability'], cmap='Reds')
#             .format({'Star Probability': '{:.1%}', 'Bust Probability': '{:.1%}'}),
#             use_container_width=True,
#             height=350
#         )
#
#     with col2:
#         st.subheader("Risk vs Reward Mapping")
#         fig = px.scatter(filtered_df, x="Bust Probability", y="Star Probability",
#                          text="Name", color="Position", size="Projected Pick",
#                          size_max=20,
#                          title="Prospect Landscape")
#         fig.update_traces(textposition='top center')
#         fig.update_layout(xaxis_tickformat='.0%', yaxis_tickformat='.0%', plot_bgcolor="rgba(0,0,0,0.05)")
#         st.plotly_chart(fig, use_container_width=True)
#
#
# def page_hidden_gems(data_api):
#     """Renders the hidden gems (undervalued players) tool."""
#     st.title("💎 Hidden Gems Finder")
#     st.markdown("Isolate players who vastly out-produce their salary and usage rate constraints.")
#
#     # Controls layout in a nice formatted box
#     with st.container():
#         st.markdown("### 🎛️ Adjust Market Constraints")
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             min_ws = st.slider("Minimum Win Shares", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
#         with c2:
#             max_usage = st.slider("Max Usage Rate (%)", min_value=10.0, max_value=40.0, value=20.0, step=1.0)
#         with c3:
#             max_salary = st.slider("Max Salary ($M)", min_value=1.0, max_value=50.0, value=15.0, step=1.0)
#
#     st.divider()
#
#     df = data_api.get_hidden_gems(min_ws, max_usage, max_salary)
#
#     if df.empty:
#         st.error("No players found matching these strict criteria. Adjust the market constraints above.")
#     else:
#         st.success(f"Discovered {len(df)} highly efficient, low-cost assets!")
#
#         fig = px.scatter(df, x="Salary ($M)", y="Win Shares",
#                          size="Usage Rate (%)", color="Age",
#                          hover_name="Player",
#                          title="Value Matrix: Production vs. Cost",
#                          color_continuous_scale=px.colors.sequential.RdBu)
#         fig.update_layout(plot_bgcolor="rgba(0,0,0,0.02)")
#         st.plotly_chart(fig, use_container_width=True)
#
#         st.markdown("### Shortlisted Players")
#         st.dataframe(df, use_container_width=True)
#
#
# def main():
#     """Main function that controls the flow of the Streamlit app."""
#     # Apply custom NBA aesthetics
#     inject_custom_css()
#
#     # Instantiate the API connector
#     data_api = DataManager()
#
#     # Get navigation choice
#     selected_page = render_sidebar()
#
#     # Route to the correct page function
#     if selected_page == "Home / Overview":
#         page_home()
#     elif selected_page == "Player Production":
#         page_player_production(data_api)
#     elif selected_page == "Draft Predictor War Room":
#         page_draft_predictor(data_api)
#     elif selected_page == "Hidden Gems Finder":
#         page_hidden_gems(data_api)
#
#
# if __name__ == "__main__":
#     main()

#
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
#
# # Configure the Streamlit page layout and settings
# st.set_page_config(
#     page_title="NBA Draft & Analytics Hub",
#     page_icon="🏀",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
#
#
# # =====================================================================
# # DATA INTEGRATION API - YOUR PARTNER WILL EDIT THIS SECTION
# # =====================================================================
# class DataManager:
#     """
#     This class acts as the API between the Streamlit UI and the Data Backend.
#     Currently, it generates mock data.
#     INSTRUCTION FOR DATA PARTNER: Replace the logic inside these methods
#     to return your actual Pandas DataFrames. Keep the return types consistent.
#     """
#
#     def __init__(self):
#         # We use st.cache_data to ensure mock data doesn't regenerate on every click
#         pass
#
#     @staticmethod
#     @st.cache_data
#     def get_player_list():
#         """Returns a list of all NBA players available in the dataset."""
#         return ["LeBron James", "Nikola Jokic", "Stephen Curry", "Giannis Antetokounmpo", "Luka Doncic", "Jayson Tatum"]
#
#     @staticmethod
#     @st.cache_data
#     def get_player_production(player_name):
#         """
#         Returns production stats for a specific player across multiple seasons.
#         Expected format: Pandas DataFrame with columns ['Season', 'PTS', 'AST', 'REB', 'PER', 'WS']
#         """
#         # MOCK DATA GENERATION
#         seasons = [f"20{18 + i}-{19 + i}" for i in range(6)]
#         base_pts = np.random.uniform(20, 30)
#         df = pd.DataFrame({
#             "Season": seasons,
#             "PTS": [base_pts + np.random.normal(0, 2) for _ in range(6)],
#             "AST": [np.random.uniform(5, 10) for _ in range(6)],
#             "REB": [np.random.uniform(4, 12) for _ in range(6)],
#             "PER": [np.random.uniform(18, 30) for _ in range(6)],
#             "Win Shares": [np.random.uniform(5, 15) for _ in range(6)]
#         })
#         return df
#
#     @staticmethod
#     @st.cache_data
#     def get_draft_prospects():
#         """
#         Returns a dataset of current draft prospects.
#         Expected format: Pandas DataFrame with prospect details and predictions.
#         """
#         # MOCK DATA GENERATION
#         names = ["Prospect A", "Prospect B", "Prospect C", "Prospect D", "Prospect E", "Prospect F"]
#         positions = ["PG", "SG", "SF", "PF", "C", "PG"]
#         colleges = ["Duke", "Kentucky", "UCLA", "Kansas", "Gonzaga", "Ignite"]
#         proj_pick = [1, 5, 12, 18, 25, 35]
#         bust_prob = [np.random.uniform(0.05, 0.4) for _ in range(6)]
#         star_prob = [np.random.uniform(0.1, 0.8) for _ in range(6)]
#
#         df = pd.DataFrame({
#             "Name": names,
#             "Position": positions,
#             "College/Team": colleges,
#             "Projected Pick": proj_pick,
#             "Star Probability": star_prob,
#             "Bust Probability": bust_prob
#         })
#         return df
#
#     @staticmethod
#     @st.cache_data
#     def get_hidden_gems(min_win_shares, max_usage, max_salary):
#         """
#         Filters the dataset to find 'Hidden Gems' based on user UI criteria.
#         Expected format: Pandas DataFrame of undervalued players.
#         """
#         # MOCK DATA GENERATION
#         players = [f"Undervalued Player {i}" for i in range(1, 21)]
#         df = pd.DataFrame({
#             "Player": players,
#             "Win Shares": np.random.uniform(min_win_shares, 12, 20),
#             "Usage Rate (%)": np.random.uniform(10, max_usage, 20),
#             "Salary ($M)": np.random.uniform(1.0, max_salary, 20),
#             "Age": np.random.randint(20, 32, 20)
#         })
#         # Mock filtering based on UI sliders
#         df = df[(df["Win Shares"] >= min_win_shares) &
#                 (df["Usage Rate (%)"] <= max_usage) &
#                 (df["Salary ($M)"] <= max_salary)]
#         return df.sort_values(by="Win Shares", ascending=False)
#
#
# # =====================================================================
# # UI RENDERING FUNCTIONS
# # =====================================================================
#
# def inject_custom_css():
#     """Injects custom HTML/CSS to style the app with NBA colors and professional spacing."""
#     st.markdown("""
#     <style>
#     /* Typography and Colors */
#     h1 { color: #1d428a !important; font-weight: 800 !important; letter-spacing: -1px; }
#     h2 { color: #c9082a !important; font-weight: 700 !important; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; margin-top: 20px;}
#     h3 { color: #1d428a !important; font-weight: 600 !important; }
#
#     /* Image styling for a polished look */
#     img {
#         border-radius: 12px;
#         box-shadow: 0 6px 12px rgba(0,0,0,0.15);
#         margin-bottom: 15px;
#         transition: transform 0.3s ease;
#     }
#     img:hover {
#         transform: scale(1.02);
#     }
#
#     /* Highlighted Metric Data */
#     [data-testid="stMetricValue"] {
#         color: #c9082a !important;
#         font-weight: 900 !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)
#
#
# def render_sidebar():
#     """Renders the sidebar navigation with the NBA logo."""
#     # Official NBA Logo from Wikimedia Commons
#     st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/0/03/National_Basketball_Association_logo.svg",
#                      width=80)
#     st.sidebar.title("NBA Analytics Pro")
#     st.sidebar.markdown("---")
#
#     # Navigation menu
#     selected_page = st.sidebar.radio(
#         "Navigation Area",
#         ["Home / Overview", "Player Production", "Draft Predictor War Room", "Hidden Gems Finder"]
#     )
#
#     st.sidebar.markdown("---")
#     st.sidebar.info(
#         "**Project Info:**\n\n"
#         "Data Science Final Project.\n\n"
#         "Designed to find the next generation of NBA superstars and undervalued talent."
#     )
#     return selected_page
#
#
# def page_home():
#     """Renders the highly visual landing page."""
#     st.markdown("<h1 style='text-align: center; color: #1d428a;'>NBA Data Science & Analytics Hub</h1>",
#                 unsafe_allow_html=True)
#     st.markdown(
#         "<p style='text-align: center; font-size: 1.1rem;'>Welcome to the premier analytics dashboard for NBA player evaluation. This tool leverages advanced machine learning and historical data to dissect player production, forecast draft prospects, and uncover hidden gems.</p>",
#         unsafe_allow_html=True)
#
#     st.markdown("### The Standard of Greatness - All-Time Legends")
#     # Banner of NBA Greats requested by user
#     c1, c2, c3, c4, c5 = st.columns(5)
#     with c1:
#         st.image("https://upload.wikimedia.org/wikipedia/commons/a/ae/Michael_Jordan_in_1992.jpg",
#                  caption="Michael Jordan", use_container_width=True)
#     with c2:
#         st.image("https://upload.wikimedia.org/wikipedia/commons/b/b8/Kareem_Abdul_Jabbar_%28cropped%29.jpg",
#                  caption="Kareem Abdul-Jabbar", use_container_width=True)
#     with c3:
#         st.image("https://upload.wikimedia.org/wikipedia/commons/a/a2/Magic_Johnson_Lipofsky.jpg",
#                  caption="Magic Johnson", use_container_width=True)
#     with c4:
#         st.image("https://upload.wikimedia.org/wikipedia/commons/4/4a/Larry_Bird_Lipofsky.jpg", caption="Larry Bird",
#                  use_container_width=True)
#     with c5:
#         st.image("https://upload.wikimedia.org/wikipedia/commons/e/ec/Julius_Erving_1981.jpg",
#                  caption="Julius Erving (Dr. J)", use_container_width=True)
#
#     st.divider()
#
#     # Hero Image of a Championship Team Raising the Trophy On Court
#     st.subheader("The Ultimate Goal: Championship Glory on the Court")
#
#     # Using an iconic visual representing the championship on the court
#     st.image(
#         "https://images.unsplash.com/photo-1504450758481-7338eba7524a?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
#         caption="Building a Championship Roster through Data.",
#         use_container_width=True)
#
#     # High-level metric cards
#     st.subheader("Database Overview")
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric(label="Total Players Analyzed", value="4,502")
#     with col2:
#         st.metric(label="Seasons Covered", value="20+")
#     with col3:
#         st.metric(label="Draft Models Trained", value="3")
#     with col4:
#         st.metric(label="Data Status", value="Live")
#
#
# def page_player_production(data_api):
#     """Renders the historical player production analysis page."""
#     st.title("📈 Player Production Analysis")
#
#     col_search, col_img = st.columns([3, 1])
#     with col_search:
#         players = data_api.get_player_list()
#         selected_player = st.selectbox("Search our database for a current or historical player:", players)
#
#     with col_img:
#         # Generic action shot for aesthetics
#         st.image(
#             "https://images.unsplash.com/photo-1519861531473-9200262188bf?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
#             use_container_width=True)
#
#     if selected_player:
#         df = data_api.get_player_production(selected_player)
#
#         st.subheader(f"Recent Performance: {selected_player}")
#         recent_stats = df.iloc[-1]
#
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Points (PTS)", f"{recent_stats['PTS']:.1f}", delta="Top 10%")
#         c2.metric("Assists (AST)", f"{recent_stats['AST']:.1f}")
#         c3.metric("Rebounds (REB)", f"{recent_stats['REB']:.1f}")
#         c4.metric("Win Shares", f"{recent_stats['Win Shares']:.1f}")
#
#         st.markdown("### Career Trajectory")
#
#         tab1, tab2, tab3 = st.tabs(["Scoring & Playmaking", "Advanced Metrics", "Raw Data View"])
#
#         with tab1:
#             fig_scoring = px.line(df, x="Season", y=["PTS", "AST", "REB"],
#                                   title=f"{selected_player} Traditional Stats over Time",
#                                   markers=True, color_discrete_sequence=["#1d428a", "#c9082a", "#868686"])
#             fig_scoring.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_scoring, use_container_width=True)
#
#         with tab2:
#             fig_adv = px.bar(df, x="Season", y=["PER", "Win Shares"],
#                              barmode="group",
#                              title=f"{selected_player} Advanced Stats",
#                              color_discrete_sequence=["#1d428a", "#c9082a"])
#             fig_adv.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_adv, use_container_width=True)
#
#         with tab3:
#             st.dataframe(df, use_container_width=True)
#
#
# def page_draft_predictor(data_api):
#     """Renders the draft prediction page."""
#     st.title("🎓 Draft Predictor War Room")
#
#     st.markdown("Evaluate incoming college and international prospects based on our predictive models.")
#
#     # Adding a visual for the Draft
#     st.image("https://upload.wikimedia.org/wikipedia/commons/d/dd/Adam_Silver_2017.jpg",
#              caption="Who will be the next generational talent?",
#              width=400)
#
#     df = data_api.get_draft_prospects()
#
#     st.sidebar.subheader("Draft Filters")
#     pos_filter = st.sidebar.multiselect("Filter by Position", ["PG", "SG", "SF", "PF", "C"],
#                                         default=["PG", "SG", "SF", "PF", "C"])
#     filtered_df = df[df["Position"].isin(pos_filter)]
#
#     col1, col2 = st.columns([2, 1])
#
#     with col1:
#         st.subheader("Model Projections")
#         st.dataframe(
#             filtered_df.style.background_gradient(subset=['Star Probability'], cmap='Blues')
#             .background_gradient(subset=['Bust Probability'], cmap='Reds')
#             .format({'Star Probability': '{:.1%}', 'Bust Probability': '{:.1%}'}),
#             use_container_width=True,
#             height=350
#         )
#
#     with col2:
#         st.subheader("Risk vs Reward Mapping")
#         fig = px.scatter(filtered_df, x="Bust Probability", y="Star Probability",
#                          text="Name", color="Position", size="Projected Pick",
#                          size_max=20,
#                          title="Prospect Landscape")
#         fig.update_traces(textposition='top center')
#         fig.update_layout(xaxis_tickformat='.0%', yaxis_tickformat='.0%', plot_bgcolor="rgba(0,0,0,0.05)")
#         st.plotly_chart(fig, use_container_width=True)
#
#
# def page_hidden_gems(data_api):
#     """Renders the hidden gems (undervalued players) tool."""
#     st.title("💎 Hidden Gems Finder")
#     st.markdown("Isolate players who vastly out-produce their salary and usage rate constraints.")
#
#     # Controls layout in a nice formatted box
#     with st.container():
#         st.markdown("### 🎛️ Adjust Market Constraints")
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             min_ws = st.slider("Minimum Win Shares", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
#         with c2:
#             max_usage = st.slider("Max Usage Rate (%)", min_value=10.0, max_value=40.0, value=20.0, step=1.0)
#         with c3:
#             max_salary = st.slider("Max Salary ($M)", min_value=1.0, max_value=50.0, value=15.0, step=1.0)
#
#     st.divider()
#
#     df = data_api.get_hidden_gems(min_ws, max_usage, max_salary)
#
#     if df.empty:
#         st.error("No players found matching these strict criteria. Adjust the market constraints above.")
#     else:
#         st.success(f"Discovered {len(df)} highly efficient, low-cost assets!")
#
#         fig = px.scatter(df, x="Salary ($M)", y="Win Shares",
#                          size="Usage Rate (%)", color="Age",
#                          hover_name="Player",
#                          title="Value Matrix: Production vs. Cost",
#                          color_continuous_scale=px.colors.sequential.RdBu)
#         fig.update_layout(plot_bgcolor="rgba(0,0,0,0.02)")
#         st.plotly_chart(fig, use_container_width=True)
#
#         st.markdown("### Shortlisted Players")
#         st.dataframe(df, use_container_width=True)
#
#
# def main():
#     """Main function that controls the flow of the Streamlit app."""
#     # Apply custom NBA aesthetics
#     inject_custom_css()
#
#     # Instantiate the API connector
#     data_api = DataManager()
#
#     # Get navigation choice
#     selected_page = render_sidebar()
#
#     # Route to the correct page function
#     if selected_page == "Home / Overview":
#         page_home()
#     elif selected_page == "Player Production":
#         page_player_production(data_api)
#     elif selected_page == "Draft Predictor War Room":
#         page_draft_predictor(data_api)
#     elif selected_page == "Hidden Gems Finder":
#         page_hidden_gems(data_api)
#
#
# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import urllib.request

# Configure the Streamlit page layout and settings
st.set_page_config(
    page_title="NBA Draft & Analytics Hub",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(show_spinner=False)
def fetch_image_bytes(url):
    """
    Fetches an image from a URL using a standard web browser User-Agent.
    This strongly bypasses hotlink protections (Error 403) from Wikipedia.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read()
    except Exception:
        # Fallback placeholder if the image fails to load for any reason
        return "https://placehold.co/400x500/1d428a/ffffff.png?text=Image+Unavailable"


# =====================================================================
# DATA INTEGRATION API - YOUR PARTNER WILL EDIT THIS SECTION
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

def go_home():
    """Callback function to reset the app navigation to the Home page."""
    st.session_state.nav_selection = "Home / Overview"


def inject_custom_css():
    """Injects custom HTML/CSS to style the app with NBA colors and professional spacing."""
    st.markdown("""
    <style>
    /* Typography and Colors */
    h1 { color: #1d428a !important; font-weight: 800 !important; letter-spacing: -1px; }
    h2 { color: #c9082a !important; font-weight: 700 !important; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; margin-top: 20px;}
    h3 { color: #1d428a !important; font-weight: 600 !important; }

    /* Image styling for a polished look */
    img {
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    }
    img:hover {
        transform: scale(1.02);
    }

    /* Highlighted Metric Data */
    [data-testid="stMetricValue"] {
        color: #c9082a !important;
        font-weight: 900 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renders the sidebar navigation with the NBA logo."""
    # Official NBA Logo from Wikimedia Commons
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/0/03/National_Basketball_Association_logo.svg",
                     width=80)
    st.sidebar.title("NBA Analytics Pro")
    st.sidebar.markdown("---")

    # Navigation menu tied to Streamlit's session state memory
    if "nav_selection" not in st.session_state:
        st.session_state.nav_selection = "Home / Overview"

    selected_page = st.sidebar.radio(
        "Navigation Area",
        ["Home / Overview", "Player Production", "Draft Predictor War Room", "Hidden Gems Finder"],
        key="nav_selection"
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Project Info:**\n\n"
        "Data Science Final Project.\n\n"
        "Designed to find the next generation of NBA superstars and undervalued talent."
    )
    return selected_page


def page_home():
    """Renders the highly visual landing page."""
    st.markdown("<h1 style='text-align: center; color: #1d428a;'>NBA Data Science & Analytics Hub</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.1rem;'>Welcome to the premier analytics dashboard for NBA player evaluation. This tool leverages advanced machine learning and historical data to dissect player production, forecast draft prospects, and uncover hidden gems.</p>",
        unsafe_allow_html=True)

    st.markdown("### The Standard of Greatness - All-Time Legends")

    c1, c2, c3, c4, c5 = st.columns(5)

    img_mj = fetch_image_bytes("https://upload.en.wikipedia.org/wiki/Michael_Jordan#/media/File:Jordan_by_Lipofsky_16577_(high_quality).jpg")
    img_kareem = fetch_image_bytes(
        "https://upload.wikimedia.org/wikipedia/commons/d/d7/Kareem_Abdul_Jabbar_%28cropped%29.jpg")
    img_magic = fetch_image_bytes("https://upload.wikimedia.org/wikipedia/commons/f/f6/Magic_Johnson_Lipofsky.jpg")
    img_bird = fetch_image_bytes("https://upload.wikimedia.org/wikipedia/commons/c/cd/Larry_Bird_Lipofsky.jpg")
    #img_drj = fetch_image_bytes("greats_png/dr j.png")

    with c1:
        st.image(img_mj, caption="Michael Jordan", use_container_width=True)
    with c2:
        st.image(img_kareem, caption="Kareem Abdul-Jabbar", use_container_width=True)
    with c3:
        st.image(img_magic, caption="Magic Johnson", use_container_width=True)
    with c4:
        st.image(img_bird, caption="Larry Bird", use_container_width=True)
    #with c5:
        #st.image(img_drj, caption="Julius Erving (Dr. J)", use_container_width=True)

    st.divider()

    st.subheader("The Ultimate Goal: Championship Glory on the Court")

    img_champ = fetch_image_bytes(
        "https://images.unsplash.com/photo-1504450758481-7338eba7524a?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80")
    st.image(img_champ, caption="Building a Championship Roster through Data.", use_container_width=True)

    # High-level metric cards
    st.subheader("Database Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Players Analyzed", value="4,502")
    with col2:
        st.metric(label="Seasons Covered", value="20+")
    with col3:
        st.metric(label="Draft Models Trained", value="3")
    with col4:
        st.metric(label="Data Status", value="Live")


def page_player_production(data_api):
    """Renders the player production analytics page."""
    st.button("🔙 Back to Home", on_click=go_home)
    st.title("📈 Player Production & Career Trajectory")
    st.markdown(
        "Analyze historical performance metrics for current NBA players. *(Tip: Double-click any chart to reset its zoom)*")

    # Use DataManager to get list of players
    players = data_api.get_player_list()
    selected_player = st.selectbox("Select a Player to Analyze", players)

    # Get production data
    df = data_api.get_player_production(selected_player)

    st.subheader(f"{selected_player} - Production Overview")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(df, x="Season", y="PTS", title="Points Per Game (PPG) over Time", markers=True)
        fig1.update_traces(line_color='#c9082a')
        fig1.update_layout(plot_bgcolor="rgba(0,0,0,0.02)")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(df, x="Season", y="Win Shares", title="Win Shares over Time")
        fig2.update_traces(marker_color='#1d428a')
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0.02)")
        st.plotly_chart(fig2, use_container_width=True)


def page_draft_predictor(data_api):
    """Renders the draft predictor war room."""
    st.button("🔙 Back to Home", on_click=go_home)
    st.title("🎯 Draft Predictor War Room")
    st.markdown(
        "Evaluate incoming college and international prospects using our predictive models. *(Tip: Double-click any chart to reset its zoom)*")

    img_draft = fetch_image_bytes("https://upload.wikimedia.org/wikipedia/commons/d/dd/Adam_Silver_2017.jpg")
    st.image(img_draft, caption="Who will be the next generational talent?", width=400)

    df = data_api.get_draft_prospects()

    st.sidebar.subheader("Draft Filters")
    pos_filter = st.sidebar.multiselect("Filter by Position", ["PG", "SG", "SF", "PF", "C"],
                                        default=["PG", "SG", "SF", "PF", "C"])
    filtered_df = df[df["Position"].isin(pos_filter)]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Model Projections")
        st.dataframe(
            filtered_df.style.background_gradient(subset=['Star Probability'], cmap='Blues')
            .background_gradient(subset=['Bust Probability'], cmap='Reds')
            .format({'Star Probability': '{:.1%}', 'Bust Probability': '{:.1%}'}),
            use_container_width=True,
            height=350
        )

    with col2:
        st.subheader("Risk vs Reward Mapping")
        fig = px.scatter(filtered_df, x="Bust Probability", y="Star Probability",
                         text="Name", color="Position", size="Projected Pick",
                         size_max=20,
                         title="Prospect Landscape")
        fig.update_traces(textposition='top center')
        fig.update_layout(xaxis_tickformat='.0%', yaxis_tickformat='.0%', plot_bgcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)


def page_hidden_gems(data_api):
    """Renders the hidden gems (undervalued players) tool."""
    st.button("🔙 Back to Home", on_click=go_home)
    st.title("💎 Hidden Gems Finder")
    st.markdown(
        "Isolate players who vastly out-produce their salary and usage rate constraints. *(Tip: Double-click any chart to reset its zoom)*")

    # Controls layout in a nice formatted box
    with st.container():
        st.markdown("### 🎛️ Adjust Market Constraints")
        c1, c2, c3 = st.columns(3)
        with c1:
            min_ws = st.slider("Minimum Win Shares", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
        with c2:
            max_usage = st.slider("Max Usage Rate (%)", min_value=10.0, max_value=40.0, value=20.0, step=1.0)
        with c3:
            max_salary = st.slider("Max Salary ($M)", min_value=1.0, max_value=50.0, value=15.0, step=1.0)

    st.divider()

    df = data_api.get_hidden_gems(min_ws, max_usage, max_salary)

    if df.empty:
        st.error("No players found matching these strict criteria. Adjust the market constraints above.")
    else:
        st.success(f"Discovered {len(df)} highly efficient, low-cost assets!")

        fig = px.scatter(df, x="Salary ($M)", y="Win Shares",
                         size="Usage Rate (%)", color="Age",
                         hover_name="Player",
                         title="Value Matrix: Production vs. Cost",
                         color_continuous_scale=px.colors.sequential.RdBu)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0.02)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Shortlisted Players")
        st.dataframe(df, use_container_width=True)


def main():
    """Main function that controls the flow of the Streamlit app."""
    # Apply custom NBA aesthetics
    inject_custom_css()

    # Instantiate the API connector
    data_api = DataManager()

    # Get navigation choice
    selected_page = render_sidebar()

    # Route to the correct page function
    if selected_page == "Home / Overview":
        page_home()
    elif selected_page == "Player Production":
        page_player_production(data_api)
    elif selected_page == "Draft Predictor War Room":
        page_draft_predictor(data_api)
    elif selected_page == "Hidden Gems Finder":
        page_hidden_gems(data_api)


if __name__ == "__main__":
    main()