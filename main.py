import pandas as pd # Pandas
import streamlit as st # User Interface
from langchain_experimental.agents import create_pandas_dataframe_agent # agent lib
# from langchain.agents import AgentType
from langchain_groq import ChatGroq  # Langchain Groq


GROQ_API_KEY = "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"   


# // StreamLit config
st.set_page_config(page_icon="👋🏻", page_title="Stats Bot Chat", layout="wide")


# // Read File
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


# // loading llms
llm = ChatGroq(model="llama3-70b-8192",api_key=GROQ_API_KEY, temperature=1)


if "df" not in st.session_state:
    st.session_state.df = None



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


center, right = st.columns([2, 1], gap="medium")


with right:
    box = st.container(border=True)
    box.subheader("Data Preview")

    uploaded_file = box.file_uploader("file upload", type=["csv", "xlsx", "xls"])



    # // file uploaded convert dataframe

    if uploaded_file:
        data = load_data(uploaded_file)
        st.session_state.df = data

        box.write("Your Data Preview")

        box.dataframe(st.session_state.df.head())

        # calling agents
        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df=st.session_state.df,
            verbose=True,
            agent_type="tool-calling",
            allow_dangerous_code=True,
        )
# ---------------------------------------------------------------------
# // Ui page Design
with center:
    box = st.container()
    box.title("📈 Stats Chat")
    box.markdown(
        """Explore data and find 
    meaningful information with it , automating whole  <br> data science EDA (Exploratory Data Analysis) process. """,
        unsafe_allow_html=True,
    )

    messages = box.container(height=400)

    # display chat messagges
    for msg in st.session_state.chat_history:
        with messages.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = box.chat_input("Ask About Data...")

    if user_prompt:

        if st.session_state.df is None:
            st.toast("Please Add data first !!")
        else:
            messages.chat_message("user").write(user_prompt)

            st.session_state.chat_history.append(
                {"role": "user", "content": user_prompt}
            )

            description = """
            your name is Stats bot,
            you are helpful bot give me insight about data based on my question in well strutured format
            like charts , markdown , tables etc.,you can use markdown tables without any code snippets
            """

            allmessages = [
                {"role": "system", "content": description},
                *st.session_state.chat_history,
            ]

            response = pandas_df_agent.invoke(allmessages)

            print(response["output"])

            assitant_response = response["output"]

            st.session_state.chat_history.append(
                {"role": "assistant", "content": assitant_response}
            )

            messages.chat_message("assistant").markdown(assitant_response) # ui of assistant response


