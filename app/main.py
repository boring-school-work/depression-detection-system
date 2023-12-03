import streamlit as st
from google.oauth2 import service_account
import vertexai
from vertexai.language_models import TextGenerationModel


service_account_info = st.secrets["SERVICE_ACCOUNT_INFO"]

# Load the credentials from the JSON key file
credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)


vertexai.init(project="130460717287", location="us-central1", credentials=credentials)
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40,
}
model = TextGenerationModel.from_pretrained("text-bison@001")
model = model.get_tuned_model(
    "projects/130460717287/locations/us-central1/models/304078736754475008"
)


def predict(prompt):
    response = model.predict(
        f"""Analyse the input text and determine whether the person is depressed or not.

    input: wednesday my b day don t know what do
    output: no

    input: Depression sucks especially accompanied by insomnia and chronic pain chronic life.
    output: yes

    input: {prompt}
    output:
    """,
        **parameters,
    )
    return response.text


def display(status):
    if status == "yes":
        st.error("You most likely depressed :(")
    else:
        st.success("Your mind is healthy :)")


st.title("Depression Prediction System")
st.write("by: [David Saah](https://github.com/davesaah)")

st.subheader("Accuracy metrics", divider="rainbow")
col1, col2, col3, col4 = st.columns(spec=[0.5, 0.5, 0.5, 0.5])
col1.metric("AuPRC score", "0.83")
col2.metric("Recall", "0.96")
col3.metric("Precision", "0.84")
col4.metric("F1 Score", "0.90", "0.6")
st.subheader("", divider="rainbow")

response = st.text_area(label="What's on your mind", placeholder="max 1024 characters.")


if st.button("Am I depressed?"):
    display(predict(response))
