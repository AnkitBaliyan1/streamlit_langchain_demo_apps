import streamlit as st
import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import LLMRequestsChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain

def single_chain(place):
    llm = OpenAI()
    prompt = PromptTemplate(
        input_variables=["place"],
        template="5 Best place to visit in {place}? \
            Only provide the name of place and not any description."
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    answer = chain.run(place)
    return answer

def simple_sequential_chain(place):
    llm_1 = OpenAI()
    template_1 = """You have to suggest 5 best places to visit in {place}. 
    Do not provide any details but just the name of place to visit.
    Your Resposne: 
    """
    prompt_1 = PromptTemplate(
    input_variables=["place"],
    template=template_1
    )
    chain_1 = LLMChain(llm=llm_1, prompt = prompt_1, output_key = "best_places", verbose=True)
    
    

    llm_2 = OpenAI()
    template_2 = """Given a list a places, please estimate the expenses to visit all of them in local currency and also the days needed
        {expenses}

        YOUR RESPONSE:
        """
    prompt_2 = PromptTemplate(
            input_variables=["expense"],
            template=template_2
            )

    chain_2 = LLMChain(llm = llm_2, prompt = prompt_2, verbose=True)


    main_chain = SimpleSequentialChain(chains = [chain_1, chain_2], verbose=True)

    answer = main_chain.run(place)

    st.success(answer)

def summarize_chain(data):
    llm = OpenAI(temperature=0.7)

    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(data)

    docs = [Document(page_content=t) for t in texts]

    chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)

    result = chain.run(docs)

    st.write(result)

def request_chain(query):
    template = """
        Extract the answer of the question '{query}' or say "not found" if the information is not available. 
        {requests_result}
        """
    prompt = PromptTemplate(
            input_variables = ["query","requests_result"],
            template = template,
        )
    print("prompt:",prompt)
    llm=OpenAI()
    chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt, verbose=True))

    inputs = {
        "query":query,
        "url":"https://www.google.com/search?q="+ query.replace(" ","+"),
        }

    answer = chain(inputs)

    st.success(answer["output"])


def complex_chain(input):
    llm = OpenAI(temperature=0.7)

    # prompt template 1: translate to English
    first_prompt = ChatPromptTemplate.from_template(
        "Translate the following review to english:"
        "\n\n{Review}"
    )

    # chain 1: input= Review and output= English_Review

    chain_one = LLMChain(llm=llm, prompt=first_prompt,
                        output_key="English_Review",verbose=True                     
                        )
    
    # prompt template 2: Summarize the English review

    second_prompt = ChatPromptTemplate.from_template(
        "Can you summarize the following review in 1 sentence of 10 words:"
        "\n\n{English_Review}"
    )

    # chain 2: input= English_Review and output= summary

    chain_two = LLMChain(llm=llm, prompt=second_prompt,
                        output_key="summary",verbose=True
                        )
    
    # prompt template 3: translate to English

    third_prompt = ChatPromptTemplate.from_template(
        "What language is the following review:\n\n{Review}"
    )

    # chain 3: input= Review and output= language

    chain_three = LLMChain(llm=llm, prompt=third_prompt,
                        output_key="language",verbose=True
                        )
    
    # prompt template 4: follow up message

    fourth_prompt = ChatPromptTemplate.from_template(
        "Write a follow up response to the following "
        "summary in the specified language:"
        "\n\nSummary: {summary}\n\nLanguage: {language}"
    )

    # chain 4: input= summary, language and output= followup_message

    chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                        output_key="followup_message",
                        verbose=True
                        )


    overall_chain = SequentialChain(
                        chains=[chain_one, chain_two, chain_three, chain_four],
                        input_variables=["Review"],
                        output_variables=["English_Review", "summary","followup_message"],
                        verbose=True
                    )
    
    answer = overall_chain(input)

    st.success(answer["followup_message"])



st.title("LangChain Demo Apps")

with st.sidebar:
    mode = st.radio("Select Your App:", ["Single Chain",
                                         "Sequential Chain",
                                         "Summarize Text",
                                         "Internet Search with LangChain",
                                         "Complex chain at the back end"])

if mode == "Single Chain":
    st.markdown(
        """
        This is a demo for single chain.\
            This takes the name of place as input and gives up the best places to visit in the given area.
        """
    )
    user_input_single_chain = st.text_input("Enter the place here.")
    if st.button("Get Resposne"):
        if user_input_single_chain:
            answer = single_chain(place=user_input_single_chain)
            st.success(answer)
        else:
            st.error("Enter the name of place first.")
elif mode == "Sequential Chain":
    st.markdown(
        """
            This is a demo for sequential chain.
            This takes the a name of place from user and generate the 5 best place to visit.
            Further, the name of place is provided to the second chain which than estimates of expences and time to visit all the places. 
        """
        )
    user_input_sequential_chain = st.text_input("Enter the place here:")
    if st.button("Get Resposne from Sequential Chain"):
        if user_input_sequential_chain:
            simple_sequential_chain(place = user_input_sequential_chain)
            
        else:
            st.error("Enter name of place first.")

elif mode == "Summarize Text":
    st.markdown("This is to summarize the text provided by the user.\
                This is using \"`load_summarise_chain`\" from `Langchain Utility Chains`.\
                The chain type is `Map Reduce` and the model used is `OpenAI`")
    user_input = st.text_area("Enter the text below.")
    
    if st.button("Summarize text"):
        if user_input:
            summarize_chain(user_input)
        else:
            st.error("No text found. Enter the text above.")


elif mode == "Internet Search with LangChain":
    st.markdown("\
                This is \"`LLMRequestChain`\" which takes the user input and search it on the given URL. \n \
                For this use case, the search is taking place on \"`google.com`\" to generate the response. \n \
                It is using the `OpenAI` model to generate the response.")

    question = st.text_input("Enter your question here.")
    if st.button("Get Response"):
        if question:
            request_chain(question)
        else:
            st.error("No question found. Enter question above")
    
elif mode=="Complex chain at the back end":
    input = st.text_area("Enter the Movie Review here.")
    if st.button("Get Response"):
        if input:
            print("running chain")
            complex_chain(input)
        else:
            st.error("No input found. Enter the review in any language above")
