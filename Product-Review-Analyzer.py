from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch , RunnablePassthrough



from dotenv import load_dotenv

from WorkFlowClasses import ReviewAnalysis, LossAnalysis, AdvocacyAnalysis, ActionItem

load_dotenv()


def to_dict(review: ReviewAnalysis) -> dict :
    return{
        "rating" : review.rating,
        "pros" : ", ".join(review.pros) or "none",
        "cons" : ", ".join(review.cons) or "none",
        "features" : ", ".join(review.features_mentioned) or "none",
    }

def merge_for_action(payload: dict) -> dict:
    review = payload["review_obj"]      # the ReviewAnalysis
    analysis = payload["analysis"]  # Churn or Advocacy
    return {
        "rating": review.rating,
        "analysis_type": analysis.__class__.__name__,   # "ChurnAnalysis" or "AdvocacyAnalysis"
        "analysis": analysis.model_dump_json(),
    }

model = ChatGoogleGenerativeAI(model="gemini-3.5²-flash")

# ______________ EXTTRACT CHAIN _______________

extract_parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
extract_prompt = PromptTemplate(
    template=(
     "You are a product review analyzer.\n"
        "Extract structured data from the review below.\n"
        "If rating is not explicit, infer it from tone.\n\n"
        "{format_instructions}\n\n"
        "Review:\n\"\"\"\n{review}\n\"\"\"\n"
),
input_variables=["review"],
partial_variables={"format_instructions":extract_parser.get_format_instructions()},
)

extract_chain = extract_prompt | model | extract_parser

# review_text = "It's okay. Nothing special. Works as expected."
# review_text = "Perfect product,i liked the case made with good material and nice design but the screen was small and not very power light under the sun light ."
# result = extract_chain.invoke({"review": review_text})

# ___________________ Loss CHain ____________________

Loss_parser = PydanticOutputParser(pydantic_object=LossAnalysis)
Loss_prompt = PromptTemplate(
    template=(
        "This customer gave a LOW rating. Analyze Loss risk.\n\n"
        "Rating: {rating}\n"
        "Pros: {pros}\n"
        "Cons: {cons}\n"
        "Features mentioned: {features}\n\n"
        "{format_instructions}\n"
    ),
    input_variables=["rating", "pros", "cons", "features"],
    partial_variables={"format_instructions": Loss_parser.get_format_instructions()},
)
Loss_chain = Loss_prompt | model | Loss_parser

# ____________ ADVOCACY PART ______________________
advocacy_parser = PydanticOutputParser(pydantic_object=AdvocacyAnalysis)
advocacy_prompt = PromptTemplate(
    template=(
        "This customer gave a HIGH rating. Analyze advocacy potential.\n\n"
        "Rating: {rating}\n"
        "Pros: {pros}\n"
        "Cons: {cons}\n"
        "Features mentioned: {features}\n\n"
        "{format_instructions}\n"
    ),
    input_variables=["rating", "pros", "cons", "features"],
    partial_variables={"format_instructions": advocacy_parser.get_format_instructions()},
)
advocacy_chain = advocacy_prompt | model | advocacy_parser


to_prompt_input = RunnableLambda(to_dict)

branch = RunnableBranch(
    (lambda X: X.rating < 3, to_prompt_input | Loss_chain),
    to_prompt_input | advocacy_chain,
)


action_parser = PydanticOutputParser(pydantic_object=ActionItem)
action_prompt = PromptTemplate(
    template=(
        "Based on this customer analysis, write ONE action item for the product team.\n\n"
        "Original rating: {rating}\n"
        "Analysis type: {analysis_type}\n"
        "Analysis: {analysis}\n\n"
        "{format_instructions}\n"
    ),
    input_variables=["rating", "analysis_type", "analysis"],
    partial_variables={"format_instructions": action_parser.get_format_instructions()},
)
action_chain = action_prompt | model | action_parser



step1 = RunnablePassthrough.assign(review_obj=extract_chain)

# Step 2: run the branch on the extracted object
def run_branch(payload):
    return {
        "review_obj": payload["review_obj"],
        "analysis": branch.invoke(payload["review_obj"]),
    }
step2 = RunnableLambda(run_branch)

# Step 3: merge + generate action item
step3 = RunnableLambda(merge_for_action) | action_chain

# The full pipeline
full_chain = step1 | step2 | step3


review_text = "Battery dies in 2 hours. Support ghosted me. 1 star."
result = full_chain.invoke({"review": review_text})
print(result.action)


# print("Type:", type(result).__name__)
# print("Rating:", result.rating)
# print("Pros:", result.pros)
# print("Cons:", result.cons)
# print("Features:", result.features_mentioned)










