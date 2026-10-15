from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import PydanticOutputParser , OutputFixingParser
from typing import Literal , List

from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()



class Ingredient(BaseModel):
    quantity:str = Field(description="the quantity of the ingredient like : '400g', '3 tbsp', 'a pinch' ")
    name:str = Field(description="the ingredient name, including any preparation notes like 'finely chopped'.")


class Recipe(BaseModel):
    title:str = Field(description="the title of the recipe")
    ingredients:List[Ingredient] = Field(description="the ingredients of the recipe")
    steps:List[str] = Field(description="the steps of the recipe")
    prep_time:int = Field(gt=1,description="the preparation time needed for the recipe in minutes")
    difficulty:Literal["easy","medium","hard"] = Field(description="the dificulty level of the recipe")

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")


parser = PydanticOutputParser(pydantic_object=Recipe)
# safe_parser = OutputFixingParser(parser=parser,llm=model)

prompt = PromptTemplate(
    template=(
       "You are a recipe extraction assistant.\n"
        "Extract the recipe from the text below.\n"
        "If a field is not present, make a reasonable inference.\n"
        "For ingredients, split quantity and item when possible.\n\n"
        "{format_instructions}\n\n"
        "Recipe text:\n\"\"\"\n{recipe_text}\n\"\"\"\n"
     ),
    input_variables=["recipe_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
) 

chain = prompt | model | parser

text_input = """So I made this amazing pasta last night! You'll need about 400g of
spaghetti, a good glug of olive oil (maybe 3 tbsp?), 4 cloves of garlic
finely chopped, a pinch of chili flakes, 2 cans of crushed tomatoes,
salt and pepper to taste, and a big handful of fresh basil. First boil
the pasta in salted water till al dente — about 9 minutes. Meanwhile heat
the oil in a pan and gently fry the garlic with chili until fragrant.
Add the tomatoes, simmer 15 minutes. Toss in the drained pasta, tear in
the basil, season, and serve. Takes about 25 minutes total, super easy!"""

result = chain.invoke({"recipe_text": text_input})

print(result)

