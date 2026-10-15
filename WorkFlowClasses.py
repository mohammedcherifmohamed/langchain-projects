from pydantic import BaseModel,Field
from typing import List , Literal


class ReviewAnalysis(BaseModel):
    rating:int = Field(ge=1, le=5, description="star rating 1-5. Infer from tone if not explicit")
    pros:List[str] = Field(description="Positive points the reviewer mentioned. Empty if none.")
    cons:List[str] = Field(description="Negative points the reviewer mentioned. Empty if none.")
    features_mentioned:List[str] = Field(description="Product features referenced, e.g. 'battery life', 'screen', 'shipping'")


class LossAnalysis(BaseModel):
    risk_level:Literal["low","medium","high",] = Field(description="How likely this reviewer is to stop using the product.")
    main_reason: str = Field(description="The single biggest reason for dissatisfaction.")
    retention_suggestion: str = Field(description="One concrete thing the company could do to retain them.")

class AdvocacyAnalysis(BaseModel):
    loyalty_level:Literal["neutral","promoter","superFan"] = Field(description="How likely this reviewer is to recommend the product to others.")
    main_reason: str = Field(description="The single biggest reason for satisfaction.")
    amplification_idea: str = Field(
        description="One way to turn this reviewer into a public advocate (referral, testimonial, etc.)."
    )

class ActionItem(BaseModel):
    priority: Literal["P0", "P1", "P2"] = Field(description="P0=urgent, P1=soon, P2=backlog.")
    action: str = Field(description="ONE sentence telling the PM what to do next.")
    owner: Literal["product", "engineering", "marketing", "support"] = Field(
        description="Which team should own this."
    )


