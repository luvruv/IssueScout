from pydantic import BaseModel, Field


class Profile(BaseModel):
    skills: list[str] = Field(default_factory=list)
    level: str = "beginner"              # beginner | intermediate | advanced
    hours_per_week: float = 6
    weeks: float = 3
    k: int = 3
    avoid: list[str] = Field(default_factory=list)
    use_llm: bool = False
    include_uncertain: bool = True       # False hides issues where the model's top probability is under 50%