# src/features/pydantic_output.py

import json
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from src.ai_client import ask_ai
from src.utils.printer import (
    print_feature_header,
    print_concept,
    print_response,
    print_step,
    print_thinking,
    print_info,
    print_divider,
    print_prompt,
    print_success,
    print_error,
    print_table
)
from src.utils.menu import get_user_input


# ✅ Define Pydantic Models
# These are our data structures

class PersonModel(BaseModel):
    """Model for person information"""
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years", ge=0, le=150)
    job: str = Field(description="Job title or occupation")
    email: Optional[str] = Field(
        default=None,
        description="Email address if available"
    )
    skills: List[str] = Field(
        default=[],
        description="List of skills"
    )


class ProductModel(BaseModel):
    """Model for product information"""
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD", ge=0)
    category: str = Field(description="Product category")
    rating: float = Field(
        description="Rating from 1-5",
        ge=1.0,
        le=5.0
    )
    in_stock: bool = Field(description="Whether product is in stock")
    features: List[str] = Field(
        default=[],
        description="List of key features"
    )


class RecipeModel(BaseModel):
    """Model for recipe information"""
    name: str = Field(description="Recipe name")
    cuisine: str = Field(description="Type of cuisine")
    prep_time: int = Field(
        description="Preparation time in minutes",
        ge=0
    )
    cook_time: int = Field(
        description="Cooking time in minutes",
        ge=0
    )
    difficulty: str = Field(
        description="Easy, Medium, or Hard"
    )
    servings: int = Field(
        description="Number of servings",
        ge=1
    )
    ingredients: List[str] = Field(
        description="List of ingredients"
    )
    steps: List[str] = Field(
        description="Cooking steps in order"
    )
    calories: Optional[int] = Field(
        default=None,
        description="Calories per serving"
    )


class MovieReviewModel(BaseModel):
    """Model for movie review"""
    movie_title: str = Field(description="Name of the movie")
    genre: str = Field(description="Movie genre")
    rating: float = Field(
        description="Rating out of 10",
        ge=0,
        le=10
    )
    sentiment: str = Field(
        description="POSITIVE, NEGATIVE, or MIXED"
    )
    pros: List[str] = Field(description="Good points")
    cons: List[str] = Field(description="Bad points")
    summary: str = Field(
        description="One sentence summary"
    )
    recommended: bool = Field(
        description="Whether to recommend"
    )


class TravelPlanModel(BaseModel):
    """Model for travel plan"""
    destination: str = Field(description="Travel destination")
    duration_days: int = Field(
        description="Number of days",
        ge=1
    )
    budget_usd: float = Field(
        description="Total budget in USD",
        ge=0
    )
    best_season: str = Field(
        description="Best season to visit"
    )
    must_visit: List[str] = Field(
        description="Must visit places"
    )
    local_foods: List[str] = Field(
        description="Must try local foods"
    )
    tips: List[str] = Field(
        description="Travel tips"
    )
    difficulty: str = Field(
        description="Easy, Moderate, or Challenging"
    )


def get_structured_output(
    prompt: str,
    model_class: BaseModel,
    system: str = "You are a helpful assistant."
) -> tuple:
    """
    Ask AI and parse response into
    a Pydantic model

    Returns: (parsed_model, raw_response, error)
    """

    # ✅ Build schema prompt
    schema = model_class.model_json_schema()
    schema_str = json.dumps(schema, indent=2)

    structured_prompt = f"""
{prompt}

You MUST respond with valid JSON that matches 
this exact schema:

{schema_str}

Rules:
- Return ONLY valid JSON
- No extra text before or after JSON
- All required fields must be present
- Follow the exact field types
"""

    raw_response = ask_ai(
        prompt=structured_prompt,
        system=system,
        temperature=0.2
    )

    # ✅ Clean response
    clean_response = raw_response.strip()
    if "```" in clean_response:
        parts = clean_response.split("```")
        clean_response = parts[1] if len(parts) > 1 else clean_response
        if clean_response.startswith("json"):
            clean_response = clean_response[4:]

    # ✅ Parse and validate with Pydantic
    try:
        data = json.loads(clean_response)
        validated = model_class(**data)
        return validated, raw_response, None
    except json.JSONDecodeError as e:
        return None, raw_response, f"JSON Error: {e}"
    except Exception as e:
        return None, raw_response, f"Validation Error: {e}"


def display_person(person: PersonModel):
    """Display person model nicely"""
    print_table(
        title="👤 Person Information",
        columns=["Field", "Value"],
        rows=[
            ["Name", person.name],
            ["Age", str(person.age)],
            ["Job", person.job],
            ["Email", person.email or "N/A"],
            ["Skills", ", ".join(person.skills)],
        ]
    )


def display_product(product: ProductModel):
    """Display product model nicely"""
    print_table(
        title="🛍️ Product Information",
        columns=["Field", "Value"],
        rows=[
            ["Name", product.name],
            ["Price", f"${product.price}"],
            ["Category", product.category],
            ["Rating", f"⭐ {product.rating}/5"],
            ["In Stock", "✅ Yes" if product.in_stock else "❌ No"],
            ["Features", "\n".join(product.features)],
        ]
    )


def display_recipe(recipe: RecipeModel):
    """Display recipe model nicely"""
    print_table(
        title="🍳 Recipe Information",
        columns=["Field", "Value"],
        rows=[
            ["Name", recipe.name],
            ["Cuisine", recipe.cuisine],
            ["Prep Time", f"{recipe.prep_time} mins"],
            ["Cook Time", f"{recipe.cook_time} mins"],
            ["Difficulty", recipe.difficulty],
            ["Servings", str(recipe.servings)],
            ["Calories", str(recipe.calories or "N/A")],
            ["Ingredients", str(len(recipe.ingredients))],
            ["Steps", str(len(recipe.steps))],
        ]
    )

    print_info(
        "Ingredients:\n" +
        "\n".join([f"  • {i}" for i in recipe.ingredients])
    )

    print_info(
        "Steps:\n" +
        "\n".join([
            f"  {i+1}. {s}"
            for i, s in enumerate(recipe.steps)
        ])
    )


def display_movie_review(review: MovieReviewModel):
    """Display movie review nicely"""
    print_table(
        title="🎬 Movie Review",
        columns=["Field", "Value"],
        rows=[
            ["Movie", review.movie_title],
            ["Genre", review.genre],
            ["Rating", f"⭐ {review.rating}/10"],
            ["Sentiment", review.sentiment],
            ["Recommended", "✅ Yes" if review.recommended else "❌ No"],
            ["Summary", review.summary],
        ]
    )

    print_info(
        "Pros:\n" +
        "\n".join([f"  ✅ {p}" for p in review.pros])
    )

    print_info(
        "Cons:\n" +
        "\n".join([f"  ❌ {c}" for c in review.cons])
    )


def display_travel(plan: TravelPlanModel):
    """Display travel plan nicely"""
    print_table(
        title="✈️ Travel Plan",
        columns=["Field", "Value"],
        rows=[
            ["Destination", plan.destination],
            ["Duration", f"{plan.duration_days} days"],
            ["Budget", f"${plan.budget_usd}"],
            ["Best Season", plan.best_season],
            ["Difficulty", plan.difficulty],
        ]
    )

    print_info(
        "Must Visit:\n" +
        "\n".join([f"  📍 {p}" for p in plan.must_visit])
    )

    print_info(
        "Local Foods:\n" +
        "\n".join([f"  🍜 {f}" for f in plan.local_foods])
    )

    print_info(
        "Travel Tips:\n" +
        "\n".join([f"  💡 {t}" for t in plan.tips])
    )


def run():
    """Pydantic Instructor Feature"""

    # ✅ Header
    print_feature_header("Pydantic Instructor")

    # ✅ Explain concept
    print_concept(
        "What is Pydantic Instructor?",
        "Getting AI to return STRUCTURED & VALIDATED data!\n\n"
        "Problem without it:\n"
        "  AI returns random text format\n"
        "  Hard to use in real applications\n"
        "  No type validation\n"
        "  Missing fields possible\n\n"
        "Solution with Pydantic:\n"
        "  ✅ Define exact data structure\n"
        "  ✅ AI fills the structure\n"
        "  ✅ Pydantic validates the data\n"
        "  ✅ Ready to use in your app!\n\n"
        "Models we will demo:\n"
        "  👤 PersonModel\n"
        "  🛍️  ProductModel\n"
        "  🍳 RecipeModel\n"
        "  🎬 MovieReviewModel\n"
        "  ✈️  TravelPlanModel"
    )

    print_divider()

    # ✅ Demo 1: Person Model
    print_step(
        "Demo 1",
        "Extract Person Info → PersonModel"
    )

    person_text = (
        "Tell me about a fictional software engineer "
        "named Alex Kumar who is 28 years old, "
        "works at a startup, and knows Python, "
        "JavaScript, and Swift."
    )

    print_prompt(
        f"Request: {person_text}\n\n"
        f"Schema:\n"
        f"  name: str\n"
        f"  age: int (0-150)\n"
        f"  job: str\n"
        f"  email: Optional[str]\n"
        f"  skills: List[str]"
    )

    print_thinking()

    person, _, error = get_structured_output(
        prompt=person_text,
        model_class=PersonModel,
        system="You are a helpful assistant that creates fictional profiles."
    )

    if error:
        print_error(f"Validation failed: {error}")
    else:
        print_success("✅ Pydantic validation passed!")
        display_person(person)

        print_info(
            "Validated fields:\n"
            f"  name: {type(person.name).__name__} ✅\n"
            f"  age: {type(person.age).__name__} ✅\n"
            f"  job: {type(person.job).__name__} ✅\n"
            f"  skills: {type(person.skills).__name__} ✅"
        )

    print_divider()

    # ✅ Demo 2: Product Model
    print_step(
        "Demo 2",
        "Generate Product Info → ProductModel"
    )

    product_text = (
        "Create a fictional wireless headphone product "
        "with realistic price, rating, features, "
        "and availability details."
    )

    print_prompt(
        f"Request: {product_text}\n\n"
        f"Schema:\n"
        f"  name: str\n"
        f"  price: float (>=0)\n"
        f"  category: str\n"
        f"  rating: float (1-5)\n"
        f"  in_stock: bool\n"
        f"  features: List[str]"
    )

    print_thinking()

    product, _, error = get_structured_output(
        prompt=product_text,
        model_class=ProductModel,
        system="You are a product catalog generator."
    )

    if error:
        print_error(f"Validation failed: {error}")
    else:
        print_success("✅ Pydantic validation passed!")
        display_product(product)

    print_divider()

    # ✅ Demo 3: Recipe Model
    print_step(
        "Demo 3",
        "Generate Recipe → RecipeModel"
    )

    recipe_text = (
        "Create a detailed recipe for "
        "a classic Italian pasta carbonara."
    )

    print_prompt(
        f"Request: {recipe_text}\n\n"
        f"Schema:\n"
        f"  name, cuisine, prep_time\n"
        f"  cook_time, difficulty\n"
        f"  servings, ingredients[]\n"
        f"  steps[], calories"
    )

    print_thinking()

    recipe, _, error = get_structured_output(
        prompt=recipe_text,
        model_class=RecipeModel,
        system="You are a professional chef and recipe writer."
    )

    if error:
        print_error(f"Validation failed: {error}")
    else:
        print_success("✅ Pydantic validation passed!")
        display_recipe(recipe)

    print_divider()

    # ✅ Demo 4: Movie Review Model
    print_step(
        "Demo 4",
        "Analyze Movie Review → MovieReviewModel"
    )

    movie_text = (
        "Write a detailed review for the movie "
        "Inception by Christopher Nolan."
    )

    print_prompt(
        f"Request: {movie_text}\n\n"
        f"Schema:\n"
        f"  movie_title, genre\n"
        f"  rating (0-10)\n"
        f"  sentiment (POSITIVE/NEGATIVE/MIXED)\n"
        f"  pros[], cons[]\n"
        f"  summary, recommended"
    )

    print_thinking()

    review, _, error = get_structured_output(
        prompt=movie_text,
        model_class=MovieReviewModel,
        system="You are an expert movie critic."
    )

    if error:
        print_error(f"Validation failed: {error}")
    else:
        print_success("✅ Pydantic validation passed!")
        display_movie_review(review)

    print_divider()

    # ✅ Demo 5: Travel Plan Model
    print_step(
        "Demo 5",
        "Generate Travel Plan → TravelPlanModel"
    )

    travel_text = (
        "Create a detailed travel plan "
        "for visiting Japan for 7 days "
        "with a budget of $3000."
    )

    print_prompt(
        f"Request: {travel_text}\n\n"
        f"Schema:\n"
        f"  destination, duration_days\n"
        f"  budget_usd, best_season\n"
        f"  must_visit[], local_foods[]\n"
        f"  tips[], difficulty"
    )

    print_thinking()

    travel, _, error = get_structured_output(
        prompt=travel_text,
        model_class=TravelPlanModel,
        system="You are an expert travel planner."
    )

    if error:
        print_error(f"Validation failed: {error}")
    else:
        print_success("✅ Pydantic validation passed!")
        display_travel(travel)

    print_divider()

    # ✅ Demo 6: Validation Error Demo
    print_step(
        "Demo 6",
        "Validation Power — Catching wrong data!"
    )

    print_info(
        "Pydantic catches invalid data!\n\n"
        "Examples of what gets caught:\n"
        "  ❌ age = -5 (must be >= 0)\n"
        "  ❌ rating = 99 (must be 1-5)\n"
        "  ❌ price = -100 (must be >= 0)\n"
        "  ❌ name = None (required field)\n"
        "  ❌ skills = 'python' (must be List)"
    )

    # ✅ Show validation errors manually
    test_cases = [
        {
            "data": {
                "name": "John",
                "age": -5,
                "job": "Developer",
                "skills": ["Python"]
            },
            "expected_error": "age must be >= 0"
        },
        {
            "data": {
                "name": "Jane",
                "age": 25,
                "job": "Designer",
                "skills": "Figma"
            },
            "expected_error": "skills must be a list"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print_step(
            f"Test {i}",
            f"Expected error: {test['expected_error']}"
        )

        try:
            PersonModel(**test["data"])
            print_error("No error caught (unexpected!)")
        except Exception as e:
            print_success(
                f"✅ Pydantic caught the error!\n"
                f"Error: {str(e)[:200]}"
            )

    print_divider()

    # ✅ Demo 7: Custom request
    print_step(
        "Demo 7",
        "Try YOUR own structured output!"
    )

    print_table(
        title="Choose a Model",
        columns=["Option", "Model", "Use Case"],
        rows=[
            ["1", "PersonModel", "Profile extraction"],
            ["2", "ProductModel", "Product generation"],
            ["3", "RecipeModel", "Recipe creation"],
            ["4", "MovieReviewModel", "Review analysis"],
            ["5", "TravelPlanModel", "Travel planning"],
        ]
    )

    model_choice = get_user_input(
        "Choose model (1-5): "
    )

    custom_prompt = get_user_input(
        "📝 Enter your request: "
    )

    if not custom_prompt:
        custom_prompt = (
            "Create info for a fictional "
            "data scientist named Maya"
        )
        print_info(f"Using default: {custom_prompt}")
        model_choice = "1"

    print_prompt(f"Your Request: {custom_prompt}")
    print_thinking()

    # ✅ Route to correct model
    if model_choice == "1":
        result, _, error = get_structured_output(
            prompt=custom_prompt,
            model_class=PersonModel
        )
        if not error:
            display_person(result)

    elif model_choice == "2":
        result, _, error = get_structured_output(
            prompt=custom_prompt,
            model_class=ProductModel
        )
        if not error:
            display_product(result)

    elif model_choice == "3":
        result, _, error = get_structured_output(
            prompt=custom_prompt,
            model_class=RecipeModel
        )
        if not error:
            display_recipe(result)

    elif model_choice == "4":
        result, _, error = get_structured_output(
            prompt=custom_prompt,
            model_class=MovieReviewModel
        )
        if not error:
            display_movie_review(result)

    elif model_choice == "5":
        result, _, error = get_structured_output(
            prompt=custom_prompt,
            model_class=TravelPlanModel
        )
        if not error:
            display_travel(result)

    else:
        result, _, error = get_structured_output(
            prompt=custom_prompt,
            model_class=PersonModel
        )
        if not error:
            display_person(result)

    if error:
        print_error(f"Error: {error}")
    else:
        print_success(
            "✅ Structured output generated & validated!"
        )

    print_divider()

    # ✅ Summary
    print_concept(
        "Pydantic Instructor Summary",
        "Key Takeaways:\n\n"
        "Without Pydantic:\n"
        "  ❌ AI returns random text\n"
        "  ❌ No type safety\n"
        "  ❌ Missing fields possible\n"
        "  ❌ Hard to use in apps\n\n"
        "With Pydantic:\n"
        "  ✅ Structured JSON output\n"
        "  ✅ Type validated fields\n"
        "  ✅ Required fields enforced\n"
        "  ✅ Value ranges validated\n"
        "  ✅ Ready to use in code\n\n"
        "Best Practices:\n"
        "  ✅ Define clear field descriptions\n"
        "  ✅ Add validators for ranges\n"
        "  ✅ Use Optional for non-required\n"
        "  ✅ Keep models focused\n\n"
        "Real World Uses:\n"
        "  ✅ API response parsing\n"
        "  ✅ Data extraction pipelines\n"
        "  ✅ Content generation systems\n"
        "  ✅ Database record creation\n"
        "  ✅ Form auto-filling"
    )