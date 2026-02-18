
from langchain_core.tools import tool
from rag import get_retriever


@tool
def get_flight_schedule(from_city: str, to_city: str):
    """Returns flight duration and price in USD for a given route."""
    return {
        "from": from_city,
        "to": to_city,
        "duration_hours": 5.5,
        "price_usd": 620
    }

@tool
def get_hotel_schedule(city: str):
    """Returns hotel options with price per night in USD for a given city."""
    return {
        "city": city,
        "hotels": [
            {"name": "Skyline Suites", "price_per_night_usd": 180},
            {"name": "Urban Comfort", "price_per_night_usd": 140}
        ]
    }


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Converts amount from one currency to another."""
    rates = {("USD", "NGN"): 1400}
    rate = rates.get((from_currency, to_currency), 1)
    return {
        "amount_converted": amount * rate,
        "currency": to_currency
    }


@tool
def query_internal_knowledge(question: str):
    """
    Fetch relevant documents for a question from the vector store.
    """
    retriever = get_retriever()

    docs = retriever.invoke(question)

    if not docs:
        return "No relevant information found."

    return {
    "relevant_documents": [
        doc.page_content for doc in docs
    ]
}

