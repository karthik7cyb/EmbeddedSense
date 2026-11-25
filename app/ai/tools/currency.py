from langchain_core.tools import tool

@tool("usd_to_inr", description="Converts USD to INR.")
def convert_currency(amount: float) -> str:
    """Convert USD to INR using a predefined exchange rate."""
    rate = 93.72
    return f"₹{amount * rate:.2f}"

print(convert_currency.invoke({"amount": 100}))