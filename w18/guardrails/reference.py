"""Reference policy the support bot must stay grounded in."""

REFERENCE = {
    "refunds": (
        "Customers may request a full refund within 30 days of purchase. "
        "After 30 days, refunds are not available. "
        "Refunds are issued to the original payment method within 5 business days."
    ),
    "support": (
        "Support is available Monday to Friday, 9am to 6pm UTC. "
        "Premium plan customers get 24/7 support. "
        "The support email is support@acmecloud.example."
    ),
    "pricing": (
        "The Basic plan costs $10 per month. "
        "The Premium plan costs $30 per month. "
        "Annual billing gives a 20% discount."
    ),
    "data": (
        "Customer data is stored in EU data centers. "
        "Data is encrypted at rest using AES-256. "
        "Accounts deleted by the customer are purged within 90 days."
    ),
}

CATEGORIES = list(REFERENCE) + ["other"]
