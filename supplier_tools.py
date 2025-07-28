"""Utility tools for supplier lookup and ordering."""

from typing import List, Dict


def web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Search the web for the query and return a list of results.

    Note: This is a placeholder implementation because the execution
    environment has no internet access. In a production environment,
    this function should call a real search API (e.g., Perplexity) and
    parse the response.
    """
    # Placeholder stub: return example results
    return [
        {"title": f"Result {i+1} for {query}", "url": f"https://example.com/{i}"}
        for i in range(num_results)
    ]


def find_suppliers(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Locate potential suppliers using the generic web_search function."""
    search_results = web_search(f"{query} wholesale supplier", num_results)
    suppliers = []
    for result in search_results:
        suppliers.append(
            {
                "name": result["title"],
                "website": result["url"],
            }
        )
    return suppliers


def order(
    supplier_email: str,
    item: str,
    quantity: int,
    shipping_address: str,
    account_info: Dict[str, str],
    *,
    smtp_server: str = "localhost",
    smtp_port: int = 25,
) -> None:
    """Send a purchase order email to a supplier.

    Parameters
    ----------
    supplier_email : str
        Email address of the supplier.
    item : str
        Item name to order.
    quantity : int
        Quantity of the item.
    shipping_address : str
        Destination address for the order.
    account_info : dict
        Dictionary containing at least ``username`` and ``password`` keys for
        SMTP authentication. Additional keys are ignored.
    smtp_server : str, optional
        SMTP server address. Defaults to ``localhost``.
    smtp_port : int, optional
        SMTP server port. Defaults to ``25``.
    """
    from email.message import EmailMessage
    import smtplib

    msg = EmailMessage()
    msg["Subject"] = f"Purchase Order: {item} (x{quantity})"
    msg["From"] = account_info.get("username", "")
    msg["To"] = supplier_email
    body = (
        f"Please supply {quantity} units of {item}.\n\n"
        f"Ship to:\n{shipping_address}\n"
    )
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as s:
            if account_info.get("username") and account_info.get("password"):
                s.login(account_info["username"], account_info["password"])
            s.send_message(msg)
    except Exception as exc:
        raise RuntimeError(f"Failed to send order email: {exc}")
